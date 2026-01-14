#include "UrdfParser.h"
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <vector>
#include <cctype>
#include <fstream>
#include <cstdint>

namespace dyno
{
    std::string processMeshPath(const std::string& originalPath) {
        std::string processedPath = originalPath;
        
        // 移除"package://"前缀
        const std::string prefix = "package://";
        if (processedPath.size() >= prefix.size() && 
            processedPath.substr(0, prefix.size()) == prefix) {
            processedPath = processedPath.substr(prefix.size());
        }
        
        // 2. 将".dae"扩展名替换为".obj"
        const std::string daeExt = ".dae";
        if (processedPath.size() >= daeExt.size() && 
            processedPath.substr(processedPath.size() - daeExt.size()) == daeExt) {
            processedPath.replace(processedPath.size() - daeExt.size(), daeExt.size(), ".obj");
        }
        
        return processedPath;
    }

    static std::vector<int> parseIntListAttr(const char* attr)
    {
        std::vector<int> out;
        if (!attr) return out;

        std::string s(attr);
        for (char& c : s)
        {
            const bool ok = (std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+');
            if (!ok) c = ' ';
        }

        std::stringstream ss(s);
        int v = 0;
        while (ss >> v)
        {
            out.push_back(v);
        }
        return out;
    }

    static std::string getDirectoryName(const std::string& path)
    {
        const size_t pos = path.find_last_of("/\\");
        if (pos == std::string::npos) return ".";
        return path.substr(0, pos);
    }

    static bool isAbsolutePath(const std::string& p)
    {
        if (p.empty()) return false;
        if (p[0] == '/' || p[0] == '\\') return true;
        // Windows: "C:\..."
        if (p.size() >= 2 && std::isalpha(static_cast<unsigned char>(p[0])) && p[1] == ':') return true;
        return false;
    }

    static std::string joinPath(const std::string& dir, const std::string& file)
    {
        if (dir.empty() || dir == ".") return file;
        if (dir.back() == '/' || dir.back() == '\\') return dir + file;
        return dir + "/" + file;
    }

    static bool readPatchBin(const std::string& filepath,
                             std::vector<int>& outFaces,
                             std::vector<int>& outOffsets,
                             uint32_t* outVersion = nullptr)
    {
        outFaces.clear();
        outOffsets.clear();

        std::ifstream is(filepath, std::ios::binary);
        if (!is.is_open()) return false;

        uint32_t magic = 0;
        uint32_t ver = 0;
        uint32_t numFaces = 0;
        uint32_t numOffsets = 0;

        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&ver), sizeof(ver));
        is.read(reinterpret_cast<char*>(&numFaces), sizeof(numFaces));
        is.read(reinterpret_cast<char*>(&numOffsets), sizeof(numOffsets));

        if (!is.good()) return false;

        // 'PTCH' little-endian (与 Writer 中 0x48435450u 对齐)
        if (magic != 0x48435450u) return false;

        if (outVersion) *outVersion = ver;

        // 简单 sanity check，避免异常文件导致超大分配
        if (numOffsets < 2) return false;
        if (numFaces > 200000000u || numOffsets > 200000000u) return false;

        std::vector<int32_t> offsets32(numOffsets);
        std::vector<int32_t> faces32(numFaces);

        is.read(reinterpret_cast<char*>(offsets32.data()), sizeof(int32_t) * offsets32.size());
        is.read(reinterpret_cast<char*>(faces32.data()), sizeof(int32_t) * faces32.size());

        if (!is.good()) return false;

        outOffsets.resize(numOffsets);
        for (size_t i = 0; i < offsets32.size(); ++i) outOffsets[i] = static_cast<int>(offsets32[i]);

        outFaces.resize(numFaces);
        for (size_t i = 0; i < faces32.size(); ++i) outFaces[i] = static_cast<int>(faces32[i]);

        // 基本一致性检查（失败则认为读取失败，方便上层重新 patch）
        if (outOffsets.front() != 0) return false;
        if (outOffsets.back() != static_cast<int>(outFaces.size())) return false;

        return true;
    }

    bool UrdfParser::parse(const std::string& filePath, UrdfInformation& urdfInfo, bool objYUp)
    {
        urdfInfo.links.clear();
        urdfInfo.joints.clear();
        urdfInfo.robotName.clear();

        tinyxml2::XMLDocument doc;
        tinyxml2::XMLError error = doc.LoadFile(filePath.c_str());
        if (error != tinyxml2::XML_SUCCESS)
        {
            std::cerr << "Failed to load URDF file: " << filePath << std::endl;
            std::cerr << "Error: " << doc.ErrorStr() << std::endl;
            return false;
        }

        // 获取机器人根节点
        tinyxml2::XMLElement* robotElem = doc.FirstChildElement("robot");
        if (!robotElem)
        {
            std::cerr << "No robot element found in URDF file" << std::endl;
            return false;
        }

        // 获取机器人名称
        if (robotElem->Attribute("name"))
        {
            urdfInfo.robotName = robotElem->Attribute("name");
        }

        // 解析连杆
        for (tinyxml2::XMLElement* linkElem = robotElem->FirstChildElement("link");
             linkElem;
             linkElem = linkElem->NextSiblingElement("link"))
        {
            UrdfLink link;
            
            // 获取连杆名称
            if (linkElem->Attribute("name"))
            {
                link.name = linkElem->Attribute("name");
            }
            else
            {
                std::cerr << "Link without name found, skipping..." << std::endl;
                continue;
            }

            // 解析视觉信息
            tinyxml2::XMLElement* visualElem = linkElem->FirstChildElement("visual");
            if (visualElem)
            {
                // 解析原点变换
                tinyxml2::XMLElement* originElem = visualElem->FirstChildElement("origin");

                // bool yUp = true;
                Transform3f meshTransform;

                if (objYUp) {
                    Real angle = Real(M_PI) * Real(0.5);   // +90 度
                    Quat<Real> q_yUpToZUp(0, 0, angle);    // yaw=0, pitch=0, roll=+90°
                    SquareMatrix<Real, 3> R_yUpToZUp = q_yUpToZUp.toMatrix3x3();

                    Vec3f t(0, 0, 0);
                    Vec3f s(1, 1, 1);
                    meshTransform.rotation() = R_yUpToZUp;
                }

                if (originElem)
                {
                    link.T_mesh = parseOrigin(originElem);
                    // meshTransform  = origin * (yUpToZUp * p_meshYup)
                    link.T_mesh = composeTransform(link.T_mesh, meshTransform);
                } else {
                    // meshTransform = origin * (yUpToZUp)
                    link.T_mesh = meshTransform;
                }

                // 解析几何信息
                tinyxml2::XMLElement* geometryElem = visualElem->FirstChildElement("geometry");
                if (geometryElem)
                {
                    //***TODO***: Figure out which type of the visual mesh is(dae/obj/stl)
                    tinyxml2::XMLElement* meshElem = geometryElem->FirstChildElement("mesh");
                    if (meshElem && meshElem->Attribute("filename"))
                    {
                        link.visualMeshPath = processMeshPath(meshElem->Attribute("filename"));
                    }
                }
                
                link.hasPatch = false;
                link.patchFaces.clear();
                link.patchOffsets.clear();

                tinyxml2::XMLElement* patchElem = visualElem->FirstChildElement("patch");
                if (patchElem)
                {
                    // 先标记存在 patch；如果读取失败，再将 hasPatch 置回 false 以便后续重新 patch
                    link.hasPatch = true;

                    bool loadedFromBin = false;

                    // 新格式：<information filename="xxx.bin"/>
                    tinyxml2::XMLElement* infoElem = patchElem->FirstChildElement("information");
                    if (infoElem && infoElem->Attribute("filename"))
                    {
                        std::string binName = infoElem->Attribute("filename");
                        std::string binPath = binName;

                        // 相对路径：默认相对于 urdf 文件所在目录
                        if (!isAbsolutePath(binPath))
                        {
                            const std::string urdfDir = getDirectoryName(filePath);
                            binPath = joinPath(urdfDir, binPath);
                        }

                        uint32_t binVer = 0;
                        if (!readPatchBin(binPath, link.patchFaces, link.patchOffsets, &binVer))
                        {
                            std::cerr << "[UrdfParser] Warning: failed to read patch bin file "
                                      << binPath << " on link " << link.name << std::endl;

                            // 读取失败：清空并允许后续重新 patch
                            link.patchFaces.clear();
                            link.patchOffsets.clear();
                            link.hasPatch = false;
                        }
                        else
                        {
                            loadedFromBin = true;
                        }
                    }

                    // 旧格式兼容：<faceID ID="..."/> + <offset prefixsum="..."/>
                    if (!loadedFromBin)
                    {
                        tinyxml2::XMLElement* faceElem = patchElem->FirstChildElement("faceID");
                        if (faceElem && faceElem->Attribute("ID"))
                        {
                            link.patchFaces = parseIntListAttr(faceElem->Attribute("ID"));
                        }
                        else
                        {
                            std::cerr << "[UrdfParser] Warning: <patch> exists but <information filename=\"...\"> missing "
                                         "and <faceID ID=\"...\"> missing on link "
                                      << link.name << std::endl;
                        }

                        tinyxml2::XMLElement* offsetElem = patchElem->FirstChildElement("offset");
                        if (offsetElem && offsetElem->Attribute("prefixsum"))
                        {
                            link.patchOffsets = parseIntListAttr(offsetElem->Attribute("prefixsum"));
                        }
                        else
                        {
                            std::cerr << "[UrdfParser] Warning: <patch> exists but <information filename=\"...\"> missing "
                                         "and <offset prefixsum=\"...\"> missing on link "
                                      << link.name << std::endl;
                        }

                        // 一致性检查（旧格式）
                        if (!link.patchOffsets.empty())
                        {
                            if (link.patchOffsets.size() < 2 || link.patchOffsets.front() != 0)
                            {
                                std::cerr << "[UrdfParser] Warning: patchOffsets invalid (size<2 or front!=0) on link "
                                          << link.name << std::endl;
                            }
                            if (!link.patchFaces.empty() &&
                                link.patchOffsets.back() != static_cast<int>(link.patchFaces.size()))
                            {
                                std::cerr << "[UrdfParser] Warning: patchOffsets.back() != patchFaces.size() on link "
                                          << link.name << " (back=" << link.patchOffsets.back()
                                          << ", faces=" << link.patchFaces.size() << ")" << std::endl;
                            }
                        }
                    }
                }
            }
            
            tinyxml2::XMLElement* collisionElem = linkElem->FirstChildElement("collision");
            if (collisionElem)
            {
                // 解析原点变换
                // tinyxml2::XMLElement* originElem = visualElem->FirstChildElement("origin");
                // if (originElem)
                // {
                //     link.origin = parseOrigin(originElem);
                // }

                // 解析几何信息
                tinyxml2::XMLElement* geometryElem = collisionElem->FirstChildElement("geometry");
                if (geometryElem)
                {
                    tinyxml2::XMLElement* meshElem = geometryElem->FirstChildElement("mesh");
                    if (meshElem && meshElem->Attribute("filename"))
                    {
                        link.collisionMeshPath = processMeshPath(meshElem->Attribute("filename"));
                    }
                }
            }

            urdfInfo.links.push_back(link);
        }

        // 解析关节
        for (tinyxml2::XMLElement* jointElem = robotElem->FirstChildElement("joint");
             jointElem;
             jointElem = jointElem->NextSiblingElement("joint"))
        {
            UrdfJoint joint;
            
            // 获取关节名称
            if (jointElem->Attribute("name"))
            {
                joint.name = jointElem->Attribute("name");
            }
            else
            {
                std::cerr << "Joint without name found, skipping..." << std::endl;
                continue;
            }

            // 解析关节类型
            if (jointElem->Attribute("type"))
            {
                joint.type = parseJointType(jointElem->Attribute("type"));
            }

            // 解析父连杆
            tinyxml2::XMLElement* parentElem = jointElem->FirstChildElement("parent");
            if (parentElem && parentElem->Attribute("link"))
            {
                joint.parentLink = parentElem->Attribute("link");
            }

            // 解析子连杆
            tinyxml2::XMLElement* childElem = jointElem->FirstChildElement("child");
            if (childElem && childElem->Attribute("link"))
            {
                joint.childLink = childElem->Attribute("link");
            }

            // 解析原点变换
            tinyxml2::XMLElement* originElem = jointElem->FirstChildElement("origin");
            if (originElem)
            {
                joint.originLocal = parseOrigin(originElem);
            }

            // 解析关节轴
            tinyxml2::XMLElement* axisElem = jointElem->FirstChildElement("axis");
            if (axisElem)
            {
                joint.axis = parseVector(axisElem, "xyz");
            }

            // 解析关节限制
            tinyxml2::XMLElement* limitElem = jointElem->FirstChildElement("limit");
            if (limitElem)
            {
                joint.limits = parseJointLimits(limitElem);
            }

            // 解析阻尼
            tinyxml2::XMLElement* dynamicsElem = jointElem->FirstChildElement("dynamics");
            if (dynamicsElem && dynamicsElem->QueryFloatAttribute("damping", &joint.damping) != tinyxml2::XML_SUCCESS)
            {
                joint.damping = 0.0f;  // 默认阻尼
            }

            urdfInfo.joints.push_back(joint);
        }

        // link name to index
        std::unordered_map<std::string, int> linkIndex;
        for (int i = 0; i < urdfInfo.links.size(); ++i)
        {
            linkIndex[urdfInfo.links[i].name] = i;
        }

        // 记录每个 link 的子关节
        std::unordered_map<std::string, std::vector<int>> linkChildJoints;
        // 记录“谁是 child link”，用来找 root link
        std::unordered_set<std::string> childLinks;

        for (size_t j = 0; j < urdfInfo.joints.size(); ++j)
        {
            auto& joint = urdfInfo.joints[j];
            linkChildJoints[joint.parentLink].push_back(static_cast<int>(j));
            childLinks.insert(joint.childLink);

            auto pIt = linkIndex.find(joint.parentLink);
            auto cIt = linkIndex.find(joint.childLink);

            if (pIt == linkIndex.end() || cIt == linkIndex.end())
            {
                std::cerr << "URDF error: link name not found: "
                          << joint.parentLink << " or " << joint.childLink << std::endl;
                continue;
            }

            joint.parentLinkId = pIt->second;
            joint.childLinkId  = cIt->second;
        }

        // 找 root link：出现在 links 中，但不在 childLinks 中
        std::string rootLinkName;
        for (auto& link : urdfInfo.links)
        {
            if (!childLinks.count(link.name))
            {
                rootLinkName = link.name;
                link.isRoot = true;
                break;
            }
        }

        computeWorldTransforms(rootLinkName, linkIndex, linkChildJoints, urdfInfo);
        return true;
    }

    Transform3f UrdfParser::parseOrigin(tinyxml2::XMLElement* originElem)
    {
        // 解析xyz平移
        Vec3f xyz(0, 0, 0);
        if (originElem->Attribute("xyz"))
        {
            std::stringstream ss(originElem->Attribute("xyz"));
            ss >> xyz[0] >> xyz[1] >> xyz[2];
        }
        
        // 解析rpy旋转 (roll, pitch, yaw)
        Vec3f rpy(0, 0, 0);
        if (originElem->Attribute("rpy"))
        {
            std::stringstream ss(originElem->Attribute("rpy"));
            ss >> rpy[0] >> rpy[1] >> rpy[2];
        }

        Quat<Real> quat(rpy[2], rpy[1], rpy[0]);
        SquareMatrix<Real, 3> rotationMatrixLocal = quat.toMatrix3x3();
        Vec3f scale(1, 1, 1);

        Transform3f transform(xyz, rotationMatrixLocal, scale);
                             
        return transform;
    }

    UrdfJointType UrdfParser::parseJointType(const std::string& typeStr)
    {
        if (typeStr == "revolute") return REVOLUTE;
        if (typeStr == "prismatic") return PRISMATIC;
        if (typeStr == "fixed") return FIXED;
        if (typeStr == "continuous") return CONTINUOUS;
        if (typeStr == "planar") return PLANAR;
        if (typeStr == "floating") return FLOATING;
        return UNKNOWN_JOINT;
    }

    UrdfJointLimits UrdfParser::parseJointLimits(tinyxml2::XMLElement* limitElem)
    {
        UrdfJointLimits limits;
        
        limitElem->QueryFloatAttribute("lower", &limits.lower);
        limitElem->QueryFloatAttribute("upper", &limits.upper);
        limitElem->QueryFloatAttribute("effort", &limits.effort);
        limitElem->QueryFloatAttribute("velocity", &limits.velocity);
        
        return limits;
    }

    Vec3f UrdfParser::parseVector(tinyxml2::XMLElement* elem, const std::string& attrName)
    {
        Vec3f vec(0, 0, 0);
        if (elem->Attribute(attrName.c_str()))
        {
            std::stringstream ss(elem->Attribute(attrName.c_str()));
            ss >> vec[0] >> vec[1] >> vec[2];
        }
        return vec;
    }

    Transform3f composeTransform(const Transform3f& parent,
                                    const Transform3f& local)
    {
        Transform3f out;

        // R_out = R_p * R_l
        out.rotation() = parent.rotation() * local.rotation();

        // t_out = R_p * t_l + t_p
        out.translation() = parent.rotation() * local.translation()
                            + parent.translation();

        // or：parent.scale() * local.scale()
        out.scale() = parent.scale();

        return out;
    }

    void UrdfParser::computeWorldTransforms(const std::string& rootLinkName,
                                        const std::unordered_map<std::string, int>& linkIndex,
                                        const std::unordered_map<std::string, std::vector<int>>& linkChildJoints,
                                        UrdfInformation& urdfInfo)
    {
        SquareMatrix<Real, 3> R_zUpToYUp {0, 1, 0, 0, 0, 1, 1, 0, 0};

        Vec3f t(0, 0, 0);
        Vec3f s(1, 1, 1);
        Transform3f T_world_root(t, R_zUpToYUp, s);

        auto itRoot = linkIndex.find(rootLinkName);
        if (itRoot != linkIndex.end()) {
            UrdfLink& rootLink = urdfInfo.links[itRoot->second];
            Transform3f T_local_root;
            rootLink.T_local = T_local_root;
        }

        // 递归下去
        computeWorldTransformsRecursive(rootLinkName, T_world_root, linkIndex, linkChildJoints, urdfInfo);
    }

    void UrdfParser::computeWorldTransformsRecursive(
        const std::string& linkName,
        const Transform3f& T_world_link,
        const std::unordered_map<std::string, int>& linkIndex,
        const std::unordered_map<std::string, std::vector<int>>& linkChildJoints,
        UrdfInformation& urdfInfo)
    {
        // 写回 link 的 world pose
        auto itLink = linkIndex.find(linkName);
        if (itLink == linkIndex.end()) return;

        UrdfLink& link = urdfInfo.links[itLink->second];
        link.T_world = T_world_link;

        // 找这个 link 下挂了哪些关节
        auto itJoints = linkChildJoints.find(linkName);
        if (itJoints == linkChildJoints.end())
            return;

        for (int jointIdx : itJoints->second)
        {
            UrdfJoint& joint = urdfInfo.joints[jointIdx];

            // joint.originWorld = T_world_parent * originLocal
            joint.originWorld = composeTransform(T_world_link, joint.originLocal);
            joint.axisWorld = joint.originWorld.rotation() * joint.axis;

            const std::string& childName = joint.childLink;

            int childId = joint.childLinkId;
            UrdfLink& childLink = urdfInfo.links[childId];
            childLink.T_local = joint.originLocal;
            // std::cout << "Name of link: " << childLink.name << "\n"
            //           << "Rotation: " << childLink.T_local.rotation() << "\n"
            //           << "Translation: " << childLink.T_local.translation() << "\n"
            //           << std::endl;
            Transform3f T_world_child = joint.originWorld;  // world -> child

            // 递归子 link
            computeWorldTransformsRecursive(childName, T_world_child, linkIndex, linkChildJoints, urdfInfo);
        }
    }
}
