#include "UrdfParser.h"
#include <iostream>
#include <sstream>
#include <unordered_set>

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
                    link.meshTransform = parseOrigin(originElem);
                    // meshTransform  = origin * (yUpToZUp * p_meshYup)
                    link.meshTransform = composeTransform(link.meshTransform, meshTransform);
                } else {
                    // meshTransform = origin * (yUpToZUp)
                    link.meshTransform = meshTransform;
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

            // std::cout << "name of joint parent: " << joint.parentLink.c_str() <<" index of parent: " << pIt->second << std::endl;
            // std::cout << "name of joint child: " << joint.childLink.c_str() <<" index of child: " << cIt->second << std::endl;
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
        // 初始化 root link 世界变换为单位变换
        // Transform3f T_world_root;

        SquareMatrix<Real, 3> R_zUpToYUp {0, 1, 0, 0, 0, 1, 1, 0, 0};

        Vec3f t(0, 0, 0);
        Vec3f s(1, 1, 1);
        Transform3f T_world_root(t, R_zUpToYUp, s);

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

            // 若你在 link 里还有额外的 <origin> (例如视觉/碰撞)，可以再乘一次
            const std::string& childName = joint.childLink;
            Transform3f T_world_child = joint.originWorld;  // world -> child

            // 递归子 link
            computeWorldTransformsRecursive(childName, T_world_child, linkIndex, linkChildJoints, urdfInfo);
        }
    }
}
