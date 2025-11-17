#include "UrdfParser.txt"
#include <iostream>
#include <sstream>

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

    bool UrdfParser::parse(const std::string& filePath, 
                          std::vector<UrdfLink>& links, 
                          std::vector<UrdfJoint>& joints,
                          std::string& robotName)
    {
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
            robotName = robotElem->Attribute("name");
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
                if (originElem)
                {
                    link.origin = parseOrigin(originElem);
                }

                // 解析几何信息
                tinyxml2::XMLElement* geometryElem = visualElem->FirstChildElement("geometry");
                if (geometryElem)
                {
                    tinyxml2::XMLElement* meshElem = geometryElem->FirstChildElement("mesh");
                    if (meshElem && meshElem->Attribute("filename"))
                    {
                        link.visualMeshPath = processMeshPath(meshElem->Attribute("filename"));
                    }
                }
            }

            links.push_back(link);
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
                joint.origin = parseOrigin(originElem);
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

            joints.push_back(joint);
        }

        return true;
    }

    Transform3f UrdfParser::parseOrigin(tinyxml2::XMLElement* originElem)
    {
        Transform3f transform;
        
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
        
        // transform.setTranslation(xyz);
        // transform.setRotation(Quat1f(rpy[0], Vec3f(1, 0, 0)) * 
        //                      Quat1f(rpy[1], Vec3f(0, 1, 0)) * 
        //                      Quat1f(rpy[2], Vec3f(0, 0, 1)));
                             
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
}