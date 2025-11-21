#pragma once
#include <string>
#include <vector>
#include <memory>
#include <tinyxml/tinyxml2.h>
#include "Vector.h"
#include "Matrix.h"

namespace dyno
{
    // 关节类型枚举
    enum UrdfJointType
    {
        REVOLUTE,   // 旋转关节
        PRISMATIC,  // 移动关节
        FIXED,      // 固定关节
        CONTINUOUS, // 连续旋转关节
        PLANAR,     // 平面关节
        FLOATING,   // 浮动关节
        UNKNOWN_JOINT     // 未知类型
    };

    // 连杆信息结构体
    struct UrdfLink
    {
        std::string name;
        std::string visualMeshPath;  // 视觉网格路径(.dae)
        Transform3f meshTransform;          // 原点变换
        std::string collisionMeshPath; // collision mesh path
        Transform3f T_world; // world transform
        uint shapeId;
    };

    // 关节限制信息
    struct UrdfJointLimits
    {
        float lower;     // 下限
        float upper;     // 上限
        float effort;    // 力限制
        float velocity;  // 速度限制
    };

    // 关节信息结构体
    struct UrdfJoint
    {
        std::string name;
        UrdfJointType type;
        std::string parentLink;
        std::string childLink;
        Vec3f axis;               // 关节轴
        Vec3f axisWorld;
        Transform3f originLocal;  // 原点变换
        Transform3f originWorld;
        UrdfJointLimits limits;   // 关节限制
        float damping;            // 阻尼系数
    };

    // URDF解析器类
    class UrdfParser
    {
    public:
        UrdfParser() = default;
        ~UrdfParser() = default;

        bool parse(const std::string& filePath);

        std::vector<UrdfLink> links;
        std::vector<UrdfJoint> joints;
        std::string robotName;

    private:
        // 解析变换信息
        static Transform3f parseOrigin(tinyxml2::XMLElement* originElem);
        
        // 解析关节类型
        static UrdfJointType parseJointType(const std::string& typeStr);
        
        // 解析关节限制
        static UrdfJointLimits parseJointLimits(tinyxml2::XMLElement* limitElem);
        
        // 解析向量
        Vec3f parseVector(tinyxml2::XMLElement* elem, const std::string& attrName);

        void computeWorldTransforms(const std::string& rootLinkName,
                                    const std::unordered_map<std::string, int>& linkIndex,
                                    const std::unordered_map<std::string, std::vector<int>>& linkChildJoints);

        void computeWorldTransformsRecursive(
                                    const std::string& linkName,
                                    const Transform3f& T_world_link,
                                    const std::unordered_map<std::string, int>& linkIndex,
                                    const std::unordered_map<std::string, std::vector<int>>& linkChildJoints);
    };

    Transform3f composeTransform(const Transform3f& parent, const Transform3f& local);

}
