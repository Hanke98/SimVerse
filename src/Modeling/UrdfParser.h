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
        std::string visualMeshPath;       // 视觉网格路径
        std::string collisionMeshPath;    // collision mesh path
        Transform3f T_mesh;               // mesh transform
        Transform3f T_world;              // world transform
        Transform3f T_local;              // local transform
        Transform3f T_visual_bb_local;
        Transform3f T_visual_bb_world;
        Transform3f T_collision_bb_local;
        Transform3f T_collision_bb_world;
        uint visualShapeId;
        uint collisionShapeId;
        bool isRoot = false;
        Real volume;
        Mat3f localInertia;
    };

    // 关节限制信息
    struct UrdfJointLimits
    {
        Real lower;     // 下限
        Real upper;     // 上限
        Real effort;    // 力限制
        Real velocity;  // 速度限制
    };

    // 关节信息结构体
    struct UrdfJoint
    {
        std::string name;
        UrdfJointType type;
        std::string parentLink;
        std::string childLink;
        uint parentLinkId;
        uint childLinkId;
        Vec3f axis;               // 关节轴
        Vec3f axisWorld;
        Transform3f originLocal;  // 原点变换
        Transform3f originWorld;
        UrdfJointLimits limits;   // 关节限制
        Real damping;            // 阻尼系数
    };

    // TODO: Refactor this to `KinematicsChainInfo`
    struct UrdfInformation
    {
        std::vector<UrdfLink> links;
        std::vector<UrdfJoint> joints;
        std::string robotName;
    };
    
    // URDF解析器类
    class UrdfParser
    {
    public:
        UrdfParser() = default;
        ~UrdfParser() = default;

        bool parse(const std::string& filePath, UrdfInformation& urdfInfo, bool objYUp);

        // ***TODO***: create a new structure to store all the information
        // std::vector<UrdfLink> links;
        // std::vector<UrdfJoint> joints;
        // std::string robotName;

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
                                    const std::unordered_map<std::string, std::vector<int>>& linkChildJoints,
                                    UrdfInformation& urdfInfo);

        void computeWorldTransformsRecursive(
                                    const std::string& linkName,
                                    const Transform3f& T_world_link,
                                    const std::unordered_map<std::string, int>& linkIndex,
                                    const std::unordered_map<std::string, std::vector<int>>& linkChildJoints,
                                    UrdfInformation& urdfInfo);
    };

    Transform3f composeTransform(const Transform3f& parent, const Transform3f& local);

}
