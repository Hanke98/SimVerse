#pragma once
#include <vector>
#include <string>
#include <map>
#include "UrdfParser.txt"
#include "OBase.h"
#include "Module.h"
#include "ArticulatedBody.h"

#include "Topology/TriangleSet.h"

namespace dyno
{   
    template<typename TDataType>
    class Robot : virtual public ArticulatedBody<TDataType>
    {
        DECLARE_TCLASS(Robot, TDataType)
    public:
        typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

        DEF_VAR_OUT(bool, Reset, "Reset");

        Robot();
        ~Robot() override;

        // 从URDF文件加载机器人模型
        bool loadFromUrdf(const std::string& filePath);

        // 获取机器人名称
        const std::string& getName() const { return m_robotName; }

        // 获取所有连杆
        const std::vector<UrdfLink>& getLinks() const { return m_links; }
        
        // 根据名称获取连杆
        const UrdfLink* getLink(const std::string& name) const;

        // 获取所有关节
        const std::vector<UrdfJoint>& getJoints() const { return m_joints; }
        
        // 根据名称获取关节
        const UrdfJoint* getJoint(const std::string& name) const;

        // 获取关节与连杆的映射关系
        const std::map<std::string, std::vector<std::string>>& getLinkToJointsMap() const 
        { 
            return m_linkToJoints; 
        }

        void resetStates() override;

        // DEF_INSTANCE_STATE(TriangleSet<TDataType>, Link0Mesh, "Mesh of Link0");
        // DEF_INSTANCE_STATE(TriangleSet<TDataType>, Link1Mesh, "Mesh of Link1");
        // DEF_INSTANCE_STATE(TriangleSet<TDataType>, Link2Mesh, "Mesh of Link2");
        // DEF_INSTANCE_STATE(TriangleSet<TDataType>, Link3Mesh, "Mesh of Link3");
        // DEF_INSTANCE_STATE(TriangleSet<TDataType>, Link4Mesh, "Mesh of Link4");
        // DEF_INSTANCE_STATE(TriangleSet<TDataType>, Link5Mesh, "Mesh of Link5");
        // DEF_INSTANCE_STATE(TriangleSet<TDataType>, Link6Mesh, "Mesh of Link6");

    protected:
		// void resetStates() override;
        
    private:
        // 构建连杆到关节的映射
        void buildLinkToJointsMap();

    private:
        std::string m_robotName;
        std::vector<UrdfLink> m_links;
        std::vector<UrdfJoint> m_joints;
        std::map<std::string, std::vector<std::string>> m_linkToJoints;  // 连杆到关节的映射
        std::map<std::string, size_t> m_linkNameToIndex;                 // 连杆名称到索引的映射
        std::map<std::string, size_t> m_jointNameToIndex;                // 关节名称到索引的映射
    };
}