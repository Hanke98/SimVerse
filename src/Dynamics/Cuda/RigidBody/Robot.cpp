#include "Robot.h"
#include <iostream>

#include "Collision/CollisionData.h"
#include "Mapping/DiscreteElementsToTriangleSet.h"
#include "GLSurfaceVisualModule.h"

namespace dyno
{
    IMPLEMENT_TCLASS(Robot, TDataType)

    template<typename TDataType>
	Robot<TDataType>::Robot() :
		ArticulatedBody<TDataType>()
    {
		
		auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
		this->stateTopology()->connect(mapper->inDiscreteElements());
		this->graphicsPipeline()->pushModule(mapper);

		auto sRender = std::make_shared<GLSurfaceVisualModule>();
		sRender->setColor(Color(1, 1, 0));
		sRender->setAlpha(0.2);
		mapper->outTriangleSet()->connect(sRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(sRender);

	}

    template<typename TDataType>
	Robot<TDataType>::~Robot()
	{

	}

    template<typename TDataType>
	void Robot<TDataType>::resetStates()
	{
		this->clearRigidBodySystem();
		this->clearRobot();

		std::string filename = getAssetPath() + "../asset/franka_description/robots/franka_panda.urdf";
        // std::string filename = getAssetPath() + "../asset/kuka_allegro_description/kuka.urdf";
        // std::string filename = getAssetPath() + "../asset/kuka_allegro_description/kuka_allegro_touch_sensor.urdf";

		if (this->varFilePath()->getValue() != filename)
		{
			this->varFilePath()->setValue(FilePath(filename));

		} else {
            std::cout << "Robot: Error when load file path " << std::endl;
        }

        // loadFromUrdf(filename);
        auto instances = this->varVehiclesTransform()->getValue();
        uint armNum = instances.size();

        for (size_t i = 0; i < armNum; i++) {
            RigidBodyInfo rigidbody;
            rigidbody.bodyId = i;

            auto texMesh = this->stateTextureMesh()->constDataPtr();
            std::map<int, std::shared_ptr<PdActor>> actors;
            std::unordered_map<std::string, int> linkNameToActorIndex;

            this->urdfParser.links.size();
            std::unordered_map<std::string, std::shared_ptr<PdActor>> linkNameToActor;



            auto shapes = texMesh->shapes();
            // auto shapeid = texMesh->shapeIds();
            std::unordered_map<std::string, int> linkIndex;

            for (int l = 0; l < this->urdfParser.links.size(); ++l) {

                auto it = this->urdfParser.links[l].shapeId;
                linkIndex[this->urdfParser.links[l].name] = it;
                std::cout << "name of link: " << this->urdfParser.links[l].name << " | shapeId of link: " << this->urdfParser.links[l].shapeId << std::endl;
                auto up = texMesh->shapes()[it]->boundingBox.v1;
                auto down = texMesh->shapes()[it]->boundingBox.v0;

                rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[it]->boundingTransform.translation()) + instances[i].translation();
                rigidbody.angle = Quat1f(instances[i].rotation());
                if (this->urdfParser.links[l].isRoot) {
                    rigidbody.motionType = BodyType::Static;
                } else {
                    rigidbody.motionType = BodyType::Dynamic;
                }

                auto actor = this->createRigidBody(rigidbody);
                actors[it] = actor;

                BoxInfo box;

                box.halfLength = (up - down) / 2;

                this->bindBox(actor, box);

                this->bindShape(actor, Pair<uint, uint>(it, i));
            }

            for (int j = 0; j < this->urdfParser.joints.size(); ++j) {
                auto parentName = this->urdfParser.joints[j].parentLink;
                auto childName = this->urdfParser.joints[j].childLink;

                if (this->urdfParser.joints[j].type == REVOLUTE) {
                    auto &joint = this->createHingeJoint(actors[linkIndex[parentName]], actors[linkIndex[childName]]);
                    joint.setAnchorPoint(this->urdfParser.joints[j].originWorld.translation() + instances[i].translation());
                    joint.setAxis(this->urdfParser.joints[j].originWorld.rotation() * this->urdfParser.joints[j].axis);
                    joint.setRange(this->urdfParser.joints[j].limits.lower, this->urdfParser.joints[j].limits.upper);
                }
                if (this->urdfParser.joints[j].type == PRISMATIC) {
                    auto &joint = this->createSliderJoint(actors[linkIndex[parentName]], actors[linkIndex[childName]]);
                    joint.setAnchorPoint(this->urdfParser.joints[j].originWorld.translation() + instances[i].translation());
                    joint.setAxis(this->urdfParser.joints[j].originWorld.rotation() * this->urdfParser.joints[j].axis);
                    joint.setRange(this->urdfParser.joints[j].limits.lower, this->urdfParser.joints[j].limits.upper);
                }
                if (this->urdfParser.joints[j].type == FIXED) {
                    auto &joint = this->createFixedJoint(actors[linkIndex[parentName]], actors[linkIndex[childName]]);
                    joint.setAnchorPoint(this->urdfParser.joints[j].originWorld.translation() + instances[i].translation());
                }
                std::cout << j << " joint parent: "<< parentName.c_str() << " | actorId:" << linkIndex[parentName] << std::endl;
                std::cout << j << " joint child: "<< childName.c_str() << " | actorId:" << linkIndex[childName] << std::endl;
                std::cout << j << " joint axis: " <<this->urdfParser.joints[j].originWorld.rotation() * this->urdfParser.joints[j].axis << std::endl;
                std::cout << j << " joint origin: " << this->urdfParser.joints[j].originWorld.translation() << std::endl;
                std::cout << j << " lower limits and upper limits: " << this->urdfParser.joints[j].limits.lower
                << ", " << this->urdfParser.joints[j].limits.upper<< std::endl;
            }
        }

        //**************************************************//
        ArticulatedBody<TDataType>::resetStates();
    }
    
    template<typename TDataType>
    bool Robot<TDataType>::loadFromUrdf()
    {
        auto parser = this->urdfParser;
        // if (!parser.parse(filePath, m_links, m_joints, m_robotName))
        // if (!parser.parse(filePath))
        // {
        //     std::cerr << "Failed to parse URDF file: " << filePath << std::endl;
        //     return false;
        // }

        m_links = parser.links;
        m_joints = parser.joints;
        m_robotName = parser.robotName;

        // 构建名称到索引的映射
        m_linkNameToIndex.clear();
        for (size_t i = 0; i < m_links.size(); ++i)
        {
            m_linkNameToIndex[m_links[i].name] = i;
        }

        m_jointNameToIndex.clear();
        for (size_t i = 0; i < m_joints.size(); ++i)
        {
            m_jointNameToIndex[m_joints[i].name] = i;
        }

        // 构建连杆到关节的映射
        buildLinkToJointsMap();

        std::cout << "Successfully loaded robot: " << m_robotName << std::endl;
        std::cout << "Number of links: " << m_links.size() << std::endl;
        std::cout << "Number of joints: " << m_joints.size() << std::endl;

        return true;
    }

    template<typename TDataType>
    const UrdfLink* Robot<TDataType>::getLink(const std::string& name) const
    {
        auto it = m_linkNameToIndex.find(name);
        if (it != m_linkNameToIndex.end())
        {
            return &m_links[it->second];
        }
        return nullptr;
    }

    template<typename TDataType>
    const UrdfJoint* Robot<TDataType>::getJoint(const std::string& name) const
    {
        auto it = m_jointNameToIndex.find(name);
        if (it != m_jointNameToIndex.end())
        {
            return &m_joints[it->second];
        }
        return nullptr;
    }

    template<typename TDataType>
    void Robot<TDataType>::buildLinkToJointsMap()
    {
        m_linkToJoints.clear();

        // 为每个连杆初始化空的关节列表
        for (const auto& link : m_links)
        {
            m_linkToJoints[link.name] = std::vector<std::string>();
        }

        // 将关节添加到父连杆和子连杆的列表中
        for (const auto& joint : m_joints)
        {
            if (m_linkToJoints.find(joint.parentLink) != m_linkToJoints.end())
            {
                m_linkToJoints[joint.parentLink].push_back(joint.name);
            }
            
            if (m_linkToJoints.find(joint.childLink) != m_linkToJoints.end())
            {
                m_linkToJoints[joint.childLink].push_back(joint.name);
            }
        }
    }

    DEFINE_CLASS(Robot);
}