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

            for (int it = 0; it < shapes.size(); ++it) {
                auto up = texMesh->shapes()[it]->boundingBox.v1;
                auto down = texMesh->shapes()[it]->boundingBox.v0;

                rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[it]->boundingTransform.translation()) + instances[i].translation();
                rigidbody.angle = Quat1f(instances[i].rotation());
                rigidbody.motionType = BodyType::Dynamic;
                if (it == 1) {
                    rigidbody.angularVelocity = Vec3f(0, 1, 0);
                } else {
                    rigidbody.linearVelocity = Vec3f(1, 0, 0);
                }

                auto actor = this->createRigidBody(rigidbody);
                actors[it] = actor;

                BoxInfo box;

                box.halfLength = (up - down) / 2;

                this->bindBox(actor, box);

                this->bindShape(actor, Pair<uint, uint>(it, i));
            }

            // for (size_t linkId = 0; linkId < this->urdfParser.links.size(); ++linkId)
            // {
            //     const auto& link = this->urdfParser.links[linkId];
            //     auto actorIt = actors.find(linkId);  // 根据索引从 actors 中找到对应的 actor
            //
            //     if (actorIt != actors.end())
            //     {
            //         linkNameToActor[link.name] = actorIt->second;  // 将 linkName 映射到 actor
            //     }
            //     else
            //     {
            //         std::cerr << "Error: Actor not found for link: " << link.name << std::endl;
            //     }
            // }

            // for (const auto& actorPair : actors) {
            //     std::cout << "Actor index: " << actorPair.first << " | Actor: " << actorPair.second->idx << std::endl;
            // }

            for (int j = 0; j < this->urdfParser.joints.size(); ++j) {
                if (this->urdfParser.joints[j].type == 0) {
                    auto &joint = this->createHingeJoint(actors[j], actors[j+1]);
                    joint.setAnchorPoint(this->urdfParser.joints[j].originWorld.translation() + instances[i].translation());
                    joint.setAxis(this->urdfParser.joints[j].originWorld.rotation() * this->urdfParser.joints[j].axis);
                    joint.setRange(this->urdfParser.joints[j].limits.lower, this->urdfParser.joints[j].limits.upper);
                }
                if (this->urdfParser.joints[j].type == 1) {
                    auto &joint = this->createSliderJoint(actors[j], actors[j+1]);
                    joint.setAnchorPoint(this->urdfParser.joints[j].originWorld.translation() + instances[i].translation());
                    joint.setAxis(this->urdfParser.joints[j].originWorld.rotation() * this->urdfParser.joints[j].axis);
                    joint.setRange(this->urdfParser.joints[j].limits.lower, this->urdfParser.joints[j].limits.upper);
                }
                if (this->urdfParser.joints[j].type == 2) {
                    auto &joint = this->createFixedJoint(actors[j], actors[j+1]);
                    joint.setAnchorPoint(this->urdfParser.joints[j].originWorld.translation() + instances[i].translation());
                }
                std::cout << j << " joint axis: " <<this->urdfParser.joints[j].originWorld.rotation() * this->urdfParser.joints[j].axis << std::endl;
                std::cout << j << " joint origin: " << this->urdfParser.joints[j].originWorld.translation() << std::endl;
                std::cout << j << " lower limits and upper limits: " << this->urdfParser.joints[j].limits.lower
                << ", " << this->urdfParser.joints[j].limits.upper<< std::endl;
            }

            //
            // auto &joint2 = this->createHingeJoint(actors[Link_main[1]], actors[Link_main[2]]);
            // joint2.setAnchorPoint(Vec3f(0.0f, 0.333f, 0.0f) + instances[i].translation());
            // joint2.setAxis(Vec3f(1.0f, 0.0f, 0.0f));
            // joint2.setRange(-1.7628, 1.7628);
            //
            // auto &joint3 = this->createHingeJoint(actors[Link_main[2]], actors[Link_main[3]]);
            // joint3.setAnchorPoint(Vec3f(0.0f, 0.333f + 0.316f, 0.0f) + instances[i].translation());
            // joint3.setAxis(Vec3f(0.0f, 1.0f, 0.0f));
            // joint3.setRange(-2.8973, 2.8973);
            //
            // auto &joint4 = this->createHingeJoint(actors[Link_main[3]], actors[Link_main[4]]);
            // joint4.setAnchorPoint(Vec3f(0.0825f, 0.333f + 0.316f, 0.0f) + instances[i].translation());
            // joint4.setAxis(Vec3f(-1.0f, 0.0f, 0.0f ));
            // // joint4.setRange(-3.0718, -0.0698);
            // joint4.setRange(-3.0, 0.087);
            //
            // auto &joint5 = this->createHingeJoint(actors[Link_main[4]], actors[Link_main[5]]);
            // joint5.setAnchorPoint(Vec3f(0.0f, 0.333f + 0.316f + 0.384f, 0.0f) + instances[i].translation());
            // joint5.setAxis(Vec3f(0.0f, 1.0f, 0.0f ));
            // joint5.setRange(-2.8973, 2.8973);
            //
            // auto &joint6 = this->createHingeJoint(actors[Link_main[5]], actors[Link_main[6]]);
            // joint6.setAnchorPoint(Vec3f(0.0f, 0.333f + 0.316 + 0.384f, 0.0f) + instances[i].translation());
            // joint6.setAxis(Vec3f(-1.0f, 0.0f, 0.0f ));
            // joint6.setRange(-0.0175, 3.7525);
            //
            // auto &joint7 = this->createHingeJoint(actors[Link_main[6]], actors[Link_main[7]]);
            // joint7.setAnchorPoint(Vec3f(0.088f, 0.333f + 0.316 + 0.384f, 0.0f) + instances[i].translation());
            // joint7.setAxis(Vec3f(0.0f, -1.0f, 0.0f ));
            // joint7.setRange(-2.8973, 2.8973);  !!!!!!
            //
            // auto &handjoint = this->createFixedJoint(actors[Link_main[7]], actors[Link_main[8]]);
            // handjoint.setAnchorPoint(Vec3f(0.088f, 0.333f + 0.316 + 0.384f - 0.107f, 0.0f) + instances[i].translation()); !!!
            //
            // auto &leftfingerjoint = this->createSliderJoint(actors[Link_main[8]], actors[Link_main[9]]);
            // leftfingerjoint.setAnchorPoint(Vec3f(0.088f, 0.333f + 0.316 + 0.384f - 0.107f - 0.0584, 0.0f) + instances[i].translation());
            // leftfingerjoint.setAxis(Vec3f(-0.7071f, 0.0f, 0.7071f));
            // leftfingerjoint.setRange(0, 0.04);
            //
            // auto &rightfingerjoint = this->createSliderJoint(actors[Link_main[8]], actors[Link_main[10]]);
            // rightfingerjoint.setAnchorPoint(Vec3f(0.088f, 0.333f + 0.316 + 0.384f - 0.107f - 0.0584, 0.0f) + instances[i].translation());
            // rightfingerjoint.setAxis(Vec3f(0.7071f, 0.0f, -0.7071f));
            // rightfingerjoint.setRange(0, 0.04);
        }

        //**************************************************//
        ArticulatedBody<TDataType>::resetStates();
    }
    
    template<typename TDataType>
    bool Robot<TDataType>::loadFromUrdf(const std::string& filePath)
    {
        UrdfParser parser;
        // if (!parser.parse(filePath, m_links, m_joints, m_robotName))
        if (!parser.parse(filePath))
        {
            std::cerr << "Failed to parse URDF file: " << filePath << std::endl;
            return false;
        }

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