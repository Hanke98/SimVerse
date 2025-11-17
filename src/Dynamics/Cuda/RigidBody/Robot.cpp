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

		std::string filename = getAssetPath() + "franka_description/robots/franka_panda.urdf";
		if (this->varFilePath()->getValue() != filename)
		{
			this->varFilePath()->setValue(FilePath(filename));
		} else {
            std::cout << "Robot: Error when load file path " << std::endl;
        }
        // std::vector<Transform3f> transforms(2);
        // transforms[0] = Transform3f(Vec3f(0.0f, 0.0f, 0.0f), Mat3f(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f), Vec3f(0.0f, 0.0f, 0.0f));
        // transforms[1] = Transform3f(Vec3f(1.0f, 0.0f, 0.0f), Mat3f(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f), Vec3f(0.0f, 0.0f, 0.0f));
        // this->varVehiclesTransform()->setValue(transforms);
        auto instances = this->varVehiclesTransform()->getValue();
		uint armNum = instances.size();
     
        for (size_t i = 0; i < armNum; i++) {
            RigidBodyInfo rigidbody;
            rigidbody.bodyId = i;

            auto texMesh = this->stateTextureMesh()->constDataPtr();
            std::map<int, std::shared_ptr<PdActor>> actors;

            std::vector <int> Link1_Id = {12};
            std::vector <int> Link0_Id = {0, 1, 2, 3, 4, 5, 6, 7, 8,9, 10, 11};
            std::vector <int> Link2_Id = {13};
            std::vector <int> Link3_Id = {14, 15, 16, 17};
            std::vector <int> Link4_Id = {18, 19, 20, 21};
            std::vector <int> Link5_Id = {22, 23, 24};
            // std::vector <int> Link6_Id = {25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41};
            // std::vector <int> Link6_Id = {25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41}; // without 32
            std::vector <int> Link6_Id = {25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41};
            std::vector <int> Link7_Id = {42, 43, 44, 45, 46, 47, 48, 49};
            std::vector <int> Hand_Id = {50, 51, 52, 53, 54};
            std::vector <int> Left_Finger_Id = {55, 56};
            std::vector <int> Right_Finger_Id = {57, 58};
            std::vector <int> Link_main = {8, 12, 13, 14,19, 24, 25, 49, 53, 55, 57};
            // std::vector <int> finger_main = {55, 57};
            // std::vector <int> hand_main = {53};
            std::vector <std::vector <int>> Link_Id = {Link0_Id, Link1_Id, Link2_Id, Link3_Id, Link4_Id, Link5_Id, Link6_Id, Link7_Id, Hand_Id, Left_Finger_Id, Right_Finger_Id};
            // for (int it = 0; it < texMesh->shapes().size(); it++) { 
            // for (auto it : Link0_Id ) { 
            // for (auto it : finger_main  ) { 
            int j = 0;
            // for (auto it : Link_main ) { 
            for (auto link_id : Link_Id) {
                for (auto it : link_id) {
                    auto up = texMesh->shapes()[it]->boundingBox.v1;
                    auto down = texMesh->shapes()[it]->boundingBox.v0;

                    rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[it]->boundingTransform.translation()) + instances[i].translation();
                    rigidbody.angle = Quat1f(instances[i].rotation());
                    rigidbody.motionType = BodyType::Dynamic;

                    if (it == Link_main[j]) {
                        if (it == Link_main[0]) {
                            rigidbody.motionType = BodyType::Static;
                        }
                    }
                    
                    auto actor = this->createRigidBody(rigidbody);
                    actors[it] = actor;

                    BoxInfo box;
                    box.rot = Quat1f(0, Vec3f(0, 0, 1));
                    box.halfLength = (up - down) / 2;

                    this->bindBox(actor, box);	

                    this->bindShape(actor, Pair<uint, uint>(it, i));
                }
                j++;
            }
            int linkIndex = 0;
            for (auto link_id : Link_Id) {

                for (auto it : link_id) {
                    if (it != Link_main[linkIndex]) {
                        auto& fix = this->createFixedJoint(actors[it], actors[Link_main[linkIndex]]);
                        fix.setAnchorPoint(actors[Link_main[linkIndex]]->center);
                    }
                }
                linkIndex++;
            }

            auto &joint1 = this->createHingeJoint(actors[Link_main[0]], actors[Link_main[1]]); 
            joint1.setAnchorPoint(Vec3f(0.0f, 0.333f, 0.0f) + instances[i].translation());
            joint1.setAxis(Vec3f(0.0f, 1.0f, 0.0f));
            joint1.setRange(-2.8973, 2.8973);
            // joint1.setRange(1.000, 1.000);
            joint1.setMoter(0.0);

            auto &joint2 = this->createHingeJoint(actors[Link_main[1]], actors[Link_main[2]]);
            joint2.setAnchorPoint(Vec3f(0.0f, 0.333f, 0.0f) + instances[i].translation());
            joint2.setAxis(Vec3f(0.0f, 0.0f, 1.0f));
            joint2.setRange(-1.7628, 1.7628);
            // joint2.setRange(-1.000, -1.000);
            joint2.setMoter(0.0);

            auto &joint3 = this->createHingeJoint(actors[Link_main[2]], actors[Link_main[3]]);
            joint3.setAnchorPoint(Vec3f(0.0f, 0.333f + 0.316f, 0.0f) + instances[i].translation());
            joint3.setAxis(Vec3f(0.0f, 1.0f, 0.0f));
            joint3.setRange(-2.8973, 2.8973);
            // joint3.setRange(1.000, 1.000);
            joint3.setMoter(0.0);

            auto &joint4 = this->createHingeJoint(actors[Link_main[3]], actors[Link_main[4]]);
            joint4.setAnchorPoint(Vec3f(0.0825f, 0.333f + 0.316f, 0.0f) + instances[i].translation());  
            joint4.setAxis(Vec3f(0.0f, 0.0f, 1.0f ));
            joint4.setRange(-3.0718, -0.0698);
            // joint4.setRange(-3.0, 0.087);
            // joint4.setRange(-0.3475, -0.3475);
            joint4.setMoter(0.0);

            auto &joint5 = this->createHingeJoint(actors[Link_main[4]], actors[Link_main[5]]);
            joint5.setAnchorPoint(Vec3f(0.0f, 0.333f + 0.316f + 0.384f, 0.0f) + instances[i].translation());
            joint5.setAxis(Vec3f(0.0f, 1.0f, 0.0f ));
            joint5.setRange(-2.8973, 2.8973);
            // joint5.setRange(0.1879, 0.1879);
            joint5.setMoter(0.0);

            auto &joint6 = this->createHingeJoint(actors[Link_main[5]], actors[Link_main[6]]);
            joint6.setAnchorPoint(Vec3f(0.0f, 0.333f + 0.316 + 0.384f, 0.0f) + instances[i].translation());
            joint6.setAxis(Vec3f(0.0f, 0.0f, 1.0f ));
            joint6.setRange(-0.0175, 3.7525);
            // joint6.setRange(0.6876, 0.6876);
            joint6.setMoter(0.0);

            auto &joint7 = this->createHingeJoint(actors[Link_main[6]], actors[Link_main[7]]);
            joint7.setAnchorPoint(Vec3f(0.088f, 0.333f + 0.316 + 0.384f, 0.0f) + instances[i].translation());
            joint7.setAxis(Vec3f(0.0f, 1.0f, 0.0f ));
            joint7.setRange(-2.8973, 2.8973);
            // joint7.setRange(-0.2547, -0.2547);
            joint7.setMoter(0.0);

            auto &handjoint = this->createFixedJoint(actors[Link_main[7]], actors[Link_main[8]]);
            handjoint.setAnchorPoint(Vec3f(0.088f, 0.333f + 0.316 + 0.384f - 0.107f, 0.0f) + instances[i].translation());

            auto &leftfingerjoint = this->createSliderJoint(actors[Link_main[8]], actors[Link_main[9]]);
            leftfingerjoint.setAnchorPoint(Vec3f(0.088f, 0.333f + 0.316 + 0.384f - 0.107f - 0.0584, 0.0f) + instances[i].translation());
            leftfingerjoint.setAxis(Vec3f(0.7071f, 0.0f, 0.7071f));
            leftfingerjoint.setRange(0, 0.04);
            // leftfingerjoint.setRange(0.0, 0.0);
            // leftfingerjoint.setMoter(0.1);

            auto &rightfingerjoint = this->createSliderJoint(actors[Link_main[8]], actors[Link_main[10]]);
            rightfingerjoint.setAnchorPoint(Vec3f(0.088f, 0.333f + 0.316 + 0.384f - 0.107f - 0.0584, 0.0f) + instances[i].translation());
            rightfingerjoint.setAxis(Vec3f(-0.7071f, 0.0f, -0.7071f));
            rightfingerjoint.setRange(0, 0.04);
            // rightfingerjoint.setRange(0.0, 0.0);
            // rightfingerjoint.setMoter(0.1);
        }

		//**************************************************//
		ArticulatedBody<TDataType>::resetStates();
	}
    
    template<typename TDataType>
    bool Robot<TDataType>::loadFromUrdf(const std::string& filePath)
    {
        UrdfParser parser;
        if (!parser.parse(filePath, m_links, m_joints, m_robotName))
        {
            std::cerr << "Failed to parse URDF file: " << filePath << std::endl;
            return false;
        }

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