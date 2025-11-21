#include <iostream>
#include <vector>
#include <RigidBody/Robot.h>
#include <../../../src/Modeling/UrdfParser.h>
// #include "Core/DataType.h"
// #include "Framework/Framework.h"

#include <UbiApp.h>

#include <SceneGraph.h>

#include <RigidBody/Gear.h>
#include <RigidBody/MultibodySystem.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>

#include <BasicShapes/PlaneModel.h>

#include <Collision/NeighborElementQuery.h>

using namespace dyno;

std::shared_ptr<SceneGraph> createSceneGraph()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setGravity(Vec3f(0.0f, 0.0f, 0.0f));

    uint num_rotbots = 1;

    // auto convoy = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
    for (uint i = 0; i < num_rotbots; i++)
    {
        auto robot = scn->addNode(std::make_shared<Robot<DataType3f>>());
        std::vector<Transform3f> transforms(1);
        transforms[0] = Transform3f(Vec3f((i + 1.0f) * 1.0f, 0.0f, 0.0f), 
                                    Mat3f(1.0f, 0.0f, 0.0f, 
                                            0.0f, 1.0f, 0.0f, 
                                            0.0f, 0.0f, 1.0f), 
                                    Vec3f(0.0f, 0.0f, 0.0f));
        robot->varVehiclesTransform()->setValue(transforms);

        auto convoy = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	    robot->connect(convoy->importVehicles());
        
        auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
        plane->varLengthX()->setValue(50);
        plane->varLengthZ()->setValue(50);
        plane->varSegmentX()->setValue(10);
        plane->varSegmentZ()->setValue(10);

        plane->stateTriangleSet()->connect(convoy->inTriangleSet());
    }

	return scn;
}

int main() {
    // 初始化框架
    // Framework::instance()->initialize();

    try {
        // 1. 测试URDF解析器
        // std::cout << "Testing UrdfParser..." << std::endl;
        // UrdfParser parser;
        // std::string urdfPath = getAssetPath() + "../asset/franka_description/robots/franka_panda.urdf"; // 替换为实际URDF路径
        //
        // 准备解析所需的容器
        // std::vector<UrdfLink> links;
        // std::vector<UrdfJoint> joints;
        // std::string robotName;
        
        // 调用parse函数（需要4个参数）
        // bool parseSuccess = parser.parse(urdfPath, links, joints, robotName);
        // bool parseSuccess = parser.parse(urdfPath);
        // if (!parseSuccess) {
        //     throw std::runtime_error("Failed to parse URDF file: " + urdfPath);
        // }
        //
        // std::cout << "URDF parsed successfully! Robot info:" << std::endl;
        // std::cout << "  Name: " << parser.robotName << std::endl;
        // std::cout << "  Links: " << parser.links.size() << std::endl;
        // std::cout << "  Joints: " << parser.joints.size() << std::endl;
        //
        // for (const auto& link : parser.links) {
        //     std::cout << "    Link: " << link.name << ", Visual Mesh: "
        //         << link.visualMeshPath << ", Collision Mesh: " << link.collisionMeshPath
        //         // << "\n" << "    Local rotation matrix: " << link.meshTransform.rotation()
        //         << std::endl;
        // }
        //
        // for (const auto& joint : parser.joints) {
        //     Vec3f axisWorld = joint.originWorld.rotation() * joint.axis;

            // std::cout << "    Joint: " << joint.name
            //           << ", Type: " << joint.type
            //           << ", Parent: " << joint.parentLink
            //           << ", Child: " << joint.childLink
            //           << ", Axis: [" << joint.axis.x << ", " << joint.axis.y << ", " << joint.axis.z << "]"
            //           << ", AxisWorld: " << axisWorld
            //           << ", Limits: [" << joint.limits.lower << ", " << joint.limits.upper << "]"
            //           << ", Damping: " << joint.damping
            //           << ", Local Rotation Matrix:" << joint.originLocal.rotation()
            //           << ", World Rotation Matrix:" << joint.originWorld.rotation() << "\n"
            //           << ", World Translation Vector:" << joint.originWorld.translation()
            //           << std::endl;
        // }


        // 2. 测试Robot类
        std::cout << "\nTesting Robot class..." << std::endl;
        auto robot = std::make_shared<Robot<DataType3f>>();  // 修正：Robot不是模板类，去掉模板参数
        
    }
    catch (const std::exception& e) {  // 修正：正确闭合try-catch
        std::cerr << "Test failed: " << e.what() << std::endl;
        // Framework::instance()->shutdown();
        return 1;
    }

    UbiApp app(GUIType::GUI_GLFW);
	app.setSceneGraph(createSceneGraph());
    
	app.initialize(1280, 768);
	app.renderWindow()->getCamera()->setUnitScale(1.0f);
	app.mainLoop();

    return 0;
}