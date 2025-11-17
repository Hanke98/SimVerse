#include "RobotArm.h"
#include <GLRenderEngine.h>
#include <GLSurfaceVisualModule.h>
#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <UbiApp.h>
#include <GLFW/glfw3.h>
#include <OrbitCamera.h>
#include <iostream>
#include <random>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include "../../../../src/Rendering/GUI/GlfwGUI/GlfwRenderWindow.h"
#include <SceneGraphFactory.h> 

#include <BasicShapes/PlaneModel.h>




namespace dyno
{
    IMPLEMENT_TCLASS(RobotArmSimulator, TDataType)

    template<typename TDataType>
    RobotArmSimulator<TDataType>::RobotArmSimulator() :
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
	void RobotArmSimulator<TDataType>::resetStates()
	{
        this->clearRigidBodySystem();
		this->clearRobot();

		std::string filename = "/home/wjv/SimVerse_01/SimVerse/asset/franka_description/robots/franka_panda.urdf";
		if (this->varFilePath()->getValue() != filename)
		{
			this->varFilePath()->setValue(FilePath(filename));
        }
        auto instances = this->varVehiclesTransform()->getValue();
		uint armNum = instances.size();
        float density = this->varDensity()->getValue();

        // for (size_t i = 0; i < armNum; i++) {
        RigidBodyInfo rigidbody;
        int i = 0;
        rigidbody.bodyId = i;
        // rigidbody.friction = 0.0f;

        auto texMesh = this->stateTextureMesh()->constDataPtr();
        std::map<int, std::shared_ptr<PdActor>> actors;

        std::vector <int> Link0_Id = {8};
        std::vector <int> Link1_Id = {12};
        // std::vector <int> Link0_Id = {0, 1, 2, 3, 4, 5, 6, 7, 8,9, 10, 11};
        std::vector <int> Link2_Id = {13};
        // std::vector <int> Link3_Id = {14, 15, 16, 17};
        std::vector <int> Link3_Id = {14};
        // std::vector <int> Link4_Id = {18, 19, 20, 21};
        std::vector <int> Link4_Id = {19};
        // std::vector <int> Link5_Id = {22, 23, 24};
        std::vector <int> Link5_Id = {24};
        // std::vector <int> Link6_Id = {25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41};
        // std::vector <int> Link6_Id = {25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41}; // without 32
        std::vector <int> Link6_Id = {25, 41};
        // std::vector <int> Link7_Id = {42, 43, 44, 45, 46, 47, 48, 49};
        std::vector <int> Link7_Id = {49};
        // std::vector <int> Hand_Id = {50, 51, 52, 53, 54};
        std::vector <int> Hand_Id = {53};
        std::vector <int> Left_Finger_Id = {55};
        std::vector <int> Right_Finger_Id = {57};
        // std::vector <int> Link_main = {8, 12, 13, 14, 19, 24, 25, 49, 53, 55, 57};
        std::vector <std::vector <int>> Link_Id = {Link0_Id, Link1_Id, Link2_Id, Link3_Id, Link4_Id, Link5_Id, Link6_Id, Link7_Id, Hand_Id, Left_Finger_Id, Right_Finger_Id};

        int j = 0;

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
                // box.rot = Quat1f(1, Vec3f(0, 0, 0));
                // box.rot = Quat1f(0.5, 0.5, 0.5, 0.5);
                box.halfLength = (up - down) / 2;
                box.center = Vec3f(0.0f);

                this->bindBox(actor, box, density);	

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

        // create joints
        auto &joint1 = this->createFixedJoint(actors[Link_main[0]], actors[Link_main[1]]);
        joint1.setAnchorPoint(Vec3f(0.0f, 0.333f, 0.0f) + instances[i].translation());
        // joint1.setAxis(Vec3f(0.0f, 1.0f, 0.0f));
        // // joint1.setRange(-2.8973, 2.8973);
        // joint1.setRange(0.0, 0.0);
        // joint1.setMoter(0.0);

        auto &joint2 = this->createHingeJoint(actors[Link_main[1]], actors[Link_main[2]]);
        joint2.setAnchorPoint(Vec3f(0.0f, 0.333f, 0.0f) + instances[i].translation());
        joint2.setAxis(Vec3f(-1.0f, 0.0f, 0.0f));
        joint2.setRange(-1.7628, 1.7628);
        // joint2.setRange(-1.7628, -1.7628);
        // joint2.setMoter(0.0);

        auto &joint3 = this->createFixedJoint(actors[Link_main[2]], actors[Link_main[3]]);
        joint3.setAnchorPoint(Vec3f(0.0f, 0.333f + 0.316f, 0.0f) + instances[i].translation());
        // joint3.setAxis(Vec3f(0.0f, 1.0f, 0.0f));
        // // joint3.setRange(-2.8973, 2.8973);
        // joint3.setRange(0.0, 0.0);
        // joint3.setMoter(0.0);

        auto &joint4 = this->createFixedJoint(actors[Link_main[3]], actors[Link_main[4]]);
        joint4.setAnchorPoint(Vec3f(0.0f, 0.333f + 0.316f, 0.0825f) + instances[i].translation());
        // joint4.setAxis(Vec3f(-1.0f, 0.0f, 0.0f ));
        // // joint4.setRange(-3.0718, -0.0698);
        // // joint4.setRange(-3.0, 0.087);
        // // joint4.setRange(-0.087, 3.0);
        // joint4.setRange(2.6180, 2.6180);
        // joint4.setMoter(0.0);

        auto &joint5 = this->createFixedJoint(actors[Link_main[4]], actors[Link_main[5]]);
        joint5.setAnchorPoint(Vec3f(0.0f, 0.333f + 0.316f + 0.384f, 0.0f) + instances[i].translation());
        // joint5.setAxis(Vec3f(0.0f, 1.0f, 0.0f ));
        // joint5.setRange(-2.8973, 2.8973);
        // joint5.setRange(0.0, 0.0);
        // joint5.setMoter(0.0);

        // auto &joint5 = this->createFixedJoint(actors[Link_main[4]], actors[Link_main[5]]);
        // joint5.setAnchorPoint(Vec3f(0.0f, 0.333f + 0.316f + 0.384f, 0.0f) + instances[i].translation());

        auto &joint6 = this->createFixedJoint(actors[Link_main[5]], actors[Link_main[6]]);
        joint6.setAnchorPoint(Vec3f(0.0f, 0.333f + 0.316 + 0.384f, 0.0f) + instances[i].translation());
        // joint6.setAxis(Vec3f(-1.0f, 0.0f, 0.0f ));
        // // joint6.setRange(-0.0175, 3.7525);
        // // joint6.setRange(-3.7525, 0.0175);
        // joint6.setRange(-2.9416, -2.9416);
        // joint6.setMoter(0.0);

        auto &joint7 = this->createFixedJoint(actors[Link_main[6]], actors[Link_main[7]]);
        joint7.setAnchorPoint(Vec3f(0.0f, 0.333f + 0.316 + 0.384f, 0.088f) + instances[i].translation());
        // joint7.setAxis(Vec3f(0.0f, 1.0f, 0.0f ));
        // // joint7.setRange(-2.8973, 2.8973);
        // joint7.setRange(-0.7854, -0.7854);
        // joint7.setMoter(0.0);

        auto &handjoint = this->createFixedJoint(actors[Link_main[7]], actors[Link_main[8]]);
        handjoint.setAnchorPoint(Vec3f(0.0f, 0.333f + 0.316 + 0.384f - 0.107f, 0.088f) + instances[i].translation());

        // auto &leftfingerjoint = this->createSliderJoint(actors[Link_main[8]], actors[Link_main[9]]);
        // leftfingerjoint.setAnchorPoint(Vec3f(0.088f, 0.333f + 0.316 + 0.384f - 0.107f - 0.0584, 0.0f) + instances[i].translation());
        // leftfingerjoint.setAxis(Vec3f(0.7071f, 0.0f, 0.7071f));
        // leftfingerjoint.setRange(0.0, 0.0);

        auto &leftfingerjoint = this->createFixedJoint(actors[Link_main[8]], actors[Link_main[9]]);
        leftfingerjoint.setAnchorPoint(Vec3f(0.0f, 0.333f + 0.316 + 0.384f - 0.107f - 0.0584, 0.088f) + instances[i].translation());

        // auto &rightfingerjoint = this->createSliderJoint(actors[Link_main[8]], actors[Link_main[10]]);
        // rightfingerjoint.setAnchorPoint(Vec3f(0.088f, 0.333f + 0.316 + 0.384f - 0.107f - 0.0584, 0.0f) + instances[i].translation());
        // rightfingerjoint.setAxis(Vec3f(-0.7071f, 0.0f, -0.7071f));
        // rightfingerjoint.setRange(0.0, 0.0);

        auto &rightfingerjoint = this->createFixedJoint(actors[Link_main[8]], actors[Link_main[10]]);
        rightfingerjoint.setAnchorPoint(Vec3f(0.0f, 0.333f + 0.316 + 0.384f - 0.107f - 0.0584, 0.088f) + instances[i].translation());

        // create target
        SphereInfo target;
        target.radius = 0.05f;

        RigidBodyInfo target_rb;
        target_rb.position = instances[i].translation() + varTargetCenter()->getValue() + Vec3f(0.45f, -1.05f, 0.0f);
        target_rb.motionType = BodyType::Static; 
        target_rb.collisionMask = CT_Disabled;
        auto target_actor = this->addSphere(target, target_rb);
        // }

        // auto targetPos = ->addNode(std::make_shared<SphereModel<DataType3f>>());
        // target->varRadius()->setValue(0.05f);
        // target->varCenter()->setValue(Vec3f(0.2f, 0.5f, 0.0f));

        // target->stateTriangleSet()->connect(this->inTriangleSet());

		//**************************************************//
		ArticulatedBody<TDataType>::resetStates();
	}

    template<typename TDataType>
    RobotArmSimulator<TDataType>::~RobotArmSimulator() {
        terminateSimulation();
    }

    template<typename TDataType>
    int RobotArmSimulator<TDataType>::generateRigidID() {
        return nextRigidID++;
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::createScene() {
        scn = std::make_shared<SceneGraph>();
        scn->setGravity(Vec3f(0.0f, 0.0f, 0.0f));
        rigidSystems.clear();
        nextRigidID = 0;
    }

    template<typename TDataType>
    typename RobotArmSimulator<TDataType>::RigidSystemData RobotArmSimulator<TDataType>::createSingleRigidSystem(int index, const Vec3f& offset, Vec3f& targetPosition, float density) {
    
        RigidSystemData data;
        
        // 创建RigidBodySystem节点

        data.robot = scn->addNode(std::make_shared<RobotArmSimulator<DataType3f>>());
        // data.robot = scn->addNode(std::make_shared<ArticulatedBody<DataType3f>>());

        // 计算当前实例的基础位置
        Vec3f basePos = Vec3f(index * offset.x - 0.45f, offset.y + 1.05f, index * offset.z);

        // 设置机械臂初始位置和姿态
        std::vector<Transform3f> transforms(1);
        transforms[0] = Transform3f(basePos, 
                                    Mat3f(1.0f, 0.0f, 0.0f, 
                                            0.0f, 1.0f, 0.0f, 
                                            0.0f, 0.0f, 1.0f), 
                                    Vec3f(0.0f, 0.0f, 0.0f));
        
        data.robot->varVehiclesTransform()->setValue(transforms);

        data.robot->varDensity()->setValue(density);

        // fingerPosition(index);

        // data.robot->varFingerCenter()->setValue(fingerPosition(index));

        // targetPosition += Vec3f(0.45f, -1.125f, 0.0f);

        data.robot->varTargetCenter()->setValue(targetPosition);

        // data.robot->varTargetCenter()->setValue(targetPosition);

        data.system = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
        data.system->varGravityEnabled()->setValue(false);

        data.robot->connect(data.system->importVehicles());
        auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
        plane->varLengthX()->setValue(50);
        plane->varLengthZ()->setValue(50);
        plane->varSegmentX()->setValue(10);
        plane->varSegmentZ()->setValue(10);

        plane->stateTriangleSet()->connect(data.system->inTriangleSet());

        return data;
    }

    template<typename TDataType>
    int RobotArmSimulator<TDataType>::addRigidSystem(const Vec3f& offset, Vec3f& targetPosition, float density) {
        if (!scn) {
            createScene();
        }
        m_offset = offset;
        int rigidID = generateRigidID();
        // std::cout << "Add Rigid System with ID: " << rigidID << std::endl;
        rigidSystems[rigidID] = createSingleRigidSystem(rigidID, m_offset, targetPosition, density);
        // computeJointInitia(rigidID);
        std::vector<float> moterVelocities_tmp(7, 0.0f);
        motersVelocity.push_back(moterVelocities_tmp);
        return rigidID;
    }

    // template<typename TDataType>
    // void RobotArmSimulator<TDataType>::computeJointInitia(int rigidID) {
        
    //     std::vector<float> jointsInitia_temp;
        
    //     auto hingeJoints = rigidSystems[rigidID].robot->getHingeJoints();
        
    //     auto instances = rigidSystems[rigidID].robot->varVehiclesTransform()->getValue();
        
    //     float joint1_init = computeHingeEffectiveInertiaWorld(rigidID, hingeJoints[0], Vec3f(0.0f, 0.333f, 0.0f) + instances[0].translation());
    //     jointsInitia_temp.push_back(joint1_init);
    //     // float joint2_init = computeHingeEffectiveInertiaWorld(rigidID,hingeJoints[1], Vec3f(0.0f, 0.333f, 0.0f) + instances[0].translation());
    //     // jointsInitia_temp.push_back(joint2_init);
    //     // float joint3_init = computeHingeEffectiveInertiaWorld(rigidID,hingeJoints[2], Vec3f(0.0f, 0.333f + 0.316f, 0.0f) + instances[0].translation());
    //     // jointsInitia_temp.push_back(joint3_init);
    //     // float joint4_init = computeHingeEffectiveInertiaWorld(rigidID,hingeJoints[3], Vec3f(0.0825f, 0.333f + 0.316f, 0.0f) + instances[0].translation());
    //     // jointsInitia_temp.push_back(joint4_init);
    //     // float joint5_init = computeHingeEffectiveInertiaWorld(rigidID,hingeJoints[4], Vec3f(0.0f, 0.333f + 0.316f + 0.384f, 0.0f) + instances[0].translation());
    //     // jointsInitia_temp.push_back(joint5_init);
    //     float joint5_init = computeHingeEffectiveInertiaWorld(rigidID,hingeJoints[1], Vec3f(0.0f, 0.333f + 0.316f + 0.384f, 0.0f) + instances[0].translation());
    //     jointsInitia_temp.push_back(joint5_init);
    //     // float joint6_init = computeHingeEffectiveInertiaWorld(rigidID,hingeJoints[5], Vec3f(0.0f, 0.333f + 0.316f + 0.384f, 0.0f) + instances[0].translation());
    //     // jointsInitia_temp.push_back(joint6_init);
    //     // float joint7_init = computeHingeEffectiveInertiaWorld(rigidID,hingeJoints[6], Vec3f(0.088f, 0.333f + 0.316 + 0.384f, 0.0f)+ instances[0].translation());
    //     // jointsInitia_temp.push_back(joint7_init);

    //     jointsInitia.push_back(jointsInitia_temp);
    // }



    template<typename TDataType>
    void RobotArmSimulator<TDataType>::setupSceneGraph() {
        app.setSceneGraph(scn);
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::initialize(int width, int height, float scale) {
        app.initialize(width, height);
        app.renderWindow()->getCamera()->setUnitScale(scale);
        std::cout << "Initializing..." << std::endl;
        isInitialized = true;
        activeScene = SceneGraphFactory::instance()->active();
        std::cout << "resetting scene..." << std::endl;
        activeScene->reset();
        std::cout << "Initialization Done!" << std::endl;
    }

    // template<typename TDataType>
    // void RobotArmSimulator<TDataType>::setMoters(std::vector<std::vector<float>>& moterImpulses) {
    //     for (int i = 0; i < rigidSystems.size(); ++i) {
    //         auto topo = rigidSystems[i].system->stateTopology()->getDataPtr();
    //         auto hingeJoints = rigidSystems[i].robot->getHingeJoints();
            
    //         computeJointInitia(i);

    //         // for (int j = 0; j < hingeJoints.size(); ++j) {
    //         motersVelocity[i][1] += moterImpulses[i][1]/jointsInitia[i][0]/60.0f;
    //         // std::cout << "Computed Effective Inertia: " << jointsInitia[i][0] << std::endl;
    //         motersVelocity[i][4] += moterImpulses[i][4]/jointsInitia[i][1]/60.0f;
    //         hingeJoints[0].setMoter(motersVelocity[i][1]);
    //         hingeJoints[1].setMoter(motersVelocity[i][4]);
    //         // std::cout << "Moter Velocity: " << motersVelocity[i][1] << std::endl;
    //         // hingeJoints[0].setRange(hingeJointsMinAngles[j], hingeJointsMaxAngles[j]);
            
    //         // }
    //         topo->hingeJoints().assign(hingeJoints);
    //     }
    // }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::applyImpulse(std::vector<std::vector<float>>& moterImpulses) {

        for (int i = 0; i < rigidSystems.size(); ++i) {
            float rigidbodys = rigidSystems[i].system->stateExternalTorque()->size();
            // std::cout << "number of rigidbodys: " << rigidbodys << std::endl;
            std::vector<Vec3f> systemForces(rigidbodys, Vec3f(0.0f, 0.0f, 0.0f));

            systemForces[1] = Vec3f(
                -moterImpulses[i][1],
                0.0f,
                0.0f);

            systemForces[2] = Vec3f(
                moterImpulses[i][1],
                0.0f,
                0.0f);

            systemForces[4] = Vec3f(
                0.0f,
                -moterImpulses[i][4],
                0.0f);

            systemForces[5] = Vec3f(
                0.0f,
                moterImpulses[i][4],
                0.0f);

            rigidSystems[i].system->stateExternalTorque()->assign(systemForces);

            // rigidSystems[i].system->stateExternalTorque()->assign(systemForces);
            // auto externalTorques = rigidSystems[i].system->stateExternalTorque()->getDataPtr();
            // std::cout << "value of external torques: " << externalTorques->begin()[2] << std::endl;
        }
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::stepSimulation(std::vector<std::vector<float>>& deltaMoterVelocities, bool enableRendering) {
        if (!isInitialized) return;
        
        if (enableRendering) {
            // 处理事件
            glfwPollEvents();

            // applyImpulse(deltaMoterVelocities);

            if (activeScene) {

                activeScene->takeOneFrame();
                activeScene->updateGraphicsContext();
            }
            
            // 获取渲染窗口
            GlfwRenderWindow* renderWindow = dynamic_cast<GlfwRenderWindow*>(app.renderWindow());
            if (!renderWindow) return;
            
            // 获取相机
            auto camera = renderWindow->getCamera();
            if (!camera) return;
            
            // 更新渲染参数
            RenderParams& renderParams = renderWindow->getRenderParams();
            renderParams.width = camera->viewportWidth();
            renderParams.height = camera->viewportHeight();
            
            // 窗口最小化时跳过渲染
            if (renderParams.width == 0 || renderParams.height == 0) {
                return;
            }
            
            // 设置渲染变换矩阵
            renderParams.transforms.model = glm::mat4(1.0f);
            renderParams.transforms.view = camera->getViewMat();
            renderParams.transforms.proj = camera->getProjMat();
            renderParams.unitScale = camera->unitScale();
            
            // 绘制场景
            auto renderEngine = renderWindow->getRenderEngine();
            if (renderEngine && activeScene) {
                renderEngine->draw(activeScene.get(), renderParams);
            }
            
            // 处理ImGui界面
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            
            if (renderWindow->showImGUI()) {
                renderWindow->imWindow()->draw(renderWindow);
            }
            
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            
            // 交换缓冲区
            GLFWwindow* window = renderWindow->getGLFWWindow();
            if (window) {
                glfwSwapBuffers(window);
            }
            
            // setMoters(deltaMoterVelocities);
            // applyImpulse(deltaMoterVelocities);
            
        } else {
            if (activeScene) {
                activeScene->takeOneFrame();
                activeScene->updateGraphicsContext();
                // setMoters(deltaMoterVelocities);
            }
        }
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::terminateSimulation() {
        GlfwRenderWindow* renderWindow = dynamic_cast<GlfwRenderWindow*>(app.renderWindow());
        if (renderWindow) {
            GLFWwindow* window = renderWindow->getGLFWWindow();
            if (window) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
        }
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::reset(int rigidID, Vec3f& targetPosition) {
        if (rigidID != -1) {
            if (rigidSystems.find(rigidID) != rigidSystems.end()) {
                // targetPosition += Vec3f(0.45f, -1.125f, 0.0f);
                rigidSystems[rigidID].robot->varTargetCenter()->setValue(targetPosition);
                motersVelocity[rigidID] = std::vector<float>(7, 0.0f);
                activeScene->reset(rigidSystems[rigidID].robot);
            }
        } else {
            for (auto& [id, data] : rigidSystems) {
                activeScene->reset();
            }
        }
    }

    // inline Mat3f outer(const Vec3f& a, const Vec3f& b)
    // {
    //     Mat3f m;
    //     m(0,0) = a[0]*b[0]; m(0,1) = a[0]*b[1]; m(0,2) = a[0]*b[2];
    //     m(1,0) = a[1]*b[0]; m(1,1) = a[1]*b[1]; m(1,2) = a[1]*b[2];
    //     m(2,0) = a[2]*b[0]; m(2,1) = a[2]*b[1]; m(2,2) = a[2]*b[2];
    //     return m;
    // }

    // template<typename TDataType>
    // Mat3f RobotArmSimulator<TDataType>::parallelAxisTheoremWorld(const Mat3f& I_world_about_ref, Real mass, const Vec3f& com_world, const Vec3f& pointO_world)
    // {
    //     Vec3f d = com_world - pointO_world;
    //     Real d2 = dot(d, d);
    //     Mat3f I3 = Mat3f::identityMatrix();
    //     return I_world_about_ref + mass * (d2 * I3 - outer(d, d));
    // }

    // template<typename TDataType>
    // float RobotArmSimulator<TDataType>::computeHingeEffectiveInertiaWorld(int rigidID, const HingeJoint<Real>& joint, const Vec3f& jointPositionWorld) 
    // {
    //     // 1) body->world rotation matrices
    //     Mat3f R_A = joint.actor1->rot.toMatrix3x3();
    //     Mat3f R_B = joint.actor2->rot.toMatrix3x3();
    
    //     auto actor1 = joint.actor1;
    //     auto actor2 = joint.actor2;

    //     if (rigidSystems.empty()) {
    //         return 0.0f; // 或适当的错误处理
    //     }
    //     auto mHostRigidBodyStates = rigidSystems[rigidID].robot->getRigidBodyStates();

    //     auto bodyA = mHostRigidBodyStates[actor1->idx];
    //     auto bodyB = mHostRigidBodyStates[actor2->idx];
    
    //     // 2) 获取各自关于其 body-ref 的惯量并把方向转换到世界系
    //     Mat3f I_A_world_ref = bodyA.inertia;
    //     Mat3f I_B_world_ref = bodyB.inertia;

    //     // std::cout << "I_A_world_ref: " << '/n' << I_A_world_ref[0][0] << '.' << I_A_world_ref[0][1] << '.' << I_A_world_ref[0][2] << '/n' << std::endl;
    //     // std::cout << "I_B_world_ref: " << I_B_world_ref << std::endl;
  
    //     // 3) 质心（world） - 需要你工程里能拿到全局质心位置。这里用 actor->center（如果它是world）
    //     Vec3f comA_world = joint.actor1->center; // 若不是，请改成 bodyA.worldCenter 或 bodyA.position_of_COM
    //     Vec3f comB_world = joint.actor2->center;
 
    //     // 4) 把惯量搬到关节点 jointPositionWorld
    //     Mat3f I_A_about_O = parallelAxisTheoremWorld(I_A_world_ref, bodyA.mass, comA_world, jointPositionWorld);
    //     Mat3f I_B_about_O = parallelAxisTheoremWorld(I_B_world_ref, bodyB.mass, comB_world, jointPositionWorld);

    //     // 5) 关节轴（world） - 以 actor1 的 local axis 转换到 world
    //     Vec3f axis_world = R_A * joint.hingeAxisBody1;
    //     axis_world.normalize();

    //     // 6) 等效惯量（标量）
    //     Mat3f I_sum = I_A_about_O + I_B_about_O;
    //     Real I_eff = dot(axis_world, (I_sum * axis_world));

    //     // 如果数值非常小或出现 NaN，做保护
    //     const Real eps = (Real)1e-9;
    //     if (I_eff < eps) I_eff = eps;

    //     return I_eff;
    // }

    // template<typename TDataType>

    // Vec3f RobotArmSimulator<TDataType>::fingerPosition(int rigidID) {
    //     auto instances = rigidSystems[rigidID].robot->varVehiclesTransform()->getValue();
    //     auto size = rigidSystems[rigidID].system->gethCenters()->size();
    //     // std::cout << "size: " << size << std::endl;
    //     Vec3f leftFingerPos = rigidSystems[rigidID].system->gethCenters()->begin()[size - 2] - instances[0].translation();
    //     Vec3f rightFingerPos = rigidSystems[rigidID].system->gethCenters()->begin()[size - 1] - instances[0].translation();
        
    //     return ((leftFingerPos + rightFingerPos) / 2.0f + Vec3f(-0.45, 1.05, 0.0));
    // }

    // template<typename TDataType>
    // Vec3f RobotArmSimulator<TDataType>::rigidPosition(int systemID, int rigidID) {
    //     // Check bounds for systemID
    //     if (systemID < 0 || systemID >= rigidSystems.size()) {
    //         std::cerr << "Invalid systemID: " << systemID << std::endl;
    //         return Vec3f(0.0f);
    //     }
        
    //     auto instances = rigidSystems[systemID].robot->varVehiclesTransform()->getValue();
    //     auto size = rigidSystems[systemID].system->gethCenters()->size();
        
    //     // Check bounds for rigidID
    //     if (rigidID < 0 || rigidID >= static_cast<int>(size)) {
    //         std::cerr << "Invalid rigidID: " << rigidID << std::endl;
    //         return Vec3f(0.0f);
    //     }
        
    //     // Check if instances vector is not empty
    //     if (instances.empty()) {
    //         std::cerr << "Instances vector is empty" << std::endl;
    //         return Vec3f(0.0f);
    //     }
        
    //     Vec3f rigidPosition = rigidSystems[systemID].system->gethCenters()->begin()[rigidID] - instances[0].translation();
        
    //     const Vec3f OFFSET(-0.45, 1.05, 0.0);
    //     return (rigidPosition + OFFSET);
    // }

    // template<typename TDataType>
    // typename RobotArmSimulator<TDataType>::TQuat RobotArmSimulator<TDataType>::rigidRotation(int systemID, int rigidID) {
    //     // Validate systemID bounds
    //     if (systemID < 0 || systemID >= static_cast<int>(rigidSystems.size())) {
    //         std::cerr << "Invalid systemID:" << systemID << std::endl;
    //         return TQuat(); // Returning default constructed quaternion
    //     }
        
    //     // Validate rigidID bounds
    //     auto angels = rigidSystems[systemID].system->gethAngels();
    //     if (rigidID < 0 || rigidID >= static_cast<int>(angels->size())) {
    //         std::cerr << "Invalid rigidID:" << rigidID << std::endl;
    //         return TQuat(); // Returning default constructed quaternion
    //     }
        
    //     TQuat rigidRotation = angels->begin()[rigidID];
        
    //     return rigidRotation;
    // }

    // template<typename TDataType>
    // Vec3f RobotArmSimulator<TDataType>::rigidVelocity(int systemID, int rigidID) {
    //     // Check bounds for systemID
    //     if (systemID < 0 || systemID >= rigidSystems.size()) {
    //         std::cerr << "Invalid systemID: " << systemID << std::endl;
    //         return Vec3f(0.0f);
    //     }
    //     auto size = rigidSystems[systemID].system->gethVelocities()->size();
        
    //     // Check bounds for rigidID
    //     if (rigidID < 0 || rigidID >= static_cast<int>(size)) {
    //         std::cerr << "Invalid rigidID: " << rigidID << std::endl;
    //         return Vec3f(0.0f);
    //     }
        
    //     Vec3f rigidVelocity = rigidSystems[systemID].system->gethVelocities()->begin()[rigidID];
        
    //     return (rigidVelocity);
    // }

    // template<typename TDataType>
    // Vec3f RobotArmSimulator<TDataType>::rigidAngularVelocity(int systemID, int rigidID) {
    //     // Check bounds for systemID
    //     if (systemID < 0 || systemID >= rigidSystems.size()) {
    //         std::cerr << "Invalid systemID: " << systemID << std::endl;
    //         return Vec3f(0.0f);
    //     }
    //     auto size = rigidSystems[systemID].system->gethAngularVelocity()->size();
        
    //     // Check bounds for rigidID
    //     if (rigidID < 0 || rigidID >= static_cast<int>(size)) {
    //         std::cerr << "Invalid rigidID: " << rigidID << std::endl;
    //         return Vec3f(0.0f);
    //     }
        
    //     Vec3f rigidAngularVelocity = rigidSystems[systemID].system->gethAngularVelocity()->begin()[rigidID];
        
    //     return (rigidAngularVelocity);
    // }

    DEFINE_CLASS(RobotArmSimulator);
}
    