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
#include "RigidBody/Vehicle.h"
#include <SceneGraphFactory.h> 

#include <BasicShapes/PlaneModel.h>


namespace dyno
{
    IMPLEMENT_TCLASS(RobotArmSimulator, TDataType)

    template<typename TDataType>
    RobotArmSimulator<TDataType>::RobotArmSimulator() :
		ArticulatedBody<TDataType>()
    {
	}

    template<typename TDataType>
    RobotArmSimulator<TDataType>::~RobotArmSimulator() {
        terminateSimulation();
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::initBatchSolver(Real dt, Real density){
        batchSolver = std::make_shared<BatchRigidBodySystem<TDataType>>();
        batchSolver->setDt(dt);
        batchSolver->varGravityEnabled()->setValue(false);
        batchSolver->varFrictionEnabled()->setValue(false);
        Vec3f base{ -0.0f, -0.0f, -0.0f };
        Vec3f offset{ 0.0f, 0.0f, 2.0f };
        batchSolver->addExampleRigidBodies("", base, offset, density, 1, 1, 1);
        scn->addNode(batchSolver);
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::resetStates(CtrlParam& param) {
        batchSolver->resetBatchMultiBodies(param);
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::applyHingeTorques(CtrlHingeParam& param) {
        batchSolver->applyHingeTorqueControl(param);
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

        // data.system = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
        // data.system->varGravityEnabled()->setValue(false);

        // data.robot->connect(data.system->importVehicles());
        // auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
        // plane->varLengthX()->setValue(50);
        // plane->varLengthZ()->setValue(50);
        // plane->varSegmentX()->setValue(10);
        // plane->varSegmentZ()->setValue(10);
        //
        // plane->stateTriangleSet()->connect(data.system->inTriangleSet());

        return data;
    }

    template<typename TDataType>
    int RobotArmSimulator<TDataType>::addMultiBoydSystem() {

        mbSystem = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
        mbSystem->varGravityEnabled()->setValue(false);

	    auto uav = scn->addNode(std::make_shared<UAV<DataType3f>>());

	    std::vector<Transform3f> vehicleTransforms;
	    vehicleTransforms.push_back(Transform3f(Vec3f(0.5, 0, 0), Quat1f(1.57, Vec3f(0, 1, 0)).toMatrix3x3()));
	    vehicleTransforms.push_back(Transform3f(Vec3f(10, 2, 0), Quat1f(0, Vec3f(0, 1, 0)).toMatrix3x3()));
	    vehicleTransforms.push_back(Transform3f(Vec3f(10, 2, 2), Quat1f(0, Vec3f(0, 1, 0)).toMatrix3x3()));
	    uav->varVehiclesTransform()->setValue(vehicleTransforms);

	    uav->connect(mbSystem->importVehicles());

        auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
        plane->varLengthX()->setValue(50);
        plane->varLengthZ()->setValue(50);
        plane->varSegmentX()->setValue(10);
        plane->varSegmentZ()->setValue(10);

        plane->stateTriangleSet()->connect(mbSystem->inTriangleSet());

        return 0;
    }

    template<typename TDataType>
    int RobotArmSimulator<TDataType>::addRigidSystem(const Vec3f& offset, Vec3f& targetPosition, float density) {
        if (!scn) {
            createScene();
        }
        m_offset = offset;
        int rigidID = generateRigidID();
        std::cout << "Add Rigid System with ID: " << rigidID << std::endl;
        rigidSystems[rigidID] = createSingleRigidSystem(rigidID, m_offset, targetPosition, density);
        // computeJointInitia(rigidID);
        std::vector<float> moterVelocities_tmp(7, 0.0f);
        motersVelocity.push_back(moterVelocities_tmp);
        return rigidID;
    }



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

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::applyImpulse(std::vector<std::vector<float>>& moterImpulses) {

        int rigidbodys = mbSystem->stateExternalForce()->size();
        std::vector<Vec3f> systemForces(rigidbodys, Vec3f(0.0f, 0.0f, 0.0f));

        int n = rigidSystems.size();
        rigidbodys /= n;
        int st = 0;
        for (int i = 0; i < rigidSystems.size(); ++i) {
            systemForces[1 + st] = Vec3f(
                -moterImpulses[i][1],
                0.0f,
                0.0f);

            systemForces[2 + st] = Vec3f(
                moterImpulses[i][1],
                0.0f,
                0.0f);

            systemForces[4 + st] = Vec3f(
                0.0f,
                -moterImpulses[i][4],
                0.0f);

            systemForces[5 + st] = Vec3f(
                0.0f,
                moterImpulses[i][4],
                0.0f);

            st += rigidbodys;
        }
        mbSystem->stateExternalTorque()->assign(systemForces);
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::stepSimulation(std::vector<std::vector<float>>& deltaMoterVelocities, bool enableRendering) {
        if (!isInitialized) return;

        // applyImpulse(deltaMoterVelocities);

        if (activeScene) {

            activeScene->takeOneFrame();
            activeScene->updateGraphicsContext();
        }
        
        if (enableRendering) {
            // 处理事件
            glfwPollEvents();

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

    template<typename TDataType>
    std::vector<typename RobotArmSimulator<TDataType>::TQuat> RobotArmSimulator<TDataType>::getAngelsByLocalIndex(
        LocalIndexParam& param) {
        auto returnAngels = batchSolver->getAngelsByLocalIndex(param);
        return returnAngels;
    }

    template<typename TDataType>
    std::vector<Vec3f> RobotArmSimulator<TDataType>::getAngularVelocitiesByLocalIndex(
        LocalIndexParam& param) {
        auto AngularVelocities = batchSolver->getAngularVelocitiesByLocalIndex(param);
        return AngularVelocities;
    }

    template<typename TDataType>
    std::vector<float> RobotArmSimulator<TDataType>::getMassByLocalIndex(
        LocalIndexParam& param) {
        auto Mass = batchSolver->getMassByLocalIndex(param);
        return Mass;
    }

    template<typename TDataType>
    UrdfInformation RobotArmSimulator<TDataType>::getKinematicsChainInfo() {
        return batchSolver->urdfInfo;
    }

    DEFINE_CLASS(RobotArmSimulator);
}
    
