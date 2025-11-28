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
    RobotArmSimulator<TDataType>::RobotArmSimulator()
    {
	}

    template<typename TDataType>
    RobotArmSimulator<TDataType>::~RobotArmSimulator() {
        terminateSimulation();
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::initBatchSolver(){
        batchSolver = scn->addNode(std::make_shared<BatchRigidBodySystem<TDataType>>());
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::setDt(Real dt) {
        batchSolver->setDt(dt);
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::enableGravity(bool flag) {
        batchSolver->varGravityEnabled()->setValue(flag);
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::enableFriction(bool flag) {
        batchSolver->varFrictionEnabled()->setValue(flag);
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::addRobotArmRigidBodies(std::string urdf_fn,
                                                              Real density,
                                                              const std::vector<Vec3f> &target_position,
                                                              bool render_boundingbox) {
        batchSolver->addRobotArmRigidBodies(urdf_fn, density, target_position, render_boundingbox);
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::resetStates(ResetParam& param) {
        batchSolver->resetBatchMultiBodies(param);
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::setHingeTorques(HingeTorqueParam& hingetorque_param) {
        batchSolver->applyHingeTorqueControl(hingetorque_param);
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::setMass(MassParam& mass_param) {
        batchSolver->setMass(mass_param);
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::setInertia(InertiaParam& inertia_param) {
        batchSolver->setInertia(inertia_param);
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::setTransform(Vec3f base, Vec3f offset,int num_copies_x, int num_copies_y, int num_copies_z) {
        std::vector<Transform3f> transforms;
        for (int x = 0; x < num_copies_x; x++) {
            for (int y = 0; y < num_copies_y; y++) {
                for (int z = 0; z < num_copies_z; z++) {
                    Vec3f _offset = base + Vec3f(x * offset.x, y * offset.y, z * offset.z);
                    Transform3f transform(_offset,
                                         Mat3f(1.0f, 0.0f, 0.0f,
                                                  0.0f, 1.0f, 0.0f,
                                                  0.0f, 0.0f, 1.0f),
                                         Vec3f(0.0f, 0.0f, 0.0f));
                    transforms.push_back(transform);
                }
            }
        }
        batchSolver->varVehiclesTransform()->setValue(transforms);
    }

    template<typename TDataType>
    std::vector<Transform3f> RobotArmSimulator<TDataType>::getTransform(CtrlParam& param) {
        std::vector<Transform3f> transforms;
        auto instances = batchSolver->varVehiclesTransform()->getValue();
        for (int i = 0; i < param.num_bodies; i++) {
            transforms.push_back(instances[param.ids[i]]);
        }
        return transforms;
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::createScene() {
        scn = std::make_shared<SceneGraph>();
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::setupSceneGraph() {
        app.setSceneGraph(scn);
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::initialize(int width, int height, Real scale) {
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
    void RobotArmSimulator<TDataType>::stepSimulation(bool enableRendering) {
        if (!isInitialized) return;

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
    std::vector<Vec3f> RobotArmSimulator<TDataType>::getCentersByLocalIndex(
        LocalIndexParam& param) {
        return batchSolver->getCentersByLocalIndex(param);
    }

    template<typename TDataType>
    std::vector<Vec3f> RobotArmSimulator<TDataType>::getVelocitiesByLocalIndex(
        LocalIndexParam& param) {
        return batchSolver->getVelocitiesByLocalIndex(param);
    }

    template<typename TDataType>
    std::vector<std::vector<typename RobotArmSimulator<TDataType>::Real> > RobotArmSimulator<
        TDataType>::getAnglesVectorByLocalIndex(
        LocalIndexParam &param) {
        auto angles = batchSolver->getAnglesByLocalIndex(param);
        std::vector<std::vector<Real>> angles_quat_vector;
        for (int i = 0; i < angles.size(); ++i) {
            angles_quat_vector.push_back(std::vector{angles[i].x, angles[i].y, angles[i].z, angles[i].w});
        }
        return angles_quat_vector;
    }

    template<typename TDataType>
    std::vector<typename RobotArmSimulator<TDataType>::TQuat> RobotArmSimulator<TDataType>::getAnglesByLocalIndex(
        LocalIndexParam& param) {
        return batchSolver->getAnglesByLocalIndex(param);
    }

    template<typename TDataType>
    std::vector<Vec3f> RobotArmSimulator<TDataType>::getAngularVelocitiesByLocalIndex(
        LocalIndexParam& param) {
        return batchSolver->getAngularVelocitiesByLocalIndex(param);
    }

    template<typename TDataType>
    std::vector<typename RobotArmSimulator<TDataType>::Real> RobotArmSimulator<TDataType>::getMassByLocalIndex(
        LocalIndexParam &param) {
        return batchSolver->getMassByLocalIndex(param);
    }

    template<typename TDataType>
    UrdfInformation RobotArmSimulator<TDataType>::getKinematicsChainInfo() {
        return batchSolver->urdfInfo;
    }

    template<typename TDataType>
    void RobotArmSimulator<TDataType>::setAngularDamping(Real damping) {
        batchSolver->varAngularDamping()->setValue(damping);
    }

    DEFINE_CLASS(RobotArmSimulator);
}
    
