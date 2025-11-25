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
                                                              Vec3f base,
                                                              Vec3f offset,
                                                              float density,
                                                              int num_copies_x,
                                                              int num_copies_y,
                                                              int num_copies_z) {
        batchSolver->addExampleRigidBodies(urdf_fn, base, offset, density, num_copies_x, num_copies_y, num_copies_y);
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
    void RobotArmSimulator<TDataType>::createScene() {
        scn = std::make_shared<SceneGraph>();
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
    std::vector<typename RobotArmSimulator<TDataType>::TQuat> RobotArmSimulator<TDataType>::getAngelsByLocalIndex(
        LocalIndexParam& param) {
        return batchSolver->getAngelsByLocalIndex(param);
    }

    template<typename TDataType>
    std::vector<Vec3f> RobotArmSimulator<TDataType>::getAngularVelocitiesByLocalIndex(
        LocalIndexParam& param) {
        return batchSolver->getAngularVelocitiesByLocalIndex(param);
    }

    template<typename TDataType>
    std::vector<float> RobotArmSimulator<TDataType>::getMassByLocalIndex(
        LocalIndexParam& param) {
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
    
