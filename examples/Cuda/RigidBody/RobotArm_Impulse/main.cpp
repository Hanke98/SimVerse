#include "DataTypes.h"
#include "RobotArm.h"
#include <GLFW/glfw3.h>
#include <iostream>

using namespace dyno;

int main() {
    getchar();
    Real kp = 100;
    Real kv = 40;
    // 创建机械臂仿真器实例
    RobotArmSimulator<DataType3f> simulator;
    
    // 1. 创建场景
    simulator.createScene();
    std::cout << "场景创建完成" << std::endl;
    
    // simulator.initBatchSolver();

    // 2. 添加机械臂系统
    Vec3f offset(1.0f, 0.0f, 0.0f);    // 机械臂基座偏移
    Vec3f targetPos(0.2, 1.0, 0.3); // 目标位置示例
    std::vector<std::vector<float>> moterVelocities;
    std::vector<float> moterVelocities1{0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int N = 1;
    for (int i = 0; i < N; ++i) {
        moterVelocities.push_back(moterVelocities1);
        simulator.addRigidSystem(offset, targetPos, 50.0f);
    }
    simulator.addMultiBoydSystem();

    // 3. 初始化仿真环境（窗口大小1280x768）
    simulator.setupSceneGraph();
    std::cout << "初始化窗口" << std::endl;
    simulator.initialize(1280, 768, 2.5);
    std::cout << "仿真环境初始化完成" << std::endl;
    
    // 4. 主仿真循环
    std::cout << "开始仿真循环（按ESC退出）" << std::endl;
    int i = 0;
    Real roll_old, pitch_old, yaw_old = 0.0f;

    while (!glfwWindowShouldClose(glfwGetCurrentContext())) {
        simulator.stepSimulation(moterVelocities, true);
        // 处理窗口事件
        glfwPollEvents();
    }
    
    // 5. 仿真结束（析构函数会自动调用terminateSimulation）
    std::cout << "仿真结束" << std::endl;
    return 0;
}
