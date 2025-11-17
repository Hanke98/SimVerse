#include "DataTypes.h"
#include "RobotArm.h"
#include <GLFW/glfw3.h>
#include <iostream>

using namespace dyno;

int main() {
    Real kp = 100;
    Real kv = 40;
    // 创建机械臂仿真器实例
    RobotArmSimulator<DataType3f> simulator;
    
    // 1. 创建场景
    simulator.createScene();
    std::cout << "场景创建完成" << std::endl;
    
    // 2. 添加机械臂系统
    Vec3f offset(1.0f, 0.0f, 0.0f);    // 机械臂基座偏移
    Vec3f targetPos(0.2, 1.0, 0.3); // 目标位置示例
    std::vector<std::vector<float>> moterVelocities;
    std::vector<float> moterVelocities1{0.0f, 14.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    moterVelocities.push_back(moterVelocities1);
    int rigidID1 = simulator.addRigidSystem(offset, targetPos, 50.0f);
    std::cout << "创建机械臂系统，ID: " << rigidID1 << std::endl;


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
        
        // std::this_thread::sleep_for(std::chrono::milliseconds(50000));
        // Quat<Real> quat_1 = simulator.rigidRotation(0, 1);
        // Quat<Real> quat_2 = simulator.rigidRotation(0, 2);
        // Quat<Real> q1_inv = quat_1.inverse();
        // SquareMatrix<Real, 3> q1_inv_matrix = q1_inv.toMatrix3x3();
        // Quat<Real> q_rel = q1_inv * quat_2;
        // Real angle = q_rel.angle();
        // Real roll, pitch, yaw;
        // q_rel.toEulerAngle(roll, pitch, yaw);
        // // angle = (float)(angle + M_PI) % (float)(2 * M_PI) - M_PI;
        // // std::cout << "Angle: " << angle << std::endl;
        // std::cout << "rpy: " << roll << " " <<pitch << " " << yaw << std::endl;

        // std::cout << "rotation matrix: \n" << q1_inv_matrix << std::endl;

        // Vec3f ang_vel = simulator.rigidAngularVelocity(0, 2);
        // std::cout << "Ang_Vel2: " << ang_vel.x << " " << ang_vel.y << " " << ang_vel.z << std::endl;
        // std::cout << "Ang_Vel_yaw: " << (yaw - yaw_old) * 100 << std::endl;
        // Vec3f ang_vel1 = simulator.rigidAngularVelocity(0, 1);
        // std::cout << "Ang_Vel1: " << ang_vel1.x << " " << ang_vel1.y << " " << ang_vel1.z << std::endl;

        // Real torque = - kp * (yaw - 0.5) - kv * ang_vel.x;
        // Real torque = - kp * (yaw - 0.5) - kv * (yaw - yaw_old) * 100;
        // yaw_old = yaw;
        // std::cout << "spring force: " << - kp * (yaw - 0.1) << " " << "damping force: " << kv * ang_vel.x << std::endl;
        // torque = torque <= 87.0 ? torque : 87.0;
        // torque = torque >= -87.0 ? torque : -87.0;
        // printf("Torque: %f\n", torque);


        // moterVelocities1 = {0.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        // moterVelocities.pop_back();
        // moterVelocities.push_back(moterVelocities1);

        i++;
    }
    
    // 5. 仿真结束（析构函数会自动调用terminateSimulation）
    std::cout << "仿真结束" << std::endl;
    return 0;
}