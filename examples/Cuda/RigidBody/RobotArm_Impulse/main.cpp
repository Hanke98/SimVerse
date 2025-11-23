#include "DataTypes.h"
#include "RobotArm.h"
#include <GLFW/glfw3.h>
#include <iostream>

using namespace dyno;

int main() {
    getchar();
    Real kp = 1;
    Real kv = 0.4;
    // 创建机械臂仿真器实例
    RobotArmSimulator<DataType3f> simulator;
    
    // 1. 创建场景
    simulator.createScene();
    std::cout << "场景创建完成" << std::endl;
    
    simulator.initBatchSolver();

    // 2. 添加机械臂系统
    Vec3f offset(1.0f, 0.0f, 0.0f);    // 机械臂基座偏移
    Vec3f targetPos(0.2, 1.0, 0.3); // 目标位置示例
    std::vector<std::vector<float>> moterVelocities;
    std::vector<float> moterVelocities1{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // std::vector<float> moterVelocities2{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    // const int N = 1;
    // for (int i = 0; i < N; ++i) {
    //     moterVelocities.push_back(moterVelocities1);
    //     simulator.addRigidSystem(offset, targetPos, 50.0f);
    // }
    // simulator.addMultiBoydSystem();

    // 3. 初始化仿真环境（窗口大小1280x768）
    simulator.setupSceneGraph();
    std::cout << "初始化窗口" << std::endl;
    simulator.initialize(1280, 768, 2.5);
    std::cout << "仿真环境初始化完成" << std::endl;
    
    // 4. 主仿真循环
    std::cout << "开始仿真循环（按ESC退出）" << std::endl;
    int i = 0;
    Real roll_old = 0.0f, pitch_old, yaw_old = 0.0f;
    Real hingeAngle_old2 = 0.0f;
    Real hingeAngle_old5 = 0.0f;
    Quat<Real> qRel5_old(0, 0, 0, 1);

    while (!glfwWindowShouldClose(glfwGetCurrentContext())) {
        // if (i == 100) {
        //     RobotArmSimulator<DataType3f>::CtrlParam param;
        //     param.num_bodies = 1;
        //     param.ids.push_back(0);
        //     simulator.resetStates(param);
        // }
        //
        // if (i == 200) {
        //     RobotArmSimulator<DataType3f>::CtrlParam param;
        //     param.num_bodies = 1;
        //     param.ids.push_back(1);
        //     simulator.resetStates(param);
        // }
        //
        // if (i == 300) {
        //     RobotArmSimulator<DataType3f>::CtrlParam param;
        //     param.num_bodies = 2;
        //     param.ids.push_back(0);
        //     param.ids.push_back(1);
        //     simulator.resetStates(param);
        // }

        RobotArmSimulator<DataType3f>::CtrlHingeParam param;
        param.num_bodies = 1;
        param.ids.push_back(0);
        // param.ids.push_back(1);
        param.torques.push_back(moterVelocities1);
        // param.torques.push_back(moterVelocities2);
        simulator.applyHingeTorques(param);

        simulator.stepSimulation(moterVelocities, true);
        // 处理窗口事件
        glfwPollEvents();

        RobotArmSimulator<DataType3f>::LocalIndexParam local_param;
        local_param.num_bodies = 1;
        local_param.ids.push_back(0);
        for (int i = 0; i <= 7; ++i) {
            local_param.localRigidBodyid.push_back(i);
        }
        auto quat = simulator.getAngelsByLocalIndex(local_param);

        auto unwrapAngle = [](Real angle, Real prevAngle) -> Real
        {
            const Real twoPi = Real(2.0 * M_PI);

            // six candidates：±angle, ±(angle + 2π), ±(angle - 2π)
            Real cand[6] = {
                angle,
                -angle,
                angle + twoPi,
                -(angle + twoPi),
                angle - twoPi,
                -(angle - twoPi)
            };

            Real best = cand[0];
            Real bestDiff = fabs(cand[0] - prevAngle);

            for (int i = 1; i < 6; ++i)
            {
                Real d = fabs(cand[i] - prevAngle);
                if (d < bestDiff)
                {
                    best = cand[i];
                    bestDiff = d;
                }
            }

            return best;
        };

        // ---------------- joint 2 ----------------
        auto quatOfBody1 = quat[1];
        auto quatOfBody2 = quat[2];
        auto qRel2 = quatOfBody1.inverse() * quatOfBody2;
        qRel2.normalize();

        auto axisWorld2 = quatOfBody1.rotate(Vec3f(1.0f, 0.0f, 0.0f));
        axisWorld2.normalize();

        Real rot2;
        Vec3f axisRel2;
        qRel2.toRotationAxis(rot2, axisRel2);
        // std::cout << "axisRel2 is : " << axisRel2 << std::endl;
        Real sign2 = axisRel2.dot(axisWorld2) > 0 ? Real(1) : Real(-1);
        Real rawAngle2 = sign2 * rot2;

        Real hingeAngle2 = unwrapAngle(rawAngle2, hingeAngle_old2);

        std::cout << "hingeAngle of joint 2 is :" << hingeAngle2 << std::endl;

        Real torque2 = - kp * (hingeAngle2 - 0.3)
                       - kv * (hingeAngle2 - hingeAngle_old2) * 100;

        hingeAngle_old2 = hingeAngle2;

        // ---------------- joint 5 ----------------
        auto quatOfBody4 = quat[4];
        auto quatOfBody5 = quat[5];

        auto qRel5 = quatOfBody4.inverse() * quatOfBody5;
        qRel5.normalize();

        auto axisWorld5 = quatOfBody4.rotate(Vec3f(0.0f, 1.0f, 0.0f));
        Real rot5;
        Vec3f axisRel5;
        qRel5.toRotationAxis(rot5, axisRel5);
        // std::cout << "axisRel5 is : " << axisRel5 << std::endl;
        Real sign5 = axisRel5.dot(axisWorld5) > 0 ? Real(1) : Real(-1);
        Real rawAngle5 = sign5 * rot5;

        Real hingeAngle5 = unwrapAngle(rawAngle5, hingeAngle_old5);

        std::cout << "hingeAngle of joint 5 is :" << hingeAngle5 << std::endl;

        Real torque5 = - kp * (hingeAngle5 - 0.5)
                       - kv * (hingeAngle5 - hingeAngle_old5) * 100;

        hingeAngle_old5 = hingeAngle5;

        moterVelocities1 = {0.0f, torque2, 0.0f, 0.0f, torque5, 0.0f, 0.0f};
        // moterVelocities.pop_back();
        // moterVelocities.push_back(moterVelocities1);

        i++;
    }
    
    // 5. 仿真结束（析构函数会自动调用terminateSimulation）
    std::cout << "仿真结束" << std::endl;
    return 0;
}
