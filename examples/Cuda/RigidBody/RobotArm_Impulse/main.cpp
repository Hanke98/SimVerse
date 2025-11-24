#include "DataTypes.h"
#include "RobotArm.h"
#include <GLFW/glfw3.h>
#include <iostream>

using namespace dyno;

int main() {
    getchar();
    Real kp = 5;
    Real kv = 2;
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
    UrdfInformation chainInfo = simulator.getKinematicsChainInfo();
    
    // 4. 主仿真循环
    std::cout << "开始仿真循环（按ESC退出）" << std::endl;
    int i = 0;
    Real roll_old = 0.0f, pitch_old, yaw_old = 0.0f;
    Real hingeAngle_old2 = 0.0f;
    Real hingeAngle_old5 = 0.0f;
    Quat<Real> qRel5_old(0, 0, 0, 1);

    Real torque[7] = { Real(0) };
    std::vector<float> hingeAngle_old(7, 0.0f);

    // Vec3f jointAxisLocal[7] = {
    //     Vec3f(0, 1, 0),   // joint 0: link0->link1
    //     Vec3f(1, 0, 0),   // joint 1: link1->link2
    //     Vec3f(0, 1, 0),   // joint 2: link2->link3
    //     Vec3f(-1, 0, 0),   // joint 3: link3->link4
    //     Vec3f(0, 1, 0),   // joint 4: link4->link5
    //     Vec3f(-1, 0, 0),   // joint 5: link5->link6
    //     Vec3f(0, -1, 0)    // joint 6: link6->link7
    // };
    Vec3f jointAxisLocal[7] = {
        chainInfo.joints[0].axisWorld,   // joint 0: link0->link1
        chainInfo.joints[1].axisWorld,   // joint 1: link1->link2
        chainInfo.joints[2].axisWorld,   // joint 2: link2->link3
        chainInfo.joints[3].axisWorld,   // joint 3: link3->link4
        chainInfo.joints[4].axisWorld,   // joint 4: link4->link5
        chainInfo.joints[5].axisWorld,   // joint 5: link5->link6
        chainInfo.joints[6].axisWorld    // joint 6: link6->link7
    };

    float effortUpperLimit[7] = {
        chainInfo.joints[0].limits.upper,   // joint 0: link0->link1
        chainInfo.joints[1].limits.upper,   // joint 1: link1->link2
        chainInfo.joints[2].limits.upper,   // joint 2: link2->link3
        chainInfo.joints[3].limits.upper,   // joint 3: link3->link4
        chainInfo.joints[4].limits.upper,   // joint 4: link4->link5
        chainInfo.joints[5].limits.upper,   // joint 5: link5->link6
        chainInfo.joints[6].limits.upper    // joint 6: link6->link7
    };

    float effortLowerLimit[7] = {
        chainInfo.joints[0].limits.lower,   // joint 0: link0->link1
        chainInfo.joints[1].limits.lower,   // joint 1: link1->link2
        chainInfo.joints[2].limits.lower,   // joint 2: link2->link3
        chainInfo.joints[3].limits.lower,   // joint 3: link3->link4
        chainInfo.joints[4].limits.lower,   // joint 4: link4->link5
        chainInfo.joints[5].limits.lower,   // joint 5: link5->link6
        chainInfo.joints[6].limits.lower    // joint 6: link6->link7
    };

    //
    Real targetAngle[7] = {
        Real(0.0),    // joint 0
        Real(0.3),    // joint 1
        Real(0.0),    // joint 2
        Real(-0.5),    // joint 3
        Real(0.0),    // joint 4
        Real(0.0),    // joint 5
        Real(0.1)     // joint 6
    };

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

        // ------------- loop calculating 7 joint ----------------
        for (int j = 0; j < 7; ++j)
        {
            int parentId = j;       // 父连杆：j
            int childId  = j + 1;   // 子连杆：j+1

            auto quatParent = quat[parentId];
            auto quatChild  = quat[childId];

            // 相对旋转：child 相对于 parent
            auto qRel = quatParent.inverse() * quatChild;
            qRel.normalize();

            // 铰链轴在 world 坐标系下的方向
            auto axisWorld = quatParent.rotate(jointAxisLocal[j]);
            axisWorld.normalize();

            // 相对旋转 → 轴角
            Real rot;
            Vec3f axisRel;
            qRel.toRotationAxis(rot, axisRel);

            // 通过和铰链轴的点积确定“正方向”
            Real sign      = axisRel.dot(axisWorld) > 0 ? Real(1) : Real(-1);
            Real rawAngle  = sign * rot;

            // 用上一帧角度解包，得到连续的关节角
            Real hingeAngle = unwrapAngle(rawAngle, hingeAngle_old[j]);

            // PD 控制
            Real e  = hingeAngle - targetAngle[j];
            Real de = hingeAngle - hingeAngle_old[j];
            torque[j] = - kp * e - kv * de * Real(100);
            torque[j] = std::max(effortLowerLimit[j], std::min(effortUpperLimit[j], torque[j]));
            // std::cout << "torque of joint " << j << " is: " << torque[j] << std::endl;

            // 更新上一帧角度
            hingeAngle_old[j] = hingeAngle;

            if (j == 3) {
                // std::cout << "hingeAngle of joint " << j << " is: " << hingeAngle << std::endl;
            }
        }

        // moterVelocities1 = {
        //     (float)torque[0],
        //     (float)torque[1],
        //     (float)torque[2],
        //     (float)torque[3],
        //     (float)torque[4],
        //     (float)torque[5],
        //     (float)torque[6]
        // };
        moterVelocities1 = {
            (float)0.0f,
            (float)torque[1],
            (float)0.0f,
            (float)torque[3],
            (float)0.0f,
            (float)0.0f,
            (float)0.0f
        };

        // moterVelocities1 = {0.0f, (float)torque[2], 0.0f, 0.0f, (float)torque[4], 0.0f, 0.0f};
        // moterVelocities.pop_back();
        // moterVelocities.push_back(moterVelocities1);

        i++;
    }
    
    // 5. 仿真结束（析构函数会自动调用terminateSimulation）
    std::cout << "仿真结束" << std::endl;
    return 0;
}
