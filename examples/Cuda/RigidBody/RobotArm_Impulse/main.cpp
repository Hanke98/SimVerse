#include "DataTypes.h"
#include "RobotArm.h"
#include <GLFW/glfw3.h>
#include <iostream>

using namespace dyno;

int main() {
    Real scale = 1.0f;
    Real kp[7] = {
        100 * scale,
        100 * scale,
        50 * scale,
        100 * scale,
        50 * scale, // joint 4
        100 * scale,
        20 * scale,
    };
    Real kd[7] = {
        20 * scale,
        10 * scale,
        2 * scale,
        10 * scale,
        2 * scale,  //joint 4
        8 * scale,
        1 * scale,
    };

    // 创建机械臂仿真器实例
    RobotArmSimulator<DataType3f> simulator;

    // 1. 创建场景
    simulator.createScene();
    std::cout << "场景创建完成" << std::endl;

    float dt = 0.01;
    float density = 2500.0f;
    bool enableGravity = false;
    bool enableFriction = false;
    bool enableRendering = true;
    Vec3f base{ -0.0f, -0.0f, -0.0f };
    Vec3f offset{ 1.5f, 0.0f, 1.5f };
    std::vector<Vec3f> target_position;
    Vec3f target1{0.5f, 0.5f, 0.5f};
    target_position.push_back(target1);
    target_position.push_back(target1);
    int num_copies_x = 2;
    int num_copies_y = 1;
    int num_copies_z = 2;
    std::string urdf_fn = "../asset/franka_description/robots/franka_panda_custom.urdf";
    bool render_boundingbox = false;

    simulator.initBatchSolver();
    simulator.setDt(dt);
    simulator.enableGravity(enableGravity);
    simulator.enableFriction(enableFriction);
    simulator.setTransform(base, offset, num_copies_x, num_copies_y, num_copies_z);
    simulator.setAngularDamping(50.0);
    simulator.addRobotArmRigidBodies(urdf_fn, density, target_position, render_boundingbox);

    //
    std::vector<float> moterVelocities1{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> dampings{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0};

    //
    simulator.setupSceneGraph();
    std::cout << "初始化窗口" << std::endl;
    if (enableRendering) {
        simulator.initialize(1280, 768, 1.5);
    }
    std::cout << "仿真环境初始化完成" << std::endl;
    UrdfInformation chainInfo = simulator.getKinematicsChainInfo();
    
    // 4. 主仿真循环
    std::cout << "开始仿真循环（按ESC退出）" << std::endl;
    int i = 0;

    Real torque[7] = { Real(0) };
    std::vector<float> hingeAngle_old(7, 0.0f);
    std::vector<float> error(7, 0.0f);

    Vec3f jointAxisLocal[7] = {
        chainInfo.joints[0].axisWorld,   // joint 0: link0->link1
        chainInfo.joints[1].axisWorld,   // joint 1: link1->link2
        chainInfo.joints[2].axisWorld,   // joint 2: link2->link3
        chainInfo.joints[3].axisWorld,   // joint 3: link3->link4
        chainInfo.joints[4].axisWorld,   // joint 4: link4->link5
        chainInfo.joints[5].axisWorld,   // joint 5: link5->link6
        chainInfo.joints[6].axisWorld    // joint 6: link6->link7
    };

    Real effortLimit[7] = {
        chainInfo.joints[0].limits.effort,   // joint 0: link0->link1
        chainInfo.joints[1].limits.effort,   // joint 1: link1->link2
        chainInfo.joints[2].limits.effort,   // joint 2: link2->link3
        chainInfo.joints[3].limits.effort,   // joint 3: link3->link4
        chainInfo.joints[4].limits.effort,   // joint 4: link4->link5
        chainInfo.joints[5].limits.effort,   // joint 5: link5->link6
        chainInfo.joints[6].limits.effort    // joint 6: link6->link7
    };

    Real targetAngle[7] = {
        Real(1.0),    // joint 0
        Real(1.3),    // joint 1
        Real(0.3),    // joint 2
        Real(-0.3),    // joint 3
        Real(0.3),    // joint 4
        Real(1.8),    // joint 5
        Real(0.1)     // joint 6
    };

    int checkFrequancy = 100;
    int best_idx = 0;

    while (!glfwWindowShouldClose(glfwGetCurrentContext())) {

        if (i == 500) {
            RobotArmSimulator<DataType3f>::ResetParam param;
            param.num_bodies = 2;
            param.ids.push_back(0);
            param.ids.push_back(1);
            Vec3f newTarget{ 0.5f, 1.0f, 0.5f };
            param.targetPosition.push_back(newTarget);
            param.targetPosition.push_back(newTarget);
            simulator.resetStates(param);
        }

        RobotArmSimulator<DataType3f>::LocalIndexParam local_param;
        local_param.num_bodies = 2;
        local_param.ids.push_back(0);
        local_param.ids.push_back(1);
        for (int i = 0; i <= 7; ++i) {
            local_param.localRigidBodyid.push_back(i);
        }
        auto quat = simulator.getAnglesByLocalIndex(local_param);
        auto angularVelocity = simulator.getAngularVelocitiesByLocalIndex(local_param);

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

            auto angularVelocityParent = angularVelocity[parentId];
            auto angularVelocityChild = angularVelocity[childId];
            auto angularVelocityRel = angularVelocityChild - angularVelocityParent;
            float hingeVelocity = angularVelocityRel.dot(axisWorld);

            // PD 控制
            Real e  = targetAngle[j] - hingeAngle;
            torque[j] = kp[j] * e - kd[j] * hingeVelocity ;
            torque[j] = std::max(-effortLimit[j]/1, std::min(effortLimit[j]/1, torque[j]));

            // 更新上一帧角度
            hingeAngle_old[j] = hingeAngle;

            if (std::abs(e) > std::abs(error[j])) {
                error[j] = e;
            }

            if (i % checkFrequancy == 0 && j == 6) {
                float best_val = error[0];
                best_idx = 0;
                for (int idx = 1; idx < error.size(); ++idx) {
                    if (std::abs(error[idx]) > std::abs(best_val)) {
                        best_val = error[idx];
                        best_idx = idx;
                    }
                }

                std::cout << "From " << std::max(0, (i / checkFrequancy) * checkFrequancy - checkFrequancy)
                          << " to " << (i / checkFrequancy) * checkFrequancy << " steps: " << "\n"
                          << "The largest error is " << best_val << "\n"
                          << "The joint of largest error is : " << best_idx << "\n"<< std::endl;

                std::fill(error.begin(), error.end(), 0.0f);
            }

            if (j == best_idx && i % 10 == 0) {
                bool sat = std::abs(torque[j]) >= effortLimit[j]/2 - 1e-6;
                std::cout << "step " << i
                          << ", joint " << j
                          << ", err = " << e
                          << ", errV = " << -hingeVelocity
                          << ", torque = " << torque[j]
                          << (sat ? " (SATURATED)" : "")
                          << std::endl;
            }
        }

        moterVelocities1 = {
            (float)torque[0],
            (float)torque[1],
            (float)torque[2],
            (float)torque[3],
            (float)torque[4],
            (float)torque[5],
            (float)torque[6]
        };

        RobotArmSimulator<DataType3f>::HingeTorqueParam param;
        param.num_bodies = 2;
        param.ids.push_back(0);
        param.ids.push_back(1);
        param.torques.push_back(moterVelocities1);
        param.torques.push_back(moterVelocities1);
        simulator.setHingeTorques(param);

        simulator.stepSimulation(enableRendering);

        i++;
    }
    
    // 5. 仿真结束（析构函数会自动调用terminateSimulation）
    std::cout << "仿真结束" << std::endl;
    return 0;
}
