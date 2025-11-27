import os
import math
import RobotArm as ra  # 你的 pybind 模块名

def vec_sub(a, b):
    return ra.Vec3f(a.x - b.x, a.y - b.y, a.z - b.z)

def vec_dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z

def vec_norm(v):
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

def vec_normalize(v):
    n = vec_norm(v)
    if n < 1e-8:
        return ra.Vec3f(0.0, 0.0, 0.0)
    return ra.Vec3f(v.x / n, v.y / n, v.z / n)

def unwrap_angle(angle, prev_angle):
    """
    与 C++ 中的 unwrapAngle 完全对应：
    在候选 ±angle, ±(angle+2π), ±(angle-2π) 中找离 prev_angle 最近的那个
    """
    two_pi = 2.0 * math.pi
    cand = [
        angle,
        -angle,
        angle + two_pi,
        -(angle + two_pi),
        angle - two_pi,
        -(angle - two_pi),
    ]

    best = cand[0]
    best_diff = abs(cand[0] - prev_angle)
    for c in cand[1:]:
        d = abs(c - prev_angle)
        if d < best_diff:
            best = c
            best_diff = d
    return best

def main():
    # ==========================
    # 0. 控制参数（照抄 C++）
    # ==========================
    scale = 1.0

    kp = [
        100 * scale,
        100 * scale,
        50 * scale,
        100 * scale,
        50 * scale,  # joint 4
        100 * scale,
        20 * scale,
    ]
    kd = [
        20 * scale,
        10 * scale,
        2 * scale,
        10 * scale,
        2 * scale,  # joint 4
        8 * scale,
        1 * scale,
    ]

    dt = 0.01
    density = 2500.0
    enable_gravity = False
    enable_friction = False

    base = ra.Vec3f(-0.0, -0.0, -0.0)
    offset = ra.Vec3f(1.5, 0.0, 1.5)

    target_position = []
    target1 = ra.Vec3f(0.5, 0.5, 0.5)
    target_position.append(target1)
    target_position.append(target1)

    num_copies_x = 2
    num_copies_y = 1
    num_copies_z = 2

    # URDF 路径
    urdf_fn = "../asset/franka_description/robots/franka_panda_custom.urdf"
    urdf_fn = os.path.normpath(urdf_fn)
    print("使用 URDF:", urdf_fn)

    # ==========================
    # 1. 创建仿真器 & 场景
    # ==========================
    sim = ra.RobotArmSimulator()

    sim.createScene()
    print("场景创建完成")

    sim.initBatchSolver()
    sim.setDt(dt)
    sim.enableGravity(enable_gravity)
    sim.enableFriction(enable_friction)
    sim.setTransform(base, offset, num_copies_x, num_copies_y, num_copies_z)
    sim.setAngularDamping(50.0)
    sim.addRobotArmRigidBodies(urdf_fn, density, target_position)

    sim.setupSceneGraph()
    print("初始化窗口")
    sim.initialize(1280, 768, 1.5)
    print("仿真环境初始化完成")

    chain_info = sim.getKinematicsChainInfo()

    # -----------------------------
    # 关节轴 + effortLimit + 目标角度
    # -----------------------------
    joint_axis_local = []  # 实际上是 world 轴，命名沿用 C++
    effort_limit = []
    for j in range(7):
        joint = chain_info.joints[j]
        joint_axis_local.append(joint.axisWorld)
        effort_limit.append(joint.limits.effort)

    target_angle = [
        1.0,   # joint 0
        1.3,   # joint 1
        0.3,   # joint 2
        -0.3,  # joint 3
        0.3,   # joint 4
        1.8,   # joint 5
        0.1,   # joint 6
    ]

    hinge_angle_old = [0.0 for _ in range(7)]
    error = [0.0 for _ in range(7)]
    check_frequency = 100
    best_idx = 0

    print("开始仿真循环（固定步数，按 Ctrl+C 退出 Python）")

    max_steps = 2000  # 没有 glfwWindowShouldClose，只跑固定步

    for step in range(max_steps):
        # -------------------------
        # 重置：step == 500
        # -------------------------
        if step == 500:
            reset = ra.ResetParam()
            reset.num_bodies = 2
            reset.ids = [0, 1]
            new_target = ra.Vec3f(0.5, 1.0, 0.5)
            reset.targetPosition = [new_target, new_target]
            print("调用 resetStates, targetPosition size =", len(reset.targetPosition))
            sim.resetStates(reset)

        # -------------------------
        # 构造 LocalIndexParam
        # -------------------------
        local_param = ra.LocalIndexParam()
        local_param.num_bodies = 2
        local_param.ids = [0, 1]
        local_param.localRigidBodyid = list(range(8))  # 0..7

        quats = sim.getAnglesByLocalIndex(local_param)        # std::vector<TQuat>
        ang_vel = sim.getAngularVelocitiesByLocalIndex(local_param)  # std::vector<Vec3f>

        # -------------------------
        # 七个关节的 PD 控制
        # -------------------------
        torques = [0.0 for _ in range(7)]

        for j in range(7):
            parent_id = j
            child_id = j + 1

            quat_parent = quats[parent_id]
            quat_child = quats[child_id]

            # 相对旋转：qRel = qParent^{-1} * qChild
            q_rel = quat_parent.inverse() * quat_child
            q_rel.normalize()

            # 铰链轴在 world 坐标系下的方向
            axis_world = quat_parent.rotate(joint_axis_local[j])
            axis_world = vec_normalize(axis_world)

            # qRel -> 轴角
            rot, axis_rel = q_rel.to_rotation_axis()
            axis_rel = vec_normalize(axis_rel)

            # 点积决定正负
            sign = 1.0 if vec_dot(axis_rel, axis_world) > 0.0 else -1.0
            raw_angle = sign * rot

            # 连续关节角
            hinge_angle = unwrap_angle(raw_angle, hinge_angle_old[j])

            # 相对角速度
            ang_v_parent = ang_vel[parent_id]
            ang_v_child = ang_vel[child_id]
            ang_v_rel = vec_sub(ang_v_child, ang_v_parent)
            hinge_velocity = vec_dot(ang_v_rel, axis_world)

            # PD 控制
            e = target_angle[j] - hinge_angle
            torque = kp[j] * e - kd[j] * hinge_velocity

            limit = effort_limit[j] if effort_limit[j] > 0 else 1e6
            if torque > limit:
                torque = limit
            if torque < -limit:
                torque = -limit

            torques[j] = float(torque)
            hinge_angle_old[j] = hinge_angle

            # 误差最大值记录
            if abs(e) > abs(error[j]):
                error[j] = e

            # 打印最大的那个关节的状态（类似 C++）
            if j == best_idx and step % 10 == 0:
                sat = abs(torque) >= limit / 2.0 - 1e-6
                print(
                    f"step {step}, joint {j}, "
                    f"err = {e:.4f}, errV = {-hinge_velocity:.4f}, "
                    f"torque = {torque:.4f}{' (SATURATED)' if sat else ''}"
                )

        # 每隔 check_frequency 步，对 error 做统计
        if step % check_frequency == 0 and step > 0:
            best_val = error[0]
            best_idx = 0
            for idx in range(1, len(error)):
                if abs(error[idx]) > abs(best_val):
                    best_val = error[idx]
                    best_idx = idx

            print(
                f"\nFrom {max(0, (step // check_frequency) * check_frequency - check_frequency)} "
                f"to {(step // check_frequency) * check_frequency} steps:\n"
                f"  The largest error is {best_val}\n"
                f"  The joint of largest error is: {best_idx}\n"
            )
            # 清零 error
            for k in range(7):
                error[k] = 0.0

        # -------------------------
        # 写入 HingeTorqueParam
        # -------------------------
        hinge_param = ra.HingeTorqueParam()
        hinge_param.num_bodies = 2
        hinge_param.ids = [0, 1]
        hinge_param.torques = [
            torques[:],  # 对 arm0
            torques[:],  # 对 arm1
        ]
        sim.setHingeTorques(hinge_param)

        # -------------------------
        # 单步仿真 & 渲染
        # -------------------------
        sim.stepSimulation(True)

    print("仿真结束")

if __name__ == "__main__":
    main()