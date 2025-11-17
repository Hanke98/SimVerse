import RobotArm
import time

def main():
    # 创建机械臂仿真器实例，对应C++的 RobotArmSimulator<DataType3f> simulator;
    simulator = RobotArm.RobotArmSimulator()
    
    # 1. 创建场景，对应C++的 simulator.createScene();
    simulator.createScene()
    print("场景创建完成")
    
    # 2. 配置机械臂参数及添加系统
    # 机械臂基座偏移（对应C++的 Vec3f offset(1.0f, 0.0f, 0.0f)）
    offset = RobotArm.Vec3f(1.0, 0.0, 0.0)
    # 目标位置（对应C++的 Vec3f targetPos(-0.2f, 0.2f, 0.0f)）
    target_pos = RobotArm.Vec3f(-0.2, 0.2, 0.0)
    
    # 电机速度设置（对应C++的vector容器）
    moter_velocities1 = [0.0, 87.0, -87.0, 49.0441, 12.0, -12.0, 1.9525]
    moter_velocities2 = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
    moter_velocities = [moter_velocities1, moter_velocities2]
    
    # 添加第一个机械臂系统
    rigid_id1 = simulator.addRigidSystem(offset, target_pos)
    print(f"创建机械臂系统，ID: {rigid_id1}")
    
    # 添加第二个机械臂系统
    rigid_id2 = simulator.addRigidSystem(offset, target_pos)
    print(f"创建机械臂系统，ID: {rigid_id2}")
    
    # 3. 初始化仿真环境（对应C++的setupSceneGraph和initialize）
    simulator.setupSceneGraph()
    simulator.initialize(1280, 768, 2.5)  # 窗口大小1280x768
    print("仿真环境初始化完成")
    
    # 4. 主仿真循环（对应C++的while循环）
    print("开始仿真循环（按ESC退出，或Ctrl+C终止）")
    i = 0
    try:
        while True:
            # 执行单步仿真（启用渲染）
            simulator.stepSimulation(moter_velocities, True)
            
            # 定期重置第一个机械臂（每50步）
            if i % 50 == 0:
                finger_pos = simulator.fingerPosition(0)
                simulator.reset(0, finger_pos)
            
            # 在300步时重置第二个机械臂
            if i == 300:
                simulator.reset(1, RobotArm.Vec3f(0.2, 0.2, 0.0))
            
            i += 1
            # 控制仿真帧率（简单延时，类似C++的sleep）
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        # 捕获键盘中断（Ctrl+C），优雅退出
        print("\n用户终止仿真")
    
    # 5. 终止仿真（析构函数会自动处理，但显式调用更安全）
    simulator.terminateSimulation()
    print("仿真结束")

if __name__ == "__main__":
    main()