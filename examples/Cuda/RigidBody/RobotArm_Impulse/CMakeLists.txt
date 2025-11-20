if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        Modeling
        RigidBody
        ObjIO
        UbiGUI)
    add_example(RobotArm_Impulse RigidBody LIB_DEPENDENCY)
endif()


# # 寻找Python和pybind11
# find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
# find_package(pybind11 REQUIRED)
#
# # 假设PERIDYNO库已经安装或在当前项目中
# if(PERIDYNO_LIBRARY_RIGIDBODY)
#     set(LIB_DEPENDENCY 
#         Modeling
#         RigidBody
#         ObjIO
#         UbiGUI)
# endif()
#
# # 包含目录
# include_directories(
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     ../../../../src  # 根据实际目录结构调整
# )
#
# # 创建Python模块
# pybind11_add_module(RobotArm robot_arm_pybind.cpp RobotArm.cpp)
#
# # 链接库
# target_link_libraries(RobotArm PRIVATE ${LIB_DEPENDENCY})
#
# # 安装配置
# install(TARGETS RobotArm
#         LIBRARY DESTINATION .
#         RUNTIME DESTINATION .)
