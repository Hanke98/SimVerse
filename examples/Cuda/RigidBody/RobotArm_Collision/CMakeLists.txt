if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        Modeling
        RigidBody
        ObjIO
        UbiGUI)
    add_example(RobotArm_Collision RigidBody LIB_DEPENDENCY)

    # # ===========================
    # #  下面是新增的 pybind11 模块
    # # ===========================

    # # 创建 Python 模块 RobotArm_pybind
    # pybind11_add_module(RobotArm_pybind
    #         robot_arm_pybind.cpp
    #         RobotArm.cpp
    # )

    # # 链接依赖库（和 example 用一样的一套）
    # target_link_libraries(RobotArm_pybind
    #         PRIVATE
    #         ${LIB_DEPENDENCY}
    # )

    # # 把 robot_arm_pybind.cpp 从 RobotArm_Collision 这个可执行里移除，
    # get_target_property(RA_SRCS RobotArm_Collision SOURCES)
    # list(REMOVE_ITEM RA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/robot_arm_pybind.cpp)

    # set_target_properties(RobotArm_Collision PROPERTIES
    #     SOURCES "${RA_SRCS}"
    # )
endif()
