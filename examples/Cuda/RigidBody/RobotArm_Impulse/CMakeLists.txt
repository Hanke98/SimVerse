if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        Modeling
        RigidBody
        ObjIO
        UbiGUI)
    add_example(RobotArm_Impulse RigidBody LIB_DEPENDENCY)

    # ===========================
    #  下面是新增的 pybind11 模块
    # ===========================

    # 找到 pybind11
#    find_package(pybind11 CONFIG REQUIRED)
#    if(NOT pybind11_FOUND)
    # message(STATUS "pybind11 not found, using embedded version")
    add_subdirectory(
            /home/wjv/SimVerse_01/SimVerse/external/pybind11
            ${CMAKE_CURRENT_BINARY_DIR}/pybind11
            EXCLUDE_FROM_ALL
    )
#    endif()

    # 创建 Python 模块 RobotArm
    # 对应在 robot_arm_pybind.cpp 里写的：
    # PYBIND11_MODULE(RobotArm, m) { ... }
    pybind11_add_module(RobotArm
            robot_arm_pybind.cpp
            RobotArm.cpp      # 绑定里用到了 RobotArm.cpp 里的实现
    )

    # 链接依赖库（和 example 用一样的一套）
    target_link_libraries(RobotArm
            PRIVATE
            ${LIB_DEPENDENCY}
    )

    # 把 robot_arm_pybind.cpp 从 RobotArm_Impulse 这个可执行里移除，
    # 防止它也去编译 pybind 源码导致 Python.h 报错。
    get_target_property(RA_SRCS RobotArm_Impulse SOURCES)

    # CMake 一般会存成绝对路径，这里两种写法都试一下：
    list(REMOVE_ITEM RA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/robot_arm_pybind.cpp)
    list(REMOVE_ITEM RA_SRCS robot_arm_pybind.cpp)

    set_target_properties(RobotArm_Impulse PROPERTIES
        SOURCES "${RA_SRCS}"
    )
endif()
