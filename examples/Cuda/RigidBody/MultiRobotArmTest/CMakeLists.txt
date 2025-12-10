if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY
            Modeling
            RigidBody
            ObjIO
            UbiGUI)
    add_example(MultiRobotArm RigidBody LIB_DEPENDENCY)
endif()