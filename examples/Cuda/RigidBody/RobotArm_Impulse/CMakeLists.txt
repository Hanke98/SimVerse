if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        Modeling
        RigidBody
        ObjIO
        UbiGUI)
    add_example(RobotArm_Impulse RigidBody LIB_DEPENDENCY)
endif()