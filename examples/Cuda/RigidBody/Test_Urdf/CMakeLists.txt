if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        Modeling
        RigidBody
        ObjIO
        UbiGUI)
    add_example(Test_Urdf RigidBody LIB_DEPENDENCY)
endif()