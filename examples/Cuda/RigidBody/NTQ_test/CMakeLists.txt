if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY
            Modeling
            RigidBody
            ObjIO
            UbiGUI)
    add_example(NTQ_test RigidBody LIB_DEPENDENCY)
endif()