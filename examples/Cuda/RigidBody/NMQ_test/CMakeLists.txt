if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY
            Modeling
            RigidBody
            ObjIO
            UbiGUI)
    add_example(NMQ_test RigidBody LIB_DEPENDENCY)
endif()