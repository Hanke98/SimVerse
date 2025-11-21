if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY
            Modeling
            RigidBody
            ObjIO
            UbiGUI)
    add_example(CartPole_Urdf RigidBody LIB_DEPENDENCY)
endif()