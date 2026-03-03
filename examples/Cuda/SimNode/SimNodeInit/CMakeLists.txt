if(PERIDYNO_LIBRARY_SIMNODE)
    set(LIB_DEPENDENCY 
        SimNode
        GlfwGUI)
    add_example(SimNodeInit SimNode LIB_DEPENDENCY)
endif()