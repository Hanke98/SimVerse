set(LIB_DEPENDENCY
    Core
    Framework
    SimNode
    gtest)
add_peridyno_test(Test_Cholesky LIB_DEPENDENCY)

target_link_libraries(Test_Cholesky
    cublas
    cusolver)
