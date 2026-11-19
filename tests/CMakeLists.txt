cmake_minimum_required(VERSION 3.20)

find_package(GTest QUIET)

add_executable(test_turbokv_fp4 test_turbokv_fp4.cpp)
target_include_directories(test_turbokv_fp4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_turbokv_fp4 PRIVATE turbokv_sim)
target_compile_definitions(test_turbokv_fp4 PRIVATE PROJECT_SOURCE_ROOT="${CMAKE_SOURCE_DIR}")

add_executable(test_dynamoecon_nixl test_dynamoecon_nixl.cpp)
target_include_directories(test_dynamoecon_nixl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_dynamoecon_nixl PRIVATE dynamoecon_core)

if(GTest_FOUND)
    target_compile_definitions(test_turbokv_fp4 PRIVATE USE_SYSTEM_GTEST=1)
    target_link_libraries(test_turbokv_fp4 PRIVATE GTest::gtest_main)
    target_compile_definitions(test_dynamoecon_nixl PRIVATE USE_SYSTEM_GTEST=1)
    target_link_libraries(test_dynamoecon_nixl PRIVATE GTest::gtest_main)
endif()

add_test(NAME TurboKV_FP4_Verification COMMAND test_turbokv_fp4)
add_test(NAME DynamoEcon_NIXL_Verification COMMAND test_dynamoecon_nixl)
