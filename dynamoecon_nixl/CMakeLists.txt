cmake_minimum_required(VERSION 3.20)

set(DYNAMOECON_SOURCES
    src/radix_kv_indexer.cpp
    src/crossover_scheduler.cpp
    src/nixl_transfer_engine.cpp
    src/telemetry_hal.cpp
    src/econ_telemetryd.cpp
    src/failover_manager.cpp
    src/dynamoecon_c_api.cpp
)

add_library(dynamoecon_core SHARED ${DYNAMOECON_SOURCES})

target_include_directories(dynamoecon_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_definitions(dynamoecon_core PRIVATE DYNAMOECON_BUILDING_DLL=1)

if(OpenMP_CXX_FOUND)
    target_link_libraries(dynamoecon_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Standalone C++20 telemetry daemon executable (FR-2.4: econ_telemetryd)
add_executable(econ_telemetryd src/econ_telemetryd_main.cpp)
target_link_libraries(econ_telemetryd PRIVATE dynamoecon_core)
