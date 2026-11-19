cmake_minimum_required(VERSION 3.20)

set(TURBOKV_CPU_SOURCES
    cpu_sim/nvfp4_simd_cpu.cpp
    cpu_sim/roofline_trace_gen.cpp
    bindings/turbokv_c_api.cpp
)

if(USE_CUDA)
    list(APPEND TURBOKV_CUDA_SOURCES
        kernels/kv_prefill_write_fp4.cu
        kernels/kv_extend_workspace.cu
        kernels/kv_decode_fp4_fused.cu
        kernels/spec_tree_verify.cu
    )
    add_library(turbokv_sim SHARED ${TURBOKV_CPU_SOURCES} ${TURBOKV_CUDA_SOURCES})
    set_target_properties(turbokv_sim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    add_library(turbokv_sim SHARED ${TURBOKV_CPU_SOURCES})
endif()

target_include_directories(turbokv_sim PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_definitions(turbokv_sim PRIVATE TURBOKV_BUILDING_DLL=1)

if(OpenMP_CXX_FOUND)
    target_link_libraries(turbokv_sim PUBLIC OpenMP::OpenMP_CXX)
endif()

if(USE_PYBIND11)
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND)
        pybind11_add_module(turbokv_py bindings/turbokv_pybind.cpp)
        target_link_libraries(turbokv_py PRIVATE turbokv_sim)
    endif()
endif()
