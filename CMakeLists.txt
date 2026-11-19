cmake_minimum_required(VERSION 3.20)
project(NV_DEVTECH_TOKCONOMICS_2026 VERSION 3.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==============================================================================
# NV-DEVTECH-SRS-2026-V3 Dual-Target Hardware-Agnostic Build Options (FR-1.4)
# ==============================================================================
option(USE_CUDA "Compile native SM89/SM90/SM100 CUDA C++20 kernels with PTXAS diagnostics" OFF)
option(USE_CPU_SIM "Compile OpenMP + AVX2/NEON C++20 bit-exact reference & cycle simulation engine" ON)
option(USE_PYBIND11 "Build PyBind11 / Torch custom operator bindings" OFF)
option(BUILD_TESTING "Build C++20 GoogleTest-compatible verification suites" ON)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Ada (SM89), Hopper (SM90), Blackwell (SM100)
        set(CMAKE_CUDA_ARCHITECTURES "89;90;100")
    endif()
    # FR-1.4: Emit PTX/SASS register pressure diagnostics and enforce <= 128 regs/thread (>= 50% occupancy)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;--maxrregcount=128;--use_fast_math;-lineinfo>)
    add_compile_definitions(TURBOKV_USE_CUDA=1)
    message(STATUS "[NV-DEVTECH] Target Backend: Native CUDA (SM89/SM90/SM100), maxrregcount=128")
endif()

if(USE_CPU_SIM OR NOT USE_CUDA)
    add_compile_definitions(TURBOKV_USE_CPU_SIM=1)
    message(STATUS "[NV-DEVTECH] Target Backend: Deterministic CPU-Sim (OpenMP + AVX2/NEON C++20)")
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "[NV-DEVTECH] OpenMP enabled: ${OpenMP_CXX_FLAGS}")
endif()

# Compiler SIMD & optimization flags for x86_64 AVX2+FMA or ARM64 NEON
if(MSVC)
    add_compile_options(/W4 /O2 /arch:AVX2 /EHsc)
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter -O3)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
        add_compile_options(-mavx2 -mfma -mf16c)
    endif()
endif()

# Output directories for clean DLL/EXE discovery by Python Control Plane & CTest
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(turbokv_fp4)
add_subdirectory(dynamoecon_nixl)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
