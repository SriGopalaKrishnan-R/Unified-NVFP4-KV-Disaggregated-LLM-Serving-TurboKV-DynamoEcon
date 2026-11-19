#pragma once

#include <cstdint>
#include <atomic>
#include <bit>

#if defined(__CUDACC__)
    #include <cuda_runtime.h>
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
    #define TURBOKV_HOST_DEVICE __host__ __device__ __forceinline__
    #define TURBOKV_DEVICE __device__ __forceinline__
    #define TURBOKV_GLOBAL __global__
#else
    #define TURBOKV_HOST_DEVICE inline
    #define TURBOKV_DEVICE inline
    #define TURBOKV_GLOBAL

    #ifndef __CUDA_RUNTIME_H__
    struct alignas(16) uint4 {
        uint32_t x, y, z, w;
    };
    inline uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) noexcept {
        return uint4{x, y, z, w};
    }
    #endif
#endif

#if defined(_WIN32)
    #if defined(TURBOKV_BUILDING_DLL)
        #define TURBOKV_API __declspec(dllexport)
    #else
        #define TURBOKV_API
    #endif
#else
    #define TURBOKV_API __attribute__((visibility("default")))
#endif

namespace turbokv::hal {

// NFR-1.4 Verification: Tracks host-device synchronizations on the critical path.
// Decode + Speculative Verification paths MUST maintain zero host sync calls.
inline std::atomic<uint64_t>& host_sync_counter() noexcept {
    static std::atomic<uint64_t> counter{0};
    return counter;
}

inline void reset_host_sync_counter() noexcept {
    host_sync_counter().store(0, std::memory_order_relaxed);
}

inline uint64_t get_host_sync_counter() noexcept {
    return host_sync_counter().load(std::memory_order_relaxed);
}

inline void recorded_device_synchronize() noexcept {
    host_sync_counter().fetch_add(1, std::memory_order_relaxed);
#if defined(__CUDACC__)
    cudaDeviceSynchronize();
#endif
}

// Hardware PTX byte-permute (prmt.b32) & 3-input LUT logic (lop3.b32) on SM89/SM90/SM100
// with cycle/bit-exact host C++20 fallback for CPU-Sim verification.
TURBOKV_HOST_DEVICE uint32_t ptx_prmt_b32(uint32_t a, uint32_t b, uint32_t selector) noexcept {
#if defined(__CUDA_ARCH__)
    uint32_t ret;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(a), "r"(b), "r"(selector));
    return ret;
#else
    uint64_t combined = (static_cast<uint64_t>(b) << 32) | static_cast<uint64_t>(a);
    uint32_t out = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t ctrl = (selector >> (i * 4)) & 0xFu;
        uint32_t idx  = ctrl & 0x7u;
        uint8_t byte_val = static_cast<uint8_t>((combined >> (idx * 8)) & 0xFFu);
        if (ctrl & 0x8u) {
            byte_val = (byte_val & 0x80u) ? 0xFFu : 0x00u; // sign-extend mode in PTX prmt
        }
        out |= (static_cast<uint32_t>(byte_val) << (i * 8));
    }
    return out;
#endif
}

// Emulates PTX lop3.b32 d, a, b, c, imm8
template <uint8_t Imm8>
TURBOKV_HOST_DEVICE uint32_t ptx_lop3_b32(uint32_t a, uint32_t b, uint32_t c) noexcept {
#if defined(__CUDA_ARCH__)
    uint32_t ret;
    asm("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(ret) : "r"(a), "r"(b), "r"(c), "n"(Imm8));
    return ret;
#else
    uint32_t res = 0;
    if constexpr (Imm8 == 0xEA) {
        // (a & b) | c
        res = (a & b) | c;
    } else if constexpr (Imm8 == 0xC0) {
        // a & b
        res = a & b;
    } else {
        for (int bit = 0; bit < 32; ++bit) {
            uint32_t ia = (a >> bit) & 1u;
            uint32_t ib = (b >> bit) & 1u;
            uint32_t ic = (c >> bit) & 1u;
            uint32_t lut_idx = (ia << 2) | (ib << 1) | ic;
            if ((Imm8 >> lut_idx) & 1u) {
                res |= (1u << bit);
            }
        }
    }
    return res;
#endif
}

// Host-compatible __ffs (1-indexed least-significant set bit, 0 if x == 0)
TURBOKV_HOST_DEVICE int warp_ffs(uint32_t mask) noexcept {
#if defined(__CUDA_ARCH__)
    return __ffs(static_cast<int>(mask));
#else
    return (mask == 0u) ? 0 : (std::countr_zero(mask) + 1);
#endif
}

TURBOKV_HOST_DEVICE int warp_ffs64(uint64_t mask) noexcept {
#if defined(__CUDA_ARCH__)
    return __ffsll(static_cast<long long>(mask));
#else
    return (mask == 0ULL) ? 0 : (std::countr_zero(mask) + 1);
#endif
}

} // namespace turbokv::hal
