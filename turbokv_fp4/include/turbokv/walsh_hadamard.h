#pragma once

#include "turbokv/cuda_compat.h"
#include <cmath>
#include <cstddef>

#if !defined(__CUDACC__) && (defined(__AVX2__) || defined(_M_AVX2))
    #include <immintrin.h>
    #define TURBOKV_HAS_AVX2 1
#endif

namespace turbokv {

// ==============================================================================
// FR-1.1: Fast In-Register / SIMD Orthogonal Walsh-Hadamard Transform (FWHT)
// Head dimensions d in {64, 128}.
//
// Mathematical Invariants:
//   H_d is symmetric orthogonal: \tilde{H}_d = (1 / sqrt(d)) * H_d
//   1) Involution: \tilde{H}_d * (\tilde{H}_d * x) = x
//   2) Norm & Inner-Product Preservation: <\tilde{H}_d q, \tilde{H}_d k> = <q, k>
//   3) Outlier Suppression: Channel spikes (e.g., 50x channel outliers in RoPE / QKV)
//      are spread uniformly across all d channels with crest factor bounded by 1/sqrt(d),
//      preventing NVFP4 micro-block dynamic range collapse with ZERO lookup tables!
// ==============================================================================

template <int HeadDim>
TURBOKV_HOST_DEVICE void fwht_inplace_scalar(float* x) noexcept {
    static_assert(HeadDim == 64 || HeadDim == 128, "FWHT HeadDim must be 64 or 128");
    for (int len = 1; len < HeadDim; len <<= 1) {
        int step = len << 1;
        for (int i = 0; i < HeadDim; i += step) {
            for (int j = 0; j < len; ++j) {
                float u = x[i + j];
                float v = x[i + len + j];
                x[i + j]       = u + v;
                x[i + len + j] = u - v;
            }
        }
    }
    constexpr float kNorm = (HeadDim == 64) ? 0.125f : 0.08838834764831845f; // 1/sqrt(64), 1/sqrt(128)
    for (int i = 0; i < HeadDim; ++i) {
        x[i] *= kNorm;
    }
}

#if defined(TURBOKV_HAS_AVX2)
// AVX2 butterfly stages for d in {64, 128}
template <int HeadDim>
inline void fwht_inplace_avx2(float* x) noexcept {
    static_assert(HeadDim == 64 || HeadDim == 128, "FWHT HeadDim must be 64 or 128");

    // Intra-lane stages len = 1, 2, 4 within each 8-wide __m256 register
    for (int base = 0; base < HeadDim; base += 8) {
        __m256 v = _mm256_loadu_ps(x + base);

        // Stage len = 1: pairs (0,1), (2,3), (4,5), (6,7)
        __m256 even1 = _mm256_moveldup_ps(v);
        __m256 odd1  = _mm256_movehdup_ps(v);
        __m256 sign1 = _mm256_setr_ps(1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f);
        v = _mm256_fmadd_ps(odd1, sign1, even1);

        // Stage len = 2: pairs (0,2), (1,3), (4,6), (5,7)
        __m256 lo2   = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 hi2   = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 sign2 = _mm256_setr_ps(1.f, 1.f, -1.f, -1.f, 1.f, 1.f, -1.f, -1.f);
        v = _mm256_fmadd_ps(hi2, sign2, lo2);

        // Stage len = 4: cross 128-bit lanes (0..3) +/- (4..7)
        __m256 lo4   = _mm256_permute2f128_ps(v, v, 0x00);
        __m256 hi4   = _mm256_permute2f128_ps(v, v, 0x11);
        __m256 sign4 = _mm256_setr_ps(1.f, 1.f, 1.f, 1.f, -1.f, -1.f, -1.f, -1.f);
        v = _mm256_fmadd_ps(hi4, sign4, lo4);

        _mm256_storeu_ps(x + base, v);
    }

    // Inter-register stages: len = 8, 16, 32, (and 64 if HeadDim == 128)
    for (int len = 8; len < HeadDim; len <<= 1) {
        int step = len << 1;
        for (int i = 0; i < HeadDim; i += step) {
            for (int j = 0; j < len; j += 8) {
                __m256 u = _mm256_loadu_ps(x + i + j);
                __m256 w = _mm256_loadu_ps(x + i + len + j);
                _mm256_storeu_ps(x + i + j,       _mm256_add_ps(u, w));
                _mm256_storeu_ps(x + i + len + j, _mm256_sub_ps(u, w));
            }
        }
    }

    constexpr float kNorm = (HeadDim == 64) ? 0.125f : 0.08838834764831845f;
    __m256 vnorm = _mm256_set1_ps(kNorm);
    for (int base = 0; base < HeadDim; base += 8) {
        __m256 v = _mm256_loadu_ps(x + base);
        _mm256_storeu_ps(x + base, _mm256_mul_ps(v, vnorm));
    }
}
#endif

inline void fwht_inplace(float* x, int head_dim) noexcept {
#if defined(TURBOKV_HAS_AVX2)
    if (head_dim == 64) {
        fwht_inplace_avx2<64>(x);
        return;
    }
    if (head_dim == 128) {
        fwht_inplace_avx2<128>(x);
        return;
    }
#endif
    if (head_dim == 64) {
        fwht_inplace_scalar<64>(x);
    } else if (head_dim == 128) {
        fwht_inplace_scalar<128>(x);
    }
}

#if defined(__CUDACC__)
// Warp-register Walsh-Hadamard transform for 32 threads collaborating on HeadDim = 64 or 128
// (2 or 4 elements per lane -> local butterfly + 5 __shfl_xor_sync stages!)
template <int HeadDim>
TURBOKV_DEVICE void fwht_warp_registers(float vals[HeadDim / 32]) noexcept {
    static_assert(HeadDim == 64 || HeadDim == 128, "HeadDim must be 64 or 128");
    constexpr int kElemsPerLane = HeadDim / 32;

    // Intra-thread stages
    if constexpr (kElemsPerLane == 2) {
        float u = vals[0], v = vals[1];
        vals[0] = u + v;
        vals[1] = u - v;
    } else if constexpr (kElemsPerLane == 4) {
        float u0 = vals[0] + vals[1], v0 = vals[0] - vals[1];
        float u1 = vals[2] + vals[3], v1 = vals[2] - vals[3];
        vals[0] = u0 + u1;
        vals[1] = v0 + v1;
        vals[2] = u0 - u1;
        vals[3] = v0 - v1;
    }

    // 5 warp-shuffle butterfly stages across 32 lanes (zero shared memory!)
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int mask = 1; mask < 32; mask <<= 1) {
        bool is_upper = (lane & mask) != 0;
        #pragma unroll
        for (int e = 0; e < kElemsPerLane; ++e) {
            float partner = __shfl_xor_sync(0xFFFFFFFFu, vals[e], mask);
            vals[e] = is_upper ? (partner - vals[e]) : (vals[e] + partner);
        }
    }

    constexpr float kNorm = (HeadDim == 64) ? 0.125f : 0.08838834764831845f;
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        vals[e] *= kNorm;
    }
}
#endif

} // namespace turbokv
