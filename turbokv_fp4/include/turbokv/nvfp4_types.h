#pragma once

#include "turbokv/cuda_compat.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace turbokv {

// ==============================================================================
// Exact Bit-Level Codecs for Blackwell NVFP4 (E2M1), OFP8 (E4M3FN), and BF16
// Matches PTX cvt.rn.satfinite.e2m1x2.f32 and cvt.rn.satfinite.e4m3x2.f32
// ==============================================================================

constexpr float NVFP4_E2M1_MAX = 6.0f;
constexpr float FP8_E4M3_MAX   = 448.0f;
constexpr int   NVFP4_MICRO_BLOCK_SIZE = 16; // FR-1.1: 1 FP8 E4M3 scale per 16 elements

TURBOKV_HOST_DEVICE uint32_t float_as_u32(float f) noexcept {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(uint32_t));
    return u;
}

TURBOKV_HOST_DEVICE float u32_as_float(uint32_t u) noexcept {
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

// IEEE-754 Round-to-Nearest-Even FP32 -> BF16
TURBOKV_HOST_DEVICE uint16_t fp32_to_bf16_rne(float val) noexcept {
    uint32_t bits = float_as_u32(val);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return static_cast<uint16_t>((bits >> 16) | 0x0040u); // quiet NaN
    }
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounding_bias = 0x00007FFFu + lsb;
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

TURBOKV_HOST_DEVICE float bf16_to_fp32(uint16_t bf16) noexcept {
    return u32_as_float(static_cast<uint32_t>(bf16) << 16);
}

// Exact OFP8 E4M3FN (1 sign, 4 exp bias=7, 3 mantissa, max finite +/-448.0, satfinite)
TURBOKV_HOST_DEVICE uint8_t fp32_to_fp8_e4m3_rne(float val) noexcept {
    uint32_t bits = float_as_u32(val);
    uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
    float abs_v = std::fabs(val);

    if (std::isnan(val)) {
        return static_cast<uint8_t>(sign | 0x7Eu); // satfinite
    }
    if (abs_v >= FP8_E4M3_MAX) {
        return static_cast<uint8_t>(sign | 0x7Eu); // 0x7E = +448.0, 0xFE = -448.0
    }
    // Smallest subnormal step is 2^-9 = 1/512 = 0.001953125
    constexpr float kSubnormalStep = 1.0f / 512.0f;
    if (abs_v < 0.015625f) { // < 2^-6 (subnormal range e=0)
        float scaled = abs_v / kSubnormalStep;
        int m = static_cast<int>(std::floor(scaled));
        float frac = scaled - static_cast<float>(m);
        if (frac > 0.5f || (frac == 0.5f && (m & 1))) {
            ++m;
        }
        if (m >= 8) {
            return static_cast<uint8_t>(sign | 0x08u); // transitions to e=1, m=0 (2^-6)
        }
        return static_cast<uint8_t>(sign | static_cast<uint8_t>(m));
    }

    int exp_unbiased = 0;
    float mant = std::frexp(abs_v, &exp_unbiased); // abs_v = mant * 2^exp_unbiased, mant in [0.5, 1.0)
    mant *= 2.0f;                                  // mant in [1.0, 2.0)
    exp_unbiased -= 1;

    int e_biased = exp_unbiased + 7;
    if (e_biased <= 0) {
        return sign;
    }
    float m_exact = (mant - 1.0f) * 8.0f;
    int m = static_cast<int>(std::floor(m_exact));
    float frac = m_exact - static_cast<float>(m);
    if (frac > 0.5f || (frac == 0.5f && (m & 1))) {
        ++m;
    }
    if (m == 8) {
        m = 0;
        ++e_biased;
    }
    if (e_biased > 15 || (e_biased == 15 && m >= 7)) {
        return static_cast<uint8_t>(sign | 0x7Eu); // satfinite 448.0
    }
    return static_cast<uint8_t>(sign | (static_cast<uint8_t>(e_biased) << 3) | static_cast<uint8_t>(m));
}

TURBOKV_HOST_DEVICE float fp8_e4m3_to_fp32(uint8_t code) noexcept {
    uint32_t sign = (code & 0x80u) ? 0x80000000u : 0u;
    uint32_t exp  = (code >> 3) & 0x0Fu;
    uint32_t mant = code & 0x07u;

    if (exp == 0u) {
        if (mant == 0u) {
            return u32_as_float(sign);
        }
        float val = static_cast<float>(mant) * (1.0f / 512.0f);
        return (sign != 0u) ? -val : val;
    }
    if (exp == 15u && mant == 7u) {
        float val = FP8_E4M3_MAX;
        return (sign != 0u) ? -val : val;
    }
    uint32_t fp32_exp  = (exp - 7u + 127u) << 23;
    uint32_t fp32_mant = mant << 20;
    return u32_as_float(sign | fp32_exp | fp32_mant);
}

// Exact Blackwell NVFP4 E2M1 Round-to-Nearest-Even (RNE)
// Magnitudes: 0:0.0, 1:0.5, 2:1.0, 3:1.5, 4:2.0, 5:3.0, 6:4.0, 7:6.0
// Tie-to-even (m=0 even nibbles: 0, 2, 4, 6):
//   0.25 -> 0 (0.0)
//   0.75 -> 2 (1.0)
//   1.25 -> 2 (1.0)
//   1.75 -> 4 (2.0)
//   2.50 -> 4 (2.0)
//   3.50 -> 6 (4.0)
//   5.00 -> 6 (4.0)
TURBOKV_HOST_DEVICE uint8_t fp32_to_nvfp4_e2m1_rne(float val) noexcept {
    uint32_t bits = float_as_u32(val);
    uint8_t sign = (bits & 0x80000000u) ? 0x08u : 0x00u;
    float a = std::fabs(val);
    if (std::isnan(val)) {
        return static_cast<uint8_t>(sign | 0x07u);
    }

    uint8_t mag = 0u;
    if (a <= 0.25f) {
        mag = 0u;          // <= 0.25 (tie 0.25 -> even m=0 -> 0.0)
    } else if (a < 0.75f) {
        mag = 1u;          // (0.25, 0.75) -> 0.5 (m=1)
    } else if (a <= 1.25f) {
        mag = 2u;          // [0.75, 1.25] (ties 0.75 & 1.25 -> even m=0 -> 1.0)
    } else if (a < 1.75f) {
        mag = 3u;          // (1.25, 1.75) -> 1.5 (m=1)
    } else if (a <= 2.50f) {
        mag = 4u;          // [1.75, 2.50] (ties 1.75 & 2.50 -> even m=0 -> 2.0)
    } else if (a < 3.50f) {
        mag = 5u;          // (2.50, 3.50) -> 3.0 (m=1)
    } else if (a <= 5.00f) {
        mag = 6u;          // [3.50, 5.00] (ties 3.50 & 5.00 -> even m=0 -> 4.0)
    } else {
        mag = 7u;          // > 5.00 -> 6.0 (satfinite)
    }
    return static_cast<uint8_t>(sign | mag);
}

TURBOKV_HOST_DEVICE float nvfp4_e2m1_to_fp32(uint8_t nibble) noexcept {
    static constexpr float kLut[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return kLut[nibble & 0x0Fu];
}

// Pack 8 NVFP4 elements into a single 32-bit word (elem i occupies bits [4*i+3 : 4*i])
TURBOKV_HOST_DEVICE uint32_t pack_nvfp4x8(const float* vals, float inv_effective_scale) noexcept {
    uint32_t word = 0u;
    for (int i = 0; i < 8; ++i) {
        uint8_t q = fp32_to_nvfp4_e2m1_rne(vals[i] * inv_effective_scale);
        word |= (static_cast<uint32_t>(q & 0x0Fu) << (i * 4));
    }
    return word;
}

// ==============================================================================
// FR-1.2 Path 3 PTX Register Unpacking (prmt.b32 + lop3.b32, ZERO Shared Memory)
// Unpacks 8 packed E2M1 nibbles in `packed_word` into 8 exact FP32 values scaled
// by `effective_scale = global_scale * fp8_e4m3_to_fp32(micro_scale)`.
//
// How it works in hardware registers:
//   - E2M1 magnitude codes 0..7 map 1-to-1 to exact FP8 E4M3 bytes:
//       lut_lo = 0x3C383000 (0.0, 0.5, 1.0, 1.5)
//       lut_hi = 0x4C484440 (2.0, 3.0, 4.0, 6.0)
//   - `lop3.b32` isolates 3-bit magnitude selectors for even/odd nibbles and
//     merges sign bits (bit 3 of nibble -> bit 7 of FP8 byte) with zero branches!
//   - `prmt.b32` performs 4 simultaneous byte lookups per instruction in 1 cycle!
// ==============================================================================
TURBOKV_HOST_DEVICE void unpack_nvfp4x8_ptx_scaled(
    uint32_t packed_word,
    float effective_scale,
    float out[8]) noexcept
{
    constexpr uint32_t kLutLo = 0x3C383000u; // E4M3: {0.0, 0.5, 1.0, 1.5}
    constexpr uint32_t kLutHi = 0x4C484440u; // E4M3: {2.0, 3.0, 4.0, 6.0}

    // Even nibbles: elements 0, 2, 4, 6 sit at bits [3:0], [11:8], [19:16], [27:24]
    // Pack their 3-bit magnitudes into prmt selector nibbles [3:0], [7:4], [11:8], [15:12]
    uint32_t even_mags =
        ((packed_word >> 0)  & 0x0007u) |
        ((packed_word >> 4)  & 0x0070u) |
        ((packed_word >> 8)  & 0x0700u) |
        ((packed_word >> 12) & 0x7000u);
    uint32_t even_signs =
        ((packed_word << 4) & 0x00000080u) |
        ((packed_word << 4) & 0x00008000u) |
        ((packed_word << 4) & 0x00800000u) |
        ((packed_word << 4) & 0x80000000u);

    uint32_t even_fp8x4 = hal::ptx_prmt_b32(kLutLo, kLutHi, even_mags);
    even_fp8x4 = hal::ptx_lop3_b32<0xEA>(even_fp8x4, 0x7F7F7F7Fu, even_signs);

    // Odd nibbles: elements 1, 3, 5, 7 sit at bits [7:4], [15:12], [23:20], [31:28]
    uint32_t odd_shifted = packed_word >> 4;
    uint32_t odd_mags =
        ((odd_shifted >> 0)  & 0x0007u) |
        ((odd_shifted >> 4)  & 0x0070u) |
        ((odd_shifted >> 8)  & 0x0700u) |
        ((odd_shifted >> 12) & 0x7000u);
    uint32_t odd_signs =
        (packed_word & 0x00000080u) |
        (packed_word & 0x00008000u) |
        (packed_word & 0x00800000u) |
        (packed_word & 0x80000000u);

    uint32_t odd_fp8x4 = hal::ptx_prmt_b32(kLutLo, kLutHi, odd_mags);
    odd_fp8x4 = hal::ptx_lop3_b32<0xEA>(odd_fp8x4, 0x7F7F7F7Fu, odd_signs);

    out[0] = fp8_e4m3_to_fp32(static_cast<uint8_t>((even_fp8x4 >> 0)  & 0xFFu)) * effective_scale;
    out[2] = fp8_e4m3_to_fp32(static_cast<uint8_t>((even_fp8x4 >> 8)  & 0xFFu)) * effective_scale;
    out[4] = fp8_e4m3_to_fp32(static_cast<uint8_t>((even_fp8x4 >> 16) & 0xFFu)) * effective_scale;
    out[6] = fp8_e4m3_to_fp32(static_cast<uint8_t>((even_fp8x4 >> 24) & 0xFFu)) * effective_scale;

    out[1] = fp8_e4m3_to_fp32(static_cast<uint8_t>((odd_fp8x4 >> 0)  & 0xFFu)) * effective_scale;
    out[3] = fp8_e4m3_to_fp32(static_cast<uint8_t>((odd_fp8x4 >> 8)  & 0xFFu)) * effective_scale;
    out[5] = fp8_e4m3_to_fp32(static_cast<uint8_t>((odd_fp8x4 >> 16) & 0xFFu)) * effective_scale;
    out[7] = fp8_e4m3_to_fp32(static_cast<uint8_t>((odd_fp8x4 >> 24) & 0xFFu)) * effective_scale;
}

// Quantize a 16-element micro-block using nested Two-Level Microscaling (FR-1.1):
//   1) Computes local absmax a_g = max_{k=0..15} |x_k|
//   2) Computes FP8 E4M3 micro-scale s_hat = FP8_E4M3((a_g / 6.0f) / global_scale)
//   3) Reconstructs effective scale s_tilde = global_scale * FP32(s_hat)
//   4) Packs 16 elements into two 32-bit NVFP4 words (lo8, hi8)
struct MicroBlockNVFP4 {
    uint32_t lo8;
    uint32_t hi8;
    uint8_t  micro_scale_e4m3;
};

TURBOKV_HOST_DEVICE MicroBlockNVFP4 quantize_microblock_16(
    const float x[16],
    float global_scale) noexcept
{
    float safe_global = (global_scale > 1e-30f) ? global_scale : 1e-30f;
    float local_max = 0.0f;
    for (int i = 0; i < 16; ++i) {
        local_max = std::max(local_max, std::fabs(x[i]));
    }

    // MSE-optimal NVFP4 E2M1 micro-scale search in registers (4 candidate clipping ratios):
    // Minimizes L2 reconstruction error sum_i (x_i - dequant(quant(x_i)))^2 while storing a
    // single standard FP8 E4M3 micro-scale per 16 elements (zero decode overhead!).
    static constexpr float kClipGrid[5] = {1.00f, 0.90f, 0.82f, 0.74f, 0.66f};
    uint8_t best_s_e4m3 = 0u;
    float   best_mse    = 1e38f;

    if (local_max > 0.0f) {
        for (int c = 0; c < 5; ++c) {
            float cand_scale = (local_max * kClipGrid[c]) / NVFP4_E2M1_MAX;
            uint8_t s_code = fp32_to_fp8_e4m3_rne(cand_scale / safe_global);
            if (s_code == 0u) s_code = 0x01u;
            float eff = safe_global * fp8_e4m3_to_fp32(s_code);
            float inv = 1.0f / eff;
            float mse = 0.0f;
            for (int i = 0; i < 16; ++i) {
                uint8_t q = fp32_to_nvfp4_e2m1_rne(x[i] * inv);
                float diff = x[i] - nvfp4_e2m1_to_fp32(q) * eff;
                mse += diff * diff;
            }
            if (mse < best_mse) {
                best_mse    = mse;
                best_s_e4m3 = s_code;
            }
        }
    }

    float eff_scale = safe_global * fp8_e4m3_to_fp32(best_s_e4m3);
    float inv_scale = (eff_scale > 1e-30f) ? (1.0f / eff_scale) : 0.0f;

    MicroBlockNVFP4 out{};
    out.micro_scale_e4m3 = best_s_e4m3;
    out.lo8 = pack_nvfp4x8(x, inv_scale);
    out.hi8 = pack_nvfp4x8(x + 8, inv_scale);
    return out;
}

TURBOKV_HOST_DEVICE void dequantize_microblock_16(
    uint32_t lo8,
    uint32_t hi8,
    uint8_t micro_scale_e4m3,
    float global_scale,
    float out[16]) noexcept
{
    float eff_scale = global_scale * fp8_e4m3_to_fp32(micro_scale_e4m3);
    unpack_nvfp4x8_ptx_scaled(lo8, eff_scale, out);
    unpack_nvfp4x8_ptx_scaled(hi8, eff_scale, out + 8);
}

} // namespace turbokv
