// ==============================================================================
// NV-DEVTECH-SRS-2026-V3 | FR-1.2 Path 1 — Initial Prefill (kv_prefill_write_fp4.cu)
// Computes causal FlashAttention directly over BF16/FP8 QKV projections while
// asynchronously quantizing and packing K/V tensors into NVFP4 paged blocks
// (and FP8 Sink/Window partitions) in the epilogue WITHOUT reading back from KV cache.
// Target Architectures: SM89 (Ada), SM90 (Hopper), SM100 (Blackwell)
// Register Ceiling: <= 128 registers/thread (--maxrregcount=128, >= 50% occupancy)
// ==============================================================================

#include "turbokv/cuda_compat.h"
#include "turbokv/nvfp4_types.h"
#include "turbokv/walsh_hadamard.h"
#include <cfloat>
#include <cmath>

#if defined(__CUDACC__)

namespace turbokv::cuda_kernels {

template <int HeadDim>
__global__ __launch_bounds__(128, 4)
void kv_prefill_write_fp4_kernel(
    const __nv_bfloat16* __restrict__ q_bf16, // [seq_len, num_q_heads, HeadDim]
    const __nv_bfloat16* __restrict__ k_bf16, // [seq_len, num_kv_heads, HeadDim]
    const __nv_bfloat16* __restrict__ v_bf16, // [seq_len, num_kv_heads, HeadDim]
    __nv_bfloat16* __restrict__ out_bf16,     // [seq_len, num_q_heads, HeadDim]
    uint4* __restrict__ k_nvfp4_pool,         // [max_slots * num_kv_heads * (HeadDim/32)]
    uint4* __restrict__ v_nvfp4_pool,
    uint8_t* __restrict__ k_micro_scales,     // [max_slots * num_kv_heads * (HeadDim/16)]
    uint8_t* __restrict__ v_micro_scales,
    uint8_t* __restrict__ k_sink_fp8,
    uint8_t* __restrict__ v_sink_fp8,
    float* __restrict__ k_sink_scale,
    float* __restrict__ v_sink_scale,
    uint8_t* __restrict__ k_win_fp8,
    uint8_t* __restrict__ v_win_fp8,
    float* __restrict__ k_win_scale,
    float* __restrict__ v_win_scale,
    const int32_t* __restrict__ slot_mapping, // [seq_len] physical slot for historical tokens
    float k_global_scale,
    float v_global_scale,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int n_sink,
    int n_window)
{
    constexpr int kElemsPerLane = HeadDim / 32;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int q_token = blockIdx.x * 4 + warp_id; // 4 warps (128 threads) per CTA
    const int q_head  = blockIdx.y;

    if (q_token >= seq_len) return;

    const int gqa_group = num_q_heads / num_kv_heads;
    const int kv_head   = q_head / gqa_group;
    const float softmax_scale = rsqrtf(static_cast<float>(HeadDim));

    // 1. Load Q into registers
    float q_reg[kElemsPerLane];
    const size_t q_base = (static_cast<size_t>(q_token) * num_q_heads + q_head) * HeadDim;
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        q_reg[e] = __bfloat162float(q_bf16[q_base + lane * kElemsPerLane + e]) * softmax_scale;
    }

    // 2. Online-Softmax Causal Attention directly over incoming BF16 K/V projections
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float acc[kElemsPerLane] = {0.0f};

    for (int t = 0; t <= q_token; ++t) {
        const size_t kv_base = (static_cast<size_t>(t) * num_kv_heads + kv_head) * HeadDim;
        float dot_partial = 0.0f;
        #pragma unroll
        for (int e = 0; e < kElemsPerLane; ++e) {
            float kt = __bfloat162float(k_bf16[kv_base + lane * kElemsPerLane + e]);
            dot_partial += q_reg[e] * kt;
        }
        // Warp-level butterfly reduction (__shfl_xor_sync)
        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1) {
            dot_partial += __shfl_xor_sync(0xFFFFFFFFu, dot_partial, mask);
        }

        float m_new = fmaxf(m_prev, dot_partial);
        float alpha = __expf(m_prev - m_new);
        float beta  = __expf(dot_partial - m_new);
        l_prev = l_prev * alpha + beta;

        #pragma unroll
        for (int e = 0; e < kElemsPerLane; ++e) {
            float vt = __bfloat162float(v_bf16[kv_base + lane * kElemsPerLane + e]);
            acc[e] = acc[e] * alpha + beta * vt;
        }
        m_prev = m_new;
    }

    // Write BF16 attention output
    float inv_l = 1.0f / fmaxf(l_prev, 1e-20f);
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        out_bf16[q_base + lane * kElemsPerLane + e] = __float2bfloat16(acc[e] * inv_l);
    }

    // 3. Asynchronous Epilogue: First GQA query head per KV group packs K/V into Hybrid Cache!
    if ((q_head % gqa_group) != 0) return;

    const size_t kv_own_base = (static_cast<size_t>(q_token) * num_kv_heads + kv_head) * HeadDim;
    float k_reg[kElemsPerLane];
    float v_reg[kElemsPerLane];
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        k_reg[e] = __bfloat162float(k_bf16[kv_own_base + lane * kElemsPerLane + e]);
        v_reg[e] = __bfloat162float(v_bf16[kv_own_base + lane * kElemsPerLane + e]);
    }

    const int win_start = (seq_len > n_sink + n_window) ? (seq_len - n_window) : n_sink;

    if (q_token < n_sink || q_token >= win_start) {
        // Store in unquantized FP8 (E4M3) Sink or Sliding-Window partition
        float k_amax = 0.0f, v_amax = 0.0f;
        #pragma unroll
        for (int e = 0; e < kElemsPerLane; ++e) {
            k_amax = fmaxf(k_amax, fabsf(k_reg[e]));
            v_amax = fmaxf(v_amax, fabsf(v_reg[e]));
        }
        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1) {
            k_amax = fmaxf(k_amax, __shfl_xor_sync(0xFFFFFFFFu, k_amax, mask));
            v_amax = fmaxf(v_amax, __shfl_xor_sync(0xFFFFFFFFu, v_amax, mask));
        }
        float ks = (k_amax > 1e-12f) ? (k_amax / FP8_E4M3_MAX) : 1.0f;
        float vs = (v_amax > 1e-12f) ? (v_amax / FP8_E4M3_MAX) : 1.0f;

        if (q_token < n_sink) {
            size_t sc_idx = static_cast<size_t>(q_token) * num_kv_heads + kv_head;
            if (lane == 0) {
                k_sink_scale[sc_idx] = ks;
                v_sink_scale[sc_idx] = vs;
            }
            #pragma unroll
            for (int e = 0; e < kElemsPerLane; ++e) {
                int idx = lane * kElemsPerLane + e;
                k_sink_fp8[sc_idx * HeadDim + idx] = fp32_to_fp8_e4m3_rne(k_reg[e] / ks);
                v_sink_fp8[sc_idx * HeadDim + idx] = fp32_to_fp8_e4m3_rne(v_reg[e] / vs);
            }
        } else {
            int ring_slot = (q_token - n_sink) % n_window;
            size_t sc_idx = static_cast<size_t>(ring_slot) * num_kv_heads + kv_head;
            if (lane == 0) {
                k_win_scale[sc_idx] = ks;
                v_win_scale[sc_idx] = vs;
            }
            #pragma unroll
            for (int e = 0; e < kElemsPerLane; ++e) {
                int idx = lane * kElemsPerLane + e;
                k_win_fp8[sc_idx * HeadDim + idx] = fp32_to_fp8_e4m3_rne(k_reg[e] / ks);
                v_win_fp8[sc_idx * HeadDim + idx] = fp32_to_fp8_e4m3_rne(v_reg[e] / vs);
            }
        }
    } else {
        // Historical Compressed Block: In-register Walsh-Hadamard + Two-Level NVFP4 pack
        fwht_warp_registers<HeadDim>(k_reg);
        fwht_warp_registers<HeadDim>(v_reg);

        // Sub-warp of (16 / kElemsPerLane) lanes forms 1 micro-block of 16 elements
        constexpr int kLanesPerMicro = 16 / kElemsPerLane;
        float k_lmax = 0.0f, v_lmax = 0.0f;
        #pragma unroll
        for (int e = 0; e < kElemsPerLane; ++e) {
            k_lmax = fmaxf(k_lmax, fabsf(k_reg[e]));
            v_lmax = fmaxf(v_lmax, fabsf(v_reg[e]));
        }
        #pragma unroll
        for (int mask = kLanesPerMicro >> 1; mask >= 1; mask >>= 1) {
            k_lmax = fmaxf(k_lmax, __shfl_xor_sync(0xFFFFFFFFu, k_lmax, mask));
            v_lmax = fmaxf(v_lmax, __shfl_xor_sync(0xFFFFFFFFu, v_lmax, mask));
        }

        uint8_t k_ms = fp32_to_fp8_e4m3_rne((k_lmax / NVFP4_E2M1_MAX) / k_global_scale);
        uint8_t v_ms = fp32_to_fp8_e4m3_rne((v_lmax / NVFP4_E2M1_MAX) / v_global_scale);
        float k_eff = fmaxf(k_global_scale * fp8_e4m3_to_fp32(k_ms), 1e-30f);
        float v_eff = fmaxf(v_global_scale * fp8_e4m3_to_fp32(v_ms), 1e-30f);

        // Pack nibbles across lanes and write coalesced uint4
        uint32_t k_nibbles = 0u, v_nibbles = 0u;
        #pragma unroll
        for (int e = 0; e < kElemsPerLane; ++e) {
            k_nibbles |= (static_cast<uint32_t>(fp32_to_nvfp4_e2m1_rne(k_reg[e] / k_eff)) << (e * 4));
            v_nibbles |= (static_cast<uint32_t>(fp32_to_nvfp4_e2m1_rne(v_reg[e] / v_eff)) << (e * 4));
        }

        int32_t phys_slot = slot_mapping[q_token];
        size_t head_slot  = static_cast<size_t>(phys_slot) * num_kv_heads + kv_head;
        int micro_id      = lane / kLanesPerMicro;
        if ((lane % kLanesPerMicro) == 0) {
            k_micro_scales[head_slot * (HeadDim / 16) + micro_id] = k_ms;
            v_micro_scales[head_slot * (HeadDim / 16) + micro_id] = v_ms;
        }
        // Pack 32 elements (8 lanes when HeadDim=128) into 1 uint4
        uint32_t k_word = k_nibbles | (__shfl_down_sync(0xFFFFFFFFu, k_nibbles, 1) << (kElemsPerLane * 4));
        uint32_t v_word = v_nibbles | (__shfl_down_sync(0xFFFFFFFFu, v_nibbles, 1) << (kElemsPerLane * 4));
        if constexpr (kElemsPerLane == 2) {
            k_word |= (__shfl_down_sync(0xFFFFFFFFu, k_word, 2) << 16);
            v_word |= (__shfl_down_sync(0xFFFFFFFFu, v_word, 2) << 16);
        }
        int lanes_per_u32 = 8 / kElemsPerLane;
        uint32_t ku0 = __shfl_sync(0xFFFFFFFFu, k_word, (lane / (4 * lanes_per_u32)) * (4 * lanes_per_u32) + 0 * lanes_per_u32);
        uint32_t ku1 = __shfl_sync(0xFFFFFFFFu, k_word, (lane / (4 * lanes_per_u32)) * (4 * lanes_per_u32) + 1 * lanes_per_u32);
        uint32_t ku2 = __shfl_sync(0xFFFFFFFFu, k_word, (lane / (4 * lanes_per_u32)) * (4 * lanes_per_u32) + 2 * lanes_per_u32);
        uint32_t ku3 = __shfl_sync(0xFFFFFFFFu, k_word, (lane / (4 * lanes_per_u32)) * (4 * lanes_per_u32) + 3 * lanes_per_u32);

        uint32_t vu0 = __shfl_sync(0xFFFFFFFFu, v_word, (lane / (4 * lanes_per_u32)) * (4 * lanes_per_u32) + 0 * lanes_per_u32);
        uint32_t vu1 = __shfl_sync(0xFFFFFFFFu, v_word, (lane / (4 * lanes_per_u32)) * (4 * lanes_per_u32) + 1 * lanes_per_u32);
        uint32_t vu2 = __shfl_sync(0xFFFFFFFFu, v_word, (lane / (4 * lanes_per_u32)) * (4 * lanes_per_u32) + 2 * lanes_per_u32);
        uint32_t vu3 = __shfl_sync(0xFFFFFFFFu, v_word, (lane / (4 * lanes_per_u32)) * (4 * lanes_per_u32) + 3 * lanes_per_u32);

        int u4_idx = lane / (4 * lanes_per_u32);
        if ((lane % (4 * lanes_per_u32)) == 0) {
            k_nvfp4_pool[head_slot * (HeadDim / 32) + u4_idx] = make_uint4(ku0, ku1, ku2, ku3);
            v_nvfp4_pool[head_slot * (HeadDim / 32) + u4_idx] = make_uint4(vu0, vu1, vu2, vu3);
        }
    }
}

} // namespace turbokv::cuda_kernels
#endif
