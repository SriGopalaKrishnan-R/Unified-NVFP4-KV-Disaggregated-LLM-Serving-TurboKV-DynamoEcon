// ==============================================================================
// NV-DEVTECH-SRS-2026-V3 | FR-1.2 Path 2 — Chunked Prefill / Extend
// (kv_extend_workspace.cu)
//
// 1. Gathers cached NVFP4 historical prefix blocks via non-contiguous `block_table`,
//    applies PTX register unpacking + inverse Walsh-Hadamard, and writes into a
//    SINGLE LAYER-SHARED FP8 E4M3 WORKSPACE BUFFER (avoiding O(L * num_layers) bloat).
// 2. Concatenates Sink/Window prefix tokens + active chunk's FP8 K/V tokens into the
//    same shared workspace.
// 3. Executes fused causal FlashAttention over [0 .. prefix_len + chunk_len).
// ==============================================================================

#include "turbokv/cuda_compat.h"
#include "turbokv/nvfp4_types.h"
#include "turbokv/walsh_hadamard.h"
#include <cfloat>
#include <cmath>

#if defined(__CUDACC__)

namespace turbokv::cuda_kernels {

// Phase A: Gather & dequantize NVFP4 paged prefix blocks -> single layer-shared FP8 workspace
template <int HeadDim>
__global__ __launch_bounds__(128, 4)
void gather_nvfp4_prefix_to_fp8_workspace_kernel(
    const uint4* __restrict__ k_nvfp4_pool,
    const uint4* __restrict__ v_nvfp4_pool,
    const uint8_t* __restrict__ k_micro_scales,
    const uint8_t* __restrict__ v_micro_scales,
    const int32_t* __restrict__ block_table,
    uint8_t* __restrict__ ws_k_fp8,           // Layer-shared FP8 workspace [total_len, num_kv_heads, HeadDim]
    uint8_t* __restrict__ ws_v_fp8,
    float* __restrict__ ws_k_scale,           // [total_len, num_kv_heads]
    float* __restrict__ ws_v_scale,
    float k_global_scale,
    float v_global_scale,
    int hist_len,
    int n_sink,
    int page_size,
    int num_kv_heads)
{
    constexpr int kElemsPerLane = HeadDim / 32;
    const int warp_id  = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    const int hist_idx = blockIdx.x * 4 + warp_id;
    const int kv_head  = blockIdx.y;

    if (hist_idx >= hist_len) return;

    const int logical_blk = hist_idx / page_size;
    const int blk_offset  = hist_idx % page_size;
    const int32_t phys_slot = block_table[logical_blk] * page_size + blk_offset;
    const size_t head_slot  = static_cast<size_t>(phys_slot) * num_kv_heads + kv_head;

    // Each lane unpacks its slice from the vectorized uint4 load
    const int u4_idx = (lane * kElemsPerLane) / 32;
    const uint4 ku4  = __ldg(&k_nvfp4_pool[head_slot * (HeadDim / 32) + u4_idx]);
    const uint4 vu4  = __ldg(&v_nvfp4_pool[head_slot * (HeadDim / 32) + u4_idx]);

    const int micro_idx = (lane * kElemsPerLane) / 16;
    const float k_eff = k_global_scale * fp8_e4m3_to_fp32(k_micro_scales[head_slot * (HeadDim / 16) + micro_idx]);
    const float v_eff = v_global_scale * fp8_e4m3_to_fp32(v_micro_scales[head_slot * (HeadDim / 16) + micro_idx]);

    const int word_in_u4 = ((lane * kElemsPerLane) % 32) / 8;
    const uint32_t kw = (word_in_u4 == 0) ? ku4.x : ((word_in_u4 == 1) ? ku4.y : ((word_in_u4 == 2) ? ku4.z : ku4.w));
    const uint32_t vw = (word_in_u4 == 0) ? vu4.x : ((word_in_u4 == 1) ? vu4.y : ((word_in_u4 == 2) ? vu4.z : vu4.w));

    float k_unpacked8[8], v_unpacked8[8];
    unpack_nvfp4x8_ptx_scaled(kw, k_eff, k_unpacked8);
    unpack_nvfp4x8_ptx_scaled(vw, v_eff, v_unpacked8);

    const int sub_off = (lane * kElemsPerLane) % 8;
    float k_reg[kElemsPerLane], v_reg[kElemsPerLane];
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        k_reg[e] = k_unpacked8[sub_off + e];
        v_reg[e] = v_unpacked8[sub_off + e];
    }

    // Inverse Walsh-Hadamard transform (involutory: H_tilde * H_tilde = I)
    fwht_warp_registers<HeadDim>(k_reg);
    fwht_warp_registers<HeadDim>(v_reg);

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

    const float ks = (k_amax > 1e-12f) ? (k_amax / FP8_E4M3_MAX) : 1.0f;
    const float vs = (v_amax > 1e-12f) ? (v_amax / FP8_E4M3_MAX) : 1.0f;

    const int dst_token = n_sink + hist_idx;
    const size_t ws_sc  = static_cast<size_t>(dst_token) * num_kv_heads + kv_head;
    if (lane == 0) {
        ws_k_scale[ws_sc] = ks;
        ws_v_scale[ws_sc] = vs;
    }
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        int idx = lane * kElemsPerLane + e;
        ws_k_fp8[ws_sc * HeadDim + idx] = fp32_to_fp8_e4m3_rne(k_reg[e] / ks);
        ws_v_fp8[ws_sc * HeadDim + idx] = fp32_to_fp8_e4m3_rne(v_reg[e] / vs);
    }
}

// Phase B: Fused Causal FlashAttention over the Unified Layer-Shared FP8 Workspace
template <int HeadDim>
__global__ __launch_bounds__(128, 4)
void chunked_extend_flash_attn_fp8_ws_kernel(
    const __nv_bfloat16* __restrict__ q_chunk_bf16, // [chunk_len, num_q_heads, HeadDim]
    const uint8_t* __restrict__ ws_k_fp8,           // [prefix_len + chunk_len, num_kv_heads, HeadDim]
    const uint8_t* __restrict__ ws_v_fp8,
    const float* __restrict__ ws_k_scale,
    const float* __restrict__ ws_v_scale,
    __nv_bfloat16* __restrict__ out_chunk_bf16,
    int prefix_len,
    int chunk_len,
    int num_q_heads,
    int num_kv_heads)
{
    constexpr int kElemsPerLane = HeadDim / 32;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int c_idx   = blockIdx.x * 4 + warp_id;
    const int q_head  = blockIdx.y;

    if (c_idx >= chunk_len) return;

    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const float softmax_scale = rsqrtf(static_cast<float>(HeadDim));

    float q_reg[kElemsPerLane];
    const size_t q_base = (static_cast<size_t>(c_idx) * num_q_heads + q_head) * HeadDim;
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        q_reg[e] = __bfloat162float(q_chunk_bf16[q_base + lane * kElemsPerLane + e]) * softmax_scale;
    }

    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float acc[kElemsPerLane] = {0.0f};
    const int max_kv_pos = prefix_len + c_idx;

    for (int t = 0; t <= max_kv_pos; ++t) {
        const size_t sc_idx = static_cast<size_t>(t) * num_kv_heads + kv_head;
        const float ks = ws_k_scale[sc_idx];
        const float vs = ws_v_scale[sc_idx];
        const size_t kv_base = sc_idx * HeadDim + lane * kElemsPerLane;

        float dot = 0.0f;
        #pragma unroll
        for (int e = 0; e < kElemsPerLane; ++e) {
            float kt = fp8_e4m3_to_fp32(ws_k_fp8[kv_base + e]) * ks;
            dot += q_reg[e] * kt;
        }
        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1) {
            dot += __shfl_xor_sync(0xFFFFFFFFu, dot, mask);
        }

        float m_new = fmaxf(m_prev, dot);
        float alpha = __expf(m_prev - m_new);
        float beta  = __expf(dot - m_new);
        l_prev = l_prev * alpha + beta;

        #pragma unroll
        for (int e = 0; e < kElemsPerLane; ++e) {
            float vt = fp8_e4m3_to_fp32(ws_v_fp8[kv_base + e]) * vs;
            acc[e] = acc[e] * alpha + beta * vt;
        }
        m_prev = m_new;
    }

    const float inv_l = 1.0f / fmaxf(l_prev, 1e-20f);
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        out_chunk_bf16[q_base + lane * kElemsPerLane + e] = __float2bfloat16(acc[e] * inv_l);
    }
}

} // namespace turbokv::cuda_kernels
#endif
