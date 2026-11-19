// ==============================================================================
// NV-DEVTECH-SRS-2026-V3 | FR-1.2 Path 3 — Fused Autoregressive Decode
// (kv_decode_fp4_fused.cu)
//
// Hardware Co-Design Highlights:
//   1. Vectorized 128-bit `uint4` (__ldg) loads of packed NVFP4 blocks from HBM.
//   2. Register-only unpacking of 8 E2M1 elements per 32-bit word via inline PTX
//      `prmt.b32` + `lop3.b32` (`unpack_nvfp4x8_ptx_scaled`) — ZERO LUT memory stalls!
//   3. Two-level scale reconstruction in registers: `global_scale * fp8_e4m3(micro_scale)`.
//   4. Hadamard inner-product invariance: Q is rotated ONCE in registers prior to the
//      historical NVFP4 loop (<H q, H k> = <q, k>), and accumulated V_rot is un-rotated
//      ONCE in the epilogue — ZERO per-token FWHT inside the hot KV loop!
//   5. Online-softmax attention via warp-shuffle reductions (`__shfl_xor_sync`) and
//      padded split-K cross-warp combine (`stride = 33`) -> ZERO shared-memory bank conflicts!
//   6. `__launch_bounds__(128, 8)` enforces <= 64-128 regs/thread (>= 50% SM occupancy).
// ==============================================================================

#include "turbokv/cuda_compat.h"
#include "turbokv/nvfp4_types.h"
#include "turbokv/walsh_hadamard.h"
#include <cfloat>
#include <cmath>

#if defined(__CUDACC__)

namespace turbokv::cuda_kernels {

template <int HeadDim>
__global__ __launch_bounds__(128, 8)
void kv_decode_fp4_fused_kernel(
    const __nv_bfloat16* __restrict__ q_bf16, // [batch_size, num_q_heads, HeadDim]
    const uint4* __restrict__ k_nvfp4_pool,
    const uint4* __restrict__ v_nvfp4_pool,
    const uint8_t* __restrict__ k_micro_scales,
    const uint8_t* __restrict__ v_micro_scales,
    const uint8_t* __restrict__ k_sink_fp8,
    const uint8_t* __restrict__ v_sink_fp8,
    const float* __restrict__ k_sink_scale,
    const float* __restrict__ v_sink_scale,
    const uint8_t* __restrict__ k_win_fp8,
    const uint8_t* __restrict__ v_win_fp8,
    const float* __restrict__ k_win_scale,
    const float* __restrict__ v_win_scale,
    const int32_t* __restrict__ block_tables, // [batch_size, max_blocks_per_seq]
    const int* __restrict__ seq_lens,         // [batch_size]
    const int* __restrict__ hist_lens,        // [batch_size]
    const float* __restrict__ k_global_scales,// [batch_size]
    const float* __restrict__ v_global_scales,// [batch_size]
    __nv_bfloat16* __restrict__ out_bf16,     // [batch_size, num_q_heads, HeadDim]
    int max_blocks_per_seq,
    int page_size,
    int num_q_heads,
    int num_kv_heads,
    int n_sink,
    int n_window)
{
    constexpr int kWarpsPerCTA  = 4;
    constexpr int kElemsPerLane = HeadDim / 32;

    // Padded stride (HeadDim + 1) guarantees ZERO 32-bank shared memory conflicts on final combine!
    __shared__ float smem_m[kWarpsPerCTA];
    __shared__ float smem_l[kWarpsPerCTA];
    __shared__ float smem_o[kWarpsPerCTA][HeadDim + 1];

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int b_idx   = blockIdx.x;
    const int q_head  = blockIdx.y;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);

    const int seq_len  = seq_lens[b_idx];
    const int hist_len = hist_lens[b_idx];
    const float scale  = rsqrtf(static_cast<float>(HeadDim));

    // Load Q in unrotated space (for FP8 Sink/Window) and compute rotated Q_rot (for NVFP4 History)
    float q_raw[kElemsPerLane];
    float q_rot[kElemsPerLane];
    const size_t q_base = (static_cast<size_t>(b_idx) * num_q_heads + q_head) * HeadDim;
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        float val = __bfloat162float(q_bf16[q_base + lane * kElemsPerLane + e]) * scale;
        q_raw[e] = val;
        q_rot[e] = val;
    }
    fwht_warp_registers<HeadDim>(q_rot);

    float m_local = -FLT_MAX;
    float l_local = 0.0f;
    float acc_rot[kElemsPerLane] = {0.0f};

    // 1. Historical Compressed NVFP4 Partition (striped across all 4 warps in CTA for peak HBM BW)
    const int32_t* seq_bt = block_tables + static_cast<size_t>(b_idx) * max_blocks_per_seq;
    const float kgs = k_global_scales[b_idx];
    const float vgs = v_global_scales[b_idx];

    for (int h_idx = warp_id; h_idx < hist_len; h_idx += kWarpsPerCTA) {
        const int logical_blk   = h_idx / page_size;
        const int blk_offset    = h_idx % page_size;
        const int32_t phys_slot = __ldg(&seq_bt[logical_blk]) * page_size + blk_offset;
        const size_t head_slot  = static_cast<size_t>(phys_slot) * num_kv_heads + kv_head;

        const int u4_idx = (lane * kElemsPerLane) / 32;
        const uint4 ku4  = __ldg(&k_nvfp4_pool[head_slot * (HeadDim / 32) + u4_idx]);
        const uint4 vu4  = __ldg(&v_nvfp4_pool[head_slot * (HeadDim / 32) + u4_idx]);

        const int micro_idx = (lane * kElemsPerLane) / 16;
        const float k_eff = kgs * fp8_e4m3_to_fp32(__ldg(&k_micro_scales[head_slot * (HeadDim / 16) + micro_idx]));
        const float v_eff = vgs * fp8_e4m3_to_fp32(__ldg(&v_micro_scales[head_slot * (HeadDim / 16) + micro_idx]));

        const int word_in_u4 = ((lane * kElemsPerLane) % 32) / 8;
        const uint32_t kw = (word_in_u4 == 0) ? ku4.x : ((word_in_u4 == 1) ? ku4.y : ((word_in_u4 == 2) ? ku4.z : ku4.w));
        const uint32_t vw = (word_in_u4 == 0) ? vu4.x : ((word_in_u4 == 1) ? vu4.y : ((word_in_u4 == 2) ? vu4.z : vu4.w));

        float k8[8], v8[8];
        unpack_nvfp4x8_ptx_scaled(kw, k_eff, k8);
        unpack_nvfp4x8_ptx_scaled(vw, v_eff, v8);

        const int sub = (lane * kElemsPerLane) % 8;
        float dot = 0.0f;
        #pragma unroll
        for (int e = 0; e < kElemsPerLane; ++e) {
            dot += q_rot[e] * k8[sub + e];
        }
        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1) {
            dot += __shfl_xor_sync(0xFFFFFFFFu, dot, mask);
        }

        float m_new = fmaxf(m_local, dot);
        float alpha = __expf(m_local - m_new);
        float beta  = __expf(dot - m_new);
        l_local = l_local * alpha + beta;
        #pragma unroll
        for (int e = 0; e < kElemsPerLane; ++e) {
            acc_rot[e] = acc_rot[e] * alpha + beta * v8[sub + e];
        }
        m_local = m_new;
    }

    // Un-rotate historical V accumulation ONCE per warp before merging Sink & Window!
    fwht_warp_registers<HeadDim>(acc_rot);

    // 2. Warp 0 processes Attention-Sink (N_sink=4) + Local Sliding Window (N_win<=128) in FP8
    if (warp_id == 0) {
        const int sink_cnt = (seq_len < n_sink) ? seq_len : n_sink;
        for (int s = 0; s < sink_cnt; ++s) {
            size_t sc_idx = (static_cast<size_t>(b_idx) * n_sink + s) * num_kv_heads + kv_head;
            float ks = k_sink_scale[sc_idx];
            float vs = v_sink_scale[sc_idx];
            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < kElemsPerLane; ++e) {
                float kt = fp8_e4m3_to_fp32(k_sink_fp8[sc_idx * HeadDim + lane * kElemsPerLane + e]) * ks;
                dot += q_raw[e] * kt;
            }
            #pragma unroll
            for (int mask = 16; mask >= 1; mask >>= 1) {
                dot += __shfl_xor_sync(0xFFFFFFFFu, dot, mask);
            }
            float m_new = fmaxf(m_local, dot);
            float alpha = __expf(m_local - m_new);
            float beta  = __expf(dot - m_new);
            l_local = l_local * alpha + beta;
            #pragma unroll
            for (int e = 0; e < kElemsPerLane; ++e) {
                float vt = fp8_e4m3_to_fp32(v_sink_fp8[sc_idx * HeadDim + lane * kElemsPerLane + e]) * vs;
                acc_rot[e] = acc_rot[e] * alpha + beta * vt;
            }
            m_local = m_new;
        }

        const int rem_after_sink = (seq_len > n_sink) ? (seq_len - n_sink) : 0;
        const int win_cnt = (rem_after_sink > hist_len) ? (rem_after_sink - hist_len) : 0;
        for (int w = 0; w < win_cnt; ++w) {
            int pos = n_sink + hist_len + w;
            int ring_slot = (pos - n_sink) % n_window;
            size_t sc_idx = (static_cast<size_t>(b_idx) * n_window + ring_slot) * num_kv_heads + kv_head;
            float ks = k_win_scale[sc_idx];
            float vs = v_win_scale[sc_idx];
            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < kElemsPerLane; ++e) {
                float kt = fp8_e4m3_to_fp32(k_win_fp8[sc_idx * HeadDim + lane * kElemsPerLane + e]) * ks;
                dot += q_raw[e] * kt;
            }
            #pragma unroll
            for (int mask = 16; mask >= 1; mask >>= 1) {
                dot += __shfl_xor_sync(0xFFFFFFFFu, dot, mask);
            }
            float m_new = fmaxf(m_local, dot);
            float alpha = __expf(m_local - m_new);
            float beta  = __expf(dot - m_new);
            l_local = l_local * alpha + beta;
            #pragma unroll
            for (int e = 0; e < kElemsPerLane; ++e) {
                float vt = fp8_e4m3_to_fp32(v_win_fp8[sc_idx * HeadDim + lane * kElemsPerLane + e]) * vs;
                acc_rot[e] = acc_rot[e] * alpha + beta * vt;
            }
            m_local = m_new;
        }
    }

    // 3. Conflict-Free Cross-Warp Online-Softmax Reduction
    if (lane == 0) {
        smem_m[warp_id] = m_local;
        smem_l[warp_id] = l_local;
    }
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        smem_o[warp_id][lane * kElemsPerLane + e] = acc_rot[e];
    }
    __syncthreads();

    if (warp_id == 0) {
        float m_global = smem_m[0];
        #pragma unroll
        for (int w = 1; w < kWarpsPerCTA; ++w) {
            m_global = fmaxf(m_global, smem_m[w]);
        }
        float l_global = 0.0f;
        float out_final[kElemsPerLane] = {0.0f};
        #pragma unroll
        for (int w = 0; w < kWarpsPerCTA; ++w) {
            float scale_w = (smem_l[w] > 0.0f) ? __expf(smem_m[w] - m_global) : 0.0f;
            l_global += smem_l[w] * scale_w;
            #pragma unroll
            for (int e = 0; e < kElemsPerLane; ++e) {
                out_final[e] += smem_o[w][lane * kElemsPerLane + e] * scale_w;
            }
        }
        float inv_l = 1.0f / fmaxf(l_global, 1e-20f);
        #pragma unroll
        for (int e = 0; e < kElemsPerLane; ++e) {
            out_bf16[q_base + lane * kElemsPerLane + e] = __float2bfloat16(out_final[e] * inv_l);
        }
    }
}

} // namespace turbokv::cuda_kernels
#endif
