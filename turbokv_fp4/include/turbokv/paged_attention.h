#pragma once

#include "turbokv/kv_cache_manager.h"
#include <cstdint>
#include <vector>

namespace turbokv {

// ==============================================================================
// FR-1.2: Three-Path Paged Attention Engine Interface (CUDA + CPU-Sim)
// Supports Grouped-Query Attention (GQA: num_q_heads >= num_kv_heads)
// ==============================================================================

struct AttentionMetrics {
    double cosine_similarity     = 1.0;
    double kl_divergence         = 0.0;
    double max_abs_error         = 0.0;
    double latency_us            = 0.0;
    uint64_t host_sync_calls     = 0;
};

struct BenchmarkFidelityComparison {
    // TurboKV-FP4 (Hadamard + Two-Level NVFP4 + Sink(4) + Window(128))
    double turbokv_cosine_sim    = 0.0;
    double turbokv_kl_div        = 0.0;
    double turbokv_rmse          = 0.0;
    // Baseline Uniform 4-bit (no Hadamard, no Sink/Window, single tensor scale)
    double uniform_fp4_cosine_sim = 0.0;
    double uniform_fp4_kl_div     = 0.0;
    double uniform_fp4_rmse       = 0.0;
    // Outlier suppression telemetry
    double raw_channel_crest_factor     = 0.0;
    double rotated_channel_crest_factor = 0.0;
};

// Reference Full-Precision BF16 Causal Attention (Ground Truth for NFR-1.3)
TURBOKV_API void reference_bf16_causal_attention(
    const float* q,          // [num_tokens, num_q_heads, head_dim]
    const float* k,          // [total_kv_tokens, num_kv_heads, head_dim]
    const float* v,          // [total_kv_tokens, num_kv_heads, head_dim]
    float* out,              // [num_tokens, num_q_heads, head_dim]
    float* attn_probs_out,   // optional: [num_tokens, num_q_heads, total_kv_tokens] for exact KL-div
    int num_q_tokens,
    int total_kv_tokens,
    int q_start_pos,
    int num_q_heads,
    int num_kv_heads,
    int head_dim);

// Path 1 — Initial Prefill (FR-1.2: kv_prefill_write_fp4):
// Computes causal attention directly over BF16-rounded QKV projections while
// quantizing & packing K/V tensors into NVFP4 paged blocks + FP8 sink/window in the epilogue
// WITHOUT reading back from the KV cache.
TURBOKV_API void kv_prefill_write_fp4(
    HybridPagedKVCache& cache,
    int seq_id,
    const float* q,          // [seq_len, num_q_heads, head_dim]
    const float* k,          // [seq_len, num_kv_heads, head_dim]
    const float* v,          // [seq_len, num_kv_heads, head_dim]
    const int32_t* slot_mapping, // optional explicit slot mapping [seq_len], nullptr -> auto paged
    float* out,              // [seq_len, num_q_heads, head_dim]
    int seq_len,
    int num_q_heads);

// Path 2 — Chunked Prefill / Extend (FR-1.2: kv_extend_workspace):
// Gathers cached NVFP4 prefix blocks + sink/window into a SINGLE layer-shared FP8 workspace
// buffer, concatenates the active chunk's FP8 K/V tokens, updates the paged KV cache,
// and executes fused causal FlashAttention.
TURBOKV_API void kv_extend_workspace(
    HybridPagedKVCache& cache,
    SharedFP8Workspace& shared_workspace,
    int seq_id,
    const float* q_chunk,    // [chunk_len, num_q_heads, head_dim]
    const float* k_chunk,    // [chunk_len, num_kv_heads, head_dim]
    const float* v_chunk,    // [chunk_len, num_kv_heads, head_dim]
    float* out_chunk,        // [chunk_len, num_q_heads, head_dim]
    int chunk_len,
    int num_q_heads);

// Path 3 — Fused Autoregressive Decode (FR-1.2: kv_decode_fp4_fused):
// Vectorized uint4 loads, PTX prmt/lop3.b32 8-element register unpacking, two-level scale
// reconstruction, online-softmax attention via warp-shuffle reductions, zero bank conflicts,
// and single epilogue FWHT per head. Supports batched decode across `batch_size` sequences.
TURBOKV_API void kv_decode_fp4_fused(
    const HybridPagedKVCache& cache,
    const int* seq_ids,      // [batch_size]
    const float* q_decode,   // [batch_size, num_q_heads, head_dim]
    float* out_decode,       // [batch_size, num_q_heads, head_dim]
    float* attn_probs_last,  // optional: [num_q_heads, seq_len] for seq 0 KL-div inspection
    int batch_size,
    int num_q_heads);

// Synthetic Workload & NFR-1.3 Evaluation Suite (Needle-in-a-Haystack & Reasoning-Chain)
enum class EvalDistribution : int {
    NEEDLE_IN_A_HAYSTACK = 0,
    LONG_REASONING_CHAIN = 1,
    EXTREME_CHANNEL_OUTLIERS = 2
};

TURBOKV_API BenchmarkFidelityComparison evaluate_nfr13_fidelity(
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    EvalDistribution dist_type,
    uint32_t seed = 42u);

} // namespace turbokv
