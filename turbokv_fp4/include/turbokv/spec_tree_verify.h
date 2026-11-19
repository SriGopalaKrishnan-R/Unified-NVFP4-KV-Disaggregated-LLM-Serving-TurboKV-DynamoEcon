#pragma once

#include "turbokv/kv_cache_manager.h"
#include <cstdint>
#include <vector>

namespace turbokv {

// ==============================================================================
// FR-1.3: Parallel Speculative Tree Verification & Acceptance Kernel
//   - Draft tree depth D <= 8, width W <= 64
//   - Encoded via a 64-bit bitmask ancestor table (`ancestor_mask[W]`):
//       bit j of ancestor_mask[i] is 1 iff draft node j is an ancestor of node i (or j == i).
//   - Single-pass target verification using bitwise tree-masking in FlashAttention
//   - Parallel warp-level rejection-sampling scan (`__ballot_sync` + `__ffs`)
//   - Direct on-device commit of accepted KV blocks into paged NVFP4/Window cache
//     with ZERO cudaDeviceSynchronize() calls on the critical path (NFR-1.4).
// ==============================================================================

constexpr int kSpecMaxDepth = 8;
constexpr int kSpecMaxWidth = 64;

struct SpeculativeTreeSpec {
    int width = 0;                               // W <= 64 (node 0 is root draft token)
    int max_depth = 0;                           // D <= 8
    uint64_t ancestor_mask[kSpecMaxWidth]{};     // bit j set in ancestor_mask[i] iff j <=_tree i
    int parent_idx[kSpecMaxWidth]{};             // -1 for root (node 0), else parent in [0, i)
    int depth[kSpecMaxWidth]{};                  // 1..D
    int token_ids[kSpecMaxWidth]{};              // drafted token IDs
};

struct SpeculativeVerifyResult {
    int num_accepted             = 0;            // Number of accepted draft nodes along winning branch
    int accepted_node_indices[kSpecMaxDepth]{};  // Node indices in [0, W) committed to KV cache
    int accepted_token_ids[kSpecMaxDepth]{};     // Accepted draft token IDs
    int bonus_token_id           = 0;            // Resampled/greedy target token after last accepted node
    uint32_t warp_ballot_mask    = 0;            // Diagnostic warp rejection ballot bitmask
    int first_reject_ffs         = 0;            // Diagnostic 1-indexed __ffs result
    uint64_t host_sync_calls_delta = 0;          // NFR-1.4 verification: MUST equal 0
};

// Helper to build a balanced or realistic multi-branch speculative draft tree (D <= 8, W <= 64)
TURBOKV_API SpeculativeTreeSpec build_canonical_draft_tree(int depth, int branching_factor);

// Executes single-pass bitwise-masked tree attention over [cached_prefix || W draft nodes]
// + warp-level rejection sampling (`__ballot_sync` + `__ffs`) + direct paged KV commit!
TURBOKV_API SpeculativeVerifyResult spec_tree_verify_and_commit(
    HybridPagedKVCache& cache,
    int seq_id,
    const SpeculativeTreeSpec& tree,
    const float* q_draft,         // [W, num_q_heads, head_dim]
    const float* k_draft,         // [W, num_kv_heads, head_dim]
    const float* v_draft,         // [W, num_kv_heads, head_dim]
    float* out_draft_attn,        // [W, num_q_heads, head_dim]
    const float* p_target_nodes,  // [W] target model probability assigned to each drafted node
    const float* p_draft_nodes,   // [W] draft model probability assigned to each drafted node
    const float* uniform_rands,   // [W] deterministic uniform U(0,1) variates for rejection test
    int bonus_token_candidate,
    int num_q_heads);

} // namespace turbokv
