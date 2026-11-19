#include "turbokv/paged_attention.h"
#include "turbokv/roofline_profiler.h"
#include "turbokv/spec_tree_verify.h"
#include <cstring>

extern "C" {

struct TurboKVFullSnapshot {
    // NFR-1.1 Memory Compression
    uint64_t bf16_bytes;
    uint64_t total_hybrid_bytes;
    uint64_t sink_fp8_bytes;
    uint64_t window_fp8_bytes;
    uint64_t nvfp4_payload_bytes;
    uint64_t micro_scale_fp8_bytes;
    double   compression_ratio;

    // NFR-1.2 Roofline Bandwidth
    double   peak_hbm_bw_gbps;
    double   achieved_hbm_bw_gbps;
    double   hbm_bw_utilization_pct;
    double   estimated_kernel_us;
    double   speedup_vs_bf16_decode;
    int      registers_per_thread;
    double   achieved_occupancy_pct;
    uint64_t shared_mem_bank_conflicts;

    // NFR-1.3 Numerical Fidelity
    double   turbokv_cosine_sim;
    double   turbokv_kl_div;
    double   turbokv_rmse;
    double   uniform_fp4_cosine_sim;
    double   uniform_fp4_kl_div;
    double   uniform_fp4_rmse;
    double   raw_channel_crest_factor;
    double   rotated_channel_crest_factor;

    // FR-1.3 & NFR-1.4 Speculative Tree & Zero Host Sync
    int      spec_tree_width;
    int      spec_tree_depth;
    int      spec_num_accepted;
    int      spec_bonus_token_id;
    uint32_t spec_warp_ballot_mask;
    int      spec_first_reject_ffs;
    uint64_t host_sync_calls;
};

TURBOKV_API int turbokv_evaluate_suite(
    int seq_len,
    int batch_size,
    int head_dim,
    int arch_code,
    int dist_code,
    int spec_depth,
    int spec_branch,
    float accept_rate_bias,
    TurboKVFullSnapshot* out_snap)
{
    if (!out_snap) return -1;
    std::memset(out_snap, 0, sizeof(TurboKVFullSnapshot));

    const int num_q_heads  = 32;
    const int num_kv_heads = 8;
    const int safe_dim     = (head_dim == 64) ? 64 : 128;

    turbokv::hal::reset_host_sync_counter();

    // 1. NFR-1.1 Memory Compression Report at requested seq_len
    turbokv::HybridKVConfig cfg{};
    cfg.num_kv_heads = num_kv_heads;
    cfg.head_dim     = safe_dim;
    cfg.max_blocks   = 8192;
    turbokv::HybridPagedKVCache footprint_cache(cfg);
    auto fp = footprint_cache.compute_footprint(static_cast<size_t>(seq_len));

    out_snap->bf16_bytes            = fp.bf16_bytes;
    out_snap->total_hybrid_bytes    = fp.total_hybrid_bytes;
    out_snap->sink_fp8_bytes        = fp.sink_fp8_bytes;
    out_snap->window_fp8_bytes      = fp.window_fp8_bytes;
    out_snap->nvfp4_payload_bytes   = fp.nvfp4_payload_bytes;
    out_snap->micro_scale_fp8_bytes = fp.micro_scale_fp8_bytes;
    out_snap->compression_ratio     = fp.compression_ratio;

    // 2. NFR-1.2 Roofline Analysis
    auto arch = turbokv::GPUArchTarget::BLACKWELL_SM100_B200;
    if (arch_code == 89) arch = turbokv::GPUArchTarget::ADA_SM89_RTX4090;
    else if (arch_code == 90) arch = turbokv::GPUArchTarget::HOPPER_SM90_H100_SXM5;

    auto roof = turbokv::analyze_path3_decode_roofline(arch, batch_size, seq_len, num_q_heads, num_kv_heads, safe_dim);
    out_snap->peak_hbm_bw_gbps          = roof.peak_hbm_bw_gbps;
    out_snap->achieved_hbm_bw_gbps      = roof.achieved_hbm_bw_gbps;
    out_snap->hbm_bw_utilization_pct    = roof.hbm_bw_utilization_pct;
    out_snap->estimated_kernel_us       = roof.estimated_kernel_us;
    out_snap->speedup_vs_bf16_decode    = roof.speedup_vs_bf16_decode;
    out_snap->registers_per_thread      = roof.registers_per_thread;
    out_snap->achieved_occupancy_pct    = roof.achieved_occupancy_pct;
    out_snap->shared_mem_bank_conflicts = roof.shared_mem_bank_conflicts;

    // 3. NFR-1.3 Numerical Fidelity (cap interactive fidelity tensor size at 4096 for <20ms UI response while preserving exact statistical regime)
    int fid_len = (seq_len > 4096) ? 4096 : (seq_len < 256 ? 256 : seq_len);
    auto dist = turbokv::EvalDistribution::NEEDLE_IN_A_HAYSTACK;
    if (dist_code == 1) dist = turbokv::EvalDistribution::LONG_REASONING_CHAIN;
    else if (dist_code == 2) dist = turbokv::EvalDistribution::EXTREME_CHANNEL_OUTLIERS;

    auto fid = turbokv::evaluate_nfr13_fidelity(fid_len, num_q_heads, num_kv_heads, safe_dim, dist, 2026u);
    out_snap->turbokv_cosine_sim           = fid.turbokv_cosine_sim;
    out_snap->turbokv_kl_div               = fid.turbokv_kl_div;
    out_snap->turbokv_rmse                 = fid.turbokv_rmse;
    out_snap->uniform_fp4_cosine_sim       = fid.uniform_fp4_cosine_sim;
    out_snap->uniform_fp4_kl_div           = fid.uniform_fp4_kl_div;
    out_snap->uniform_fp4_rmse             = fid.uniform_fp4_rmse;
    out_snap->raw_channel_crest_factor     = fid.raw_channel_crest_factor;
    out_snap->rotated_channel_crest_factor = fid.rotated_channel_crest_factor;

    // 4. FR-1.3 Speculative Tree Verification & NFR-1.4 Zero Host Sync
    auto tree = turbokv::build_canonical_draft_tree(spec_depth, spec_branch);
    int W = tree.width;
    std::vector<float> qd(static_cast<size_t>(W) * num_q_heads * safe_dim, 0.15f);
    std::vector<float> kd(static_cast<size_t>(W) * num_kv_heads * safe_dim, 0.12f);
    std::vector<float> vd(static_cast<size_t>(W) * num_kv_heads * safe_dim, 0.20f);
    std::vector<float> od(qd.size(), 0.0f);
    std::vector<float> p_tgt(W, 0.85f);
    std::vector<float> p_drf(W, 0.80f);
    std::vector<float> u_rnd(W, 0.50f);
    for (int i = 0; i < W; ++i) {
        // Modulate acceptance deterministically based on depth & accept_rate_bias
        float decay = 1.0f - 0.08f * static_cast<float>(tree.depth[i] - 1);
        p_tgt[i] = accept_rate_bias * decay;
        p_drf[i] = 0.72f;
        u_rnd[i] = 0.35f + 0.07f * static_cast<float>(i % 7);
    }

    // Seed small prefix into footprint_cache seq 0
    std::vector<float> seed_kv(static_cast<size_t>(num_kv_heads) * safe_dim, 0.1f);
    for (int t = 0; t < 32; ++t) {
        footprint_cache.append_token(0, seed_kv.data(), seed_kv.data());
    }
    auto spec_res = turbokv::spec_tree_verify_and_commit(
        footprint_cache, 0, tree,
        qd.data(), kd.data(), vd.data(), od.data(),
        p_tgt.data(), p_drf.data(), u_rnd.data(),
        9999, num_q_heads);

    out_snap->spec_tree_width       = tree.width;
    out_snap->spec_tree_depth       = tree.max_depth;
    out_snap->spec_num_accepted     = spec_res.num_accepted;
    out_snap->spec_bonus_token_id   = spec_res.bonus_token_id;
    out_snap->spec_warp_ballot_mask = spec_res.warp_ballot_mask;
    out_snap->spec_first_reject_ffs = spec_res.first_reject_ffs;
    out_snap->host_sync_calls       = spec_res.host_sync_calls_delta;
    return 0;
}

} // extern "C"
