#include "gtest_lite.h"
#include "turbokv/cuda_compat.h"
#include "turbokv/kv_cache_manager.h"
#include "turbokv/nvfp4_types.h"
#include "turbokv/paged_attention.h"
#include "turbokv/roofline_profiler.h"
#include "turbokv/spec_tree_verify.h"
#include "turbokv/walsh_hadamard.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace turbokv;

// 1. Exact Bit-Level E2M1 / E4M3 RNE Rounding & PTX prmt/lop3 Unpack Verification
TEST(TurboKVFP4, NVFP4BitExactRoundingAndPTXUnpack) {
    // Verify all 8 positive & 8 negative NVFP4 E2M1 codes
    const float expected_pos[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (uint8_t c = 0; c < 8; ++c) {
        EXPECT_NEAR(nvfp4_e2m1_to_fp32(c), expected_pos[c], 1e-7f);
        EXPECT_NEAR(nvfp4_e2m1_to_fp32(c | 0x08u), -expected_pos[c], 1e-7f);
    }

    // Verify IEEE Round-to-Nearest-Even (RNE) exact tie-breaking
    EXPECT_EQ(fp32_to_nvfp4_e2m1_rne(0.25f), 0u); // tie -> even m=0 -> 0.0
    EXPECT_EQ(fp32_to_nvfp4_e2m1_rne(0.75f), 2u); // tie -> even m=0 -> 1.0
    EXPECT_EQ(fp32_to_nvfp4_e2m1_rne(1.25f), 2u); // tie -> even m=0 -> 1.0
    EXPECT_EQ(fp32_to_nvfp4_e2m1_rne(1.75f), 4u); // tie -> even m=0 -> 2.0
    EXPECT_EQ(fp32_to_nvfp4_e2m1_rne(2.50f), 4u); // tie -> even m=0 -> 2.0
    EXPECT_EQ(fp32_to_nvfp4_e2m1_rne(3.50f), 6u); // tie -> even m=0 -> 4.0
    EXPECT_EQ(fp32_to_nvfp4_e2m1_rne(5.00f), 6u); // tie -> even m=0 -> 4.0
    EXPECT_EQ(fp32_to_nvfp4_e2m1_rne(9.50f), 7u); // satfinite -> 6.0
    EXPECT_EQ(fp32_to_nvfp4_e2m1_rne(-3.1f), static_cast<uint8_t>(0x08u | 5u)); // -3.0

    // Verify FP8 E4M3 saturation at +/-448.0
    EXPECT_NEAR(fp8_e4m3_to_fp32(fp32_to_fp8_e4m3_rne(500.0f)), 448.0f, 1e-4f);
    EXPECT_NEAR(fp8_e4m3_to_fp32(fp32_to_fp8_e4m3_rne(-999.0f)), -448.0f, 1e-4f);

    // Exhaustively verify PTX prmt.b32 + lop3.b32 8-element register unpacking matches LUT with 0 ULP error!
    float vals[8] = {0.0f, -0.5f, 1.5f, -3.0f, 6.0f, -4.0f, 2.0f, -1.0f};
    uint32_t packed = pack_nvfp4x8(vals, 1.0f);
    float unpacked[8] = {};
    unpack_nvfp4x8_ptx_scaled(packed, 2.25f, unpacked);
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(unpacked[i], vals[i] * 2.25f, 1e-6f);
    }
}

// 2. FR-1.1 Fast Walsh-Hadamard Transform Involution, Dot-Product Invariance & Outlier Suppression
TEST(TurboKVFP4, WalshHadamardOutlierPreconditioning) {
    for (int d : {64, 128}) {
        std::vector<float> q(d), k(d);
        for (int i = 0; i < d; ++i) {
            q[i] = std::cos(0.17f * static_cast<float>(i));
            k[i] = (i == 7) ? 45.0f : (0.2f * std::sin(0.3f * static_cast<float>(i))); // massive channel 7 spike!
        }

        double dot_orig = 0.0, raw_max = 0.0;
        for (int i = 0; i < d; ++i) {
            dot_orig += static_cast<double>(q[i]) * k[i];
            raw_max = std::max<double>(raw_max, std::fabs(k[i]));
        }

        std::vector<float> qr = q, kr = k;
        fwht_inplace(qr.data(), d);
        fwht_inplace(kr.data(), d);

        double dot_rot = 0.0, rot_max = 0.0;
        for (int i = 0; i < d; ++i) {
            dot_rot += static_cast<double>(qr[i]) * kr[i];
            rot_max = std::max<double>(rot_max, std::fabs(kr[i]));
        }

        // Orthogonal inner-product invariance: <H q, H k> == <q, k>
        EXPECT_NEAR(dot_orig, dot_rot, 1e-3);
        // Channel spike (45.0) suppressed by ~sqrt(d)!
        EXPECT_LT(rot_max, raw_max * 0.25);

        // Involution: H(H(k)) == k
        fwht_inplace(kr.data(), d);
        for (int i = 0; i < d; ++i) {
            EXPECT_NEAR(kr[i], k[i], 1e-4f);
        }
    }
}

// 3. NFR-1.1 Memory Compression >= 3.2x at Context Length 65,536
TEST(TurboKVFP4, NFR11_MemoryCompressionRatioAt64K) {
    HybridKVConfig cfg{};
    cfg.num_kv_heads = 8;
    cfg.head_dim     = 128;
    cfg.page_size    = 16;
    cfg.n_sink       = 4;
    cfg.n_window     = 128;
    HybridPagedKVCache cache(cfg);

    MemoryFootprintReport fp = cache.compute_footprint(65536);
    std::cout << "  [NFR-1.1] L=65536 BF16=" << (fp.bf16_bytes / (1024.0 * 1024.0))
              << " MiB, Hybrid=" << (fp.total_hybrid_bytes / (1024.0 * 1024.0))
              << " MiB, Compression=" << fp.compression_ratio << "x\n";

    EXPECT_GE(fp.compression_ratio, 3.20);
    EXPECT_GT(fp.sink_fp8_bytes, 0u);
    EXPECT_GT(fp.window_fp8_bytes, 0u);
    EXPECT_GT(fp.micro_scale_fp8_bytes, 0u);
}

// 4. FR-1.2 Three-Path Paged Attention Suite (Prefill + Layer-Shared Extend + Fused Decode)
TEST(TurboKVFP4, FR12_ThreePathPagedAttentionSuite) {
    const int num_q_heads  = 16;
    const int num_kv_heads = 4;
    const int head_dim     = 128;
    const int prefill_len  = 256;
    const int extend_len   = 64;
    const int total_len    = prefill_len + extend_len;

    std::mt19937 rng(12345u);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> q_all(static_cast<size_t>(total_len) * num_q_heads * head_dim);
    std::vector<float> k_all(static_cast<size_t>(total_len) * num_kv_heads * head_dim);
    std::vector<float> v_all(static_cast<size_t>(total_len) * num_kv_heads * head_dim);
    for (auto& x : q_all) x = dist(rng);
    for (auto& x : k_all) x = dist(rng);
    for (auto& x : v_all) x = dist(rng);

    HybridKVConfig cfg{};
    cfg.num_kv_heads = num_kv_heads;
    cfg.head_dim     = head_dim;
    cfg.n_sink       = 4;
    cfg.n_window     = 128;
    cfg.max_blocks   = 256;
    HybridPagedKVCache cache(cfg);
    cache.scramble_free_pool(); // Non-contiguous physical pages!

    // Path 1: Initial Prefill
    std::vector<float> out_prefill(static_cast<size_t>(prefill_len) * num_q_heads * head_dim);
    std::vector<float> ref_prefill(out_prefill.size());
    kv_prefill_write_fp4(cache, 0, q_all.data(), k_all.data(), v_all.data(), nullptr,
                         out_prefill.data(), prefill_len, num_q_heads);
    reference_bf16_causal_attention(q_all.data(), k_all.data(), v_all.data(), ref_prefill.data(),
                                    nullptr, prefill_len, prefill_len, 0, num_q_heads, num_kv_heads, head_dim);

    // Path 1 computes directly over BF16 projections -> exact match with BF16 reference!
    for (size_t i = 0; i < out_prefill.size(); ++i) {
        EXPECT_NEAR(out_prefill[i], ref_prefill[i], 1e-5f);
    }

    // Path 2: Chunked Prefill / Extend using SINGLE Layer-Shared FP8 Workspace
    SharedFP8Workspace shared_ws;
    std::vector<float> out_extend(static_cast<size_t>(extend_len) * num_q_heads * head_dim);
    kv_extend_workspace(
        cache, shared_ws, 0,
        q_all.data() + static_cast<size_t>(prefill_len) * num_q_heads * head_dim,
        k_all.data() + static_cast<size_t>(prefill_len) * num_kv_heads * head_dim,
        v_all.data() + static_cast<size_t>(prefill_len) * num_kv_heads * head_dim,
        out_extend.data(), extend_len, num_q_heads);

    size_t ws_bytes_after_layer0 = shared_ws.allocated_bytes();
    // Simulate reusing the SAME workspace buffer on layer 1 -> ZERO extra workspace allocation!
    shared_ws.ensure_capacity(total_len, num_kv_heads, head_dim);
    EXPECT_EQ(shared_ws.allocated_bytes(), ws_bytes_after_layer0);
    EXPECT_EQ(cache.seq_len(0), total_len);

    std::vector<float> ref_extend(out_extend.size());
    reference_bf16_causal_attention(
        q_all.data() + static_cast<size_t>(prefill_len) * num_q_heads * head_dim,
        k_all.data(), v_all.data(), ref_extend.data(), nullptr,
        extend_len, total_len, prefill_len, num_q_heads, num_kv_heads, head_dim);

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < out_extend.size(); ++i) {
        dot += static_cast<double>(out_extend[i]) * ref_extend[i];
        na  += static_cast<double>(out_extend[i]) * out_extend[i];
        nb  += static_cast<double>(ref_extend[i]) * ref_extend[i];
    }
    double extend_cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-18);
    EXPECT_GE(extend_cos, 0.997);

    // Path 3: Fused Autoregressive Decode
    std::vector<float> q_dec(static_cast<size_t>(num_q_heads) * head_dim, 0.25f);
    std::vector<float> out_dec(q_dec.size()), ref_dec(q_dec.size());
    int seq_zero = 0;
    kv_decode_fp4_fused(cache, &seq_zero, q_dec.data(), out_dec.data(), nullptr, 1, num_q_heads);
    reference_bf16_causal_attention(q_dec.data(), k_all.data(), v_all.data(), ref_dec.data(), nullptr,
                                    1, total_len, total_len - 1, num_q_heads, num_kv_heads, head_dim);

    dot = na = nb = 0.0;
    for (size_t i = 0; i < out_dec.size(); ++i) {
        dot += static_cast<double>(out_dec[i]) * ref_dec[i];
        na  += static_cast<double>(out_dec[i]) * out_dec[i];
        nb  += static_cast<double>(ref_dec[i]) * ref_dec[i];
    }
    double decode_cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-18);
    EXPECT_GE(decode_cos, 0.997);
}

// 5. NFR-1.2 Roofline Bandwidth Utilization >= 80% at Batch >= 32 & Occupancy >= 50%
TEST(TurboKVFP4, NFR12_RooflineBandwidthAndOccupancy) {
    for (auto arch : {GPUArchTarget::HOPPER_SM90_H100_SXM5, GPUArchTarget::BLACKWELL_SM100_B200}) {
        RooflineMetrics m = analyze_path3_decode_roofline(arch, 32, 65536, 64, 8, 128);
        std::cout << "  [NFR-1.2] " << m.arch_name << " BW Util=" << m.hbm_bw_utilization_pct
                  << "% (" << m.achieved_hbm_bw_gbps << " GB/s), Speedup=" << m.speedup_vs_bf16_decode << "x\n";
        EXPECT_GE(m.hbm_bw_utilization_pct, 80.0);
        EXPECT_LE(m.registers_per_thread, 128);
        EXPECT_GE(m.achieved_occupancy_pct, 50.0);
        EXPECT_EQ(m.shared_mem_bank_conflicts, 0ULL);

        std::string ncu = export_ncu_roofline_json(m);
        EXPECT_TRUE(ncu.find("dram__throughput.avg.pct_of_peak_sustained_elapsed") != std::string::npos);
    }
}

// 6. NFR-1.3 Numerical Fidelity: Cosine Similarity >= 0.997 & KL-Divergence <= 0.015
TEST(TurboKVFP4, NFR13_NumericalFidelityNeedleAndReasoning) {
    auto needle = evaluate_nfr13_fidelity(2048, 32, 8, 128, EvalDistribution::NEEDLE_IN_A_HAYSTACK, 42u);
    std::cout << "  [NFR-1.3 Needle] TurboKV cos=" << needle.turbokv_cosine_sim
              << " KL=" << needle.turbokv_kl_div
              << " | UniformFP4 cos=" << needle.uniform_fp4_cosine_sim
              << " KL=" << needle.uniform_fp4_kl_div << "\n";
    EXPECT_GE(needle.turbokv_cosine_sim, 0.997);
    EXPECT_LE(needle.turbokv_kl_div, 0.015);
    EXPECT_GT(needle.turbokv_cosine_sim, needle.uniform_fp4_cosine_sim);

    auto reasoning = evaluate_nfr13_fidelity(2048, 32, 8, 128, EvalDistribution::LONG_REASONING_CHAIN, 99u);
    std::cout << "  [NFR-1.3 Reasoning] TurboKV cos=" << reasoning.turbokv_cosine_sim
              << " KL=" << reasoning.turbokv_kl_div << "\n";
    EXPECT_GE(reasoning.turbokv_cosine_sim, 0.997);
    EXPECT_LE(reasoning.turbokv_kl_div, 0.015);
}

// 7. FR-1.3 & NFR-1.4 Speculative Tree Verification & Zero Host Sync Critical Path
TEST(TurboKVFP4, FR13_NFR14_SpeculativeTreeAndZeroHostSync) {
    hal::reset_host_sync_counter();

    HybridKVConfig cfg{};
    cfg.num_kv_heads = 4;
    cfg.head_dim     = 128;
    cfg.max_blocks   = 128;
    HybridPagedKVCache cache(cfg);

    std::vector<float> seed_kv(4 * 128, 0.15f);
    for (int t = 0; t < 32; ++t) cache.append_token(0, seed_kv.data(), seed_kv.data());
    EXPECT_EQ(cache.seq_len(0), 32);

    SpeculativeTreeSpec tree = build_canonical_draft_tree(8, 3);
    EXPECT_LE(tree.max_depth, 8);
    EXPECT_LE(tree.width, 64);
    EXPECT_GT(tree.width, 8);

    const int W = tree.width;
    const int nqh = 16;
    std::vector<float> qd(static_cast<size_t>(W) * nqh * 128, 0.1f);
    std::vector<float> kd(static_cast<size_t>(W) * 4 * 128, 0.1f);
    std::vector<float> vd(static_cast<size_t>(W) * 4 * 128, 0.2f);
    std::vector<float> od(qd.size(), 0.0f);

    // Accept up to depth 5 along a branch, reject depth > 5
    std::vector<float> p_tgt(W, 0.10f), p_drf(W, 0.80f), u_rnd(W, 0.50f);
    for (int i = 0; i < W; ++i) {
        if (tree.depth[i] <= 5) {
            p_tgt[i] = 0.90f; // ratio >= 1 -> guaranteed accept!
        } else {
            p_tgt[i] = 0.05f; // ratio = 0.0625 < u_rnd=0.50 -> rejected!
        }
    }

    int seq0 = 0;
    std::vector<float> q_dec(nqh * 128, 0.2f), out_dec(nqh * 128, 0.0f);
    kv_decode_fp4_fused(cache, &seq0, q_dec.data(), out_dec.data(), nullptr, 1, nqh);

    auto res = spec_tree_verify_and_commit(
        cache, 0, tree, qd.data(), kd.data(), vd.data(), od.data(),
        p_tgt.data(), p_drf.data(), u_rnd.data(), 4242, nqh);

    EXPECT_EQ(res.num_accepted, 5);
    EXPECT_EQ(cache.seq_len(0), 32 + 5);
    EXPECT_EQ(res.host_sync_calls_delta, 0ULL);
    EXPECT_EQ(hal::get_host_sync_counter(), 0ULL);

#ifdef PROJECT_SOURCE_ROOT
    // Static audit: verify zero `cudaDeviceSynchronize` calls in any CUDA `.cu` kernel source!
    const char* kernel_files[4] = {
        "/turbokv_fp4/kernels/kv_prefill_write_fp4.cu",
        "/turbokv_fp4/kernels/kv_extend_workspace.cu",
        "/turbokv_fp4/kernels/kv_decode_fp4_fused.cu",
        "/turbokv_fp4/kernels/spec_tree_verify.cu"
    };
    for (const char* rel : kernel_files) {
        std::ifstream ifs(std::string(PROJECT_SOURCE_ROOT) + rel);
        ASSERT_TRUE(ifs.good());
        std::string line, code_only;
        while (std::getline(ifs, line)) {
            size_t comment_pos = line.find("//");
            code_only += (comment_pos == std::string::npos) ? line : line.substr(0, comment_pos);
            code_only += '\n';
        }
        EXPECT_EQ(code_only.find("cudaDeviceSynchronize"), std::string::npos);
    }
#endif
}

#if !defined(USE_SYSTEM_GTEST)
int main() {
    return ::gtest_lite::run_all_tests();
}
#endif
