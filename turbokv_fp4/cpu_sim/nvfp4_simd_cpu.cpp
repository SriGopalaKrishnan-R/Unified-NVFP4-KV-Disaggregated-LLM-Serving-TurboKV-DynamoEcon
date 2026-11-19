#include "turbokv/paged_attention.h"
#include "turbokv/spec_tree_verify.h"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>

#if defined(_OPENMP)
    #include <omp.h>
#endif

#if defined(TURBOKV_HAS_AVX2)
    #include <immintrin.h>
#endif

namespace turbokv {

namespace {

inline float dot_product_simd(const float* a, const float* b, int d) noexcept {
#if defined(TURBOKV_HAS_AVX2)
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    __m128 lo  = _mm256_castps256_ps128(sum);
    __m128 hi  = _mm256_extractf128_ps(sum, 1);
    __m128 s4  = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(s4);
    __m128 s2  = _mm_add_ps(s4, shuf);
    shuf       = _mm_movehl_ps(shuf, s2);
    __m128 s1  = _mm_add_ss(s2, shuf);
    return _mm_cvtss_f32(s1);
#else
    float s = 0.0f;
    for (int i = 0; i < d; ++i) {
        s += a[i] * b[i];
    }
    return s;
#endif
}

inline void fmadd_vec_simd(float* acc, float alpha, float beta, const float* v, int d) noexcept {
#if defined(TURBOKV_HAS_AVX2)
    __m256 valpha = _mm256_set1_ps(alpha);
    __m256 vbeta  = _mm256_set1_ps(beta);
    for (int i = 0; i < d; i += 8) {
        __m256 vacc = _mm256_loadu_ps(acc + i);
        __m256 vv   = _mm256_loadu_ps(v + i);
        vacc = _mm256_mul_ps(vacc, valpha);
        vacc = _mm256_fmadd_ps(vbeta, vv, vacc);
        _mm256_storeu_ps(acc + i, vacc);
    }
#else
    for (int i = 0; i < d; ++i) {
        acc[i] = acc[i] * alpha + beta * v[i];
    }
#endif
}

inline void round_vec_bf16(const float* src, float* dst, int d) noexcept {
    for (int i = 0; i < d; ++i) {
        dst[i] = bf16_to_fp32(fp32_to_bf16_rne(src[i]));
    }
}

} // namespace

// ==============================================================================
// Ground-Truth Full BF16 Causal Attention
// ==============================================================================
void reference_bf16_causal_attention(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    float* attn_probs_out,
    int num_q_tokens,
    int total_kv_tokens,
    int q_start_pos,
    int num_q_heads,
    int num_kv_heads,
    int head_dim)
{
    const int gqa_group = num_q_heads / num_kv_heads;
    const float scale   = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> k_bf16(static_cast<size_t>(total_kv_tokens) * num_kv_heads * head_dim);
    std::vector<float> v_bf16(static_cast<size_t>(total_kv_tokens) * num_kv_heads * head_dim);
    for (size_t i = 0; i < k_bf16.size(); ++i) {
        k_bf16[i] = bf16_to_fp32(fp32_to_bf16_rne(k[i]));
        v_bf16[i] = bf16_to_fp32(fp32_to_bf16_rne(v[i]));
    }

    #pragma omp parallel for collapse(2) if(num_q_tokens * num_q_heads >= 16)
    for (int qt = 0; qt < num_q_tokens; ++qt) {
        for (int qh = 0; qh < num_q_heads; ++qh) {
            const int kvh = qh / gqa_group;
            const int max_kv = std::min(total_kv_tokens - 1, q_start_pos + qt);

            float q_buf[128];
            const float* q_src = q + (static_cast<size_t>(qt) * num_q_heads + qh) * head_dim;
            round_vec_bf16(q_src, q_buf, head_dim);
            for (int i = 0; i < head_dim; ++i) q_buf[i] *= scale;

            float m_prev = -FLT_MAX;
            float l_prev = 0.0f;
            float acc[128] = {0.0f};
            std::vector<float> logits;
            if (attn_probs_out != nullptr) {
                logits.assign(total_kv_tokens, -FLT_MAX);
            }

            for (int t = 0; t <= max_kv; ++t) {
                const float* kt = k_bf16.data() + (static_cast<size_t>(t) * num_kv_heads + kvh) * head_dim;
                const float* vt = v_bf16.data() + (static_cast<size_t>(t) * num_kv_heads + kvh) * head_dim;
                float score = dot_product_simd(q_buf, kt, head_dim);
                if (attn_probs_out != nullptr) {
                    logits[t] = score;
                }
                float m_new = std::max(m_prev, score);
                float alpha = std::exp(m_prev - m_new);
                float beta  = std::exp(score - m_new);
                l_prev = l_prev * alpha + beta;
                fmadd_vec_simd(acc, alpha, beta, vt, head_dim);
                m_prev = m_new;
            }

            float inv_l = 1.0f / std::max(l_prev, 1e-20f);
            float* out_dst = out + (static_cast<size_t>(qt) * num_q_heads + qh) * head_dim;
            for (int i = 0; i < head_dim; ++i) {
                out_dst[i] = bf16_to_fp32(fp32_to_bf16_rne(acc[i] * inv_l));
            }
            if (attn_probs_out != nullptr) {
                float* p_dst = attn_probs_out + (static_cast<size_t>(qt) * num_q_heads + qh) * total_kv_tokens;
                for (int t = 0; t < total_kv_tokens; ++t) {
                    p_dst[t] = (t <= max_kv) ? (std::exp(logits[t] - m_prev) * inv_l) : 0.0f;
                }
            }
        }
    }
}

// ==============================================================================
// FR-1.2 Path 1 — Initial Prefill (kv_prefill_write_fp4)
// ==============================================================================
void kv_prefill_write_fp4(
    HybridPagedKVCache& cache,
    int seq_id,
    const float* q,
    const float* k,
    const float* v,
    const int32_t* slot_mapping,
    float* out,
    int seq_len,
    int num_q_heads)
{
    cache.reset_sequence(seq_id);
    const auto& cfg = cache.config();
    const int d     = cfg.head_dim;
    const int nkvh  = cfg.num_kv_heads;

    // 1. Compute causal attention directly over BF16-rounded QKV projections (no KV cache read!)
    reference_bf16_causal_attention(q, k, v, out, nullptr, seq_len, seq_len, 0, num_q_heads, nkvh, d);

    // 2. Epilogue: Partition tokens into Sink [0, n_sink), Historical [n_sink, win_start), Window [win_start, seq_len)
    const int sink_end  = std::min(seq_len, cfg.n_sink);
    const int win_start = (seq_len > cfg.n_sink + cfg.n_window)
                        ? (seq_len - cfg.n_window)
                        : sink_end;
    const int hist_len  = std::max(0, win_start - sink_end);

    // Calibrate per-tensor global FP32 scales over rotated historical K/V
    if (hist_len > 0) {
        float k_rot_max = 0.0f;
        float v_rot_max = 0.0f;
        std::vector<float> k_rot_all(static_cast<size_t>(hist_len) * nkvh * d);
        std::vector<float> v_rot_all(static_cast<size_t>(hist_len) * nkvh * d);

        #pragma omp parallel for reduction(max: k_rot_max, v_rot_max) if(hist_len >= 64)
        for (int h_idx = 0; h_idx < hist_len; ++h_idx) {
            int t = sink_end + h_idx;
            for (int h = 0; h < nkvh; ++h) {
                float* kr = k_rot_all.data() + (static_cast<size_t>(h_idx) * nkvh + h) * d;
                float* vr = v_rot_all.data() + (static_cast<size_t>(h_idx) * nkvh + h) * d;
                round_vec_bf16(k + (static_cast<size_t>(t) * nkvh + h) * d, kr, d);
                round_vec_bf16(v + (static_cast<size_t>(t) * nkvh + h) * d, vr, d);
                if (cfg.use_hadamard) {
                    fwht_inplace(kr, d);
                    fwht_inplace(vr, d);
                }
                for (int i = 0; i < d; ++i) {
                    k_rot_max = std::max(k_rot_max, std::fabs(kr[i]));
                    v_rot_max = std::max(v_rot_max, std::fabs(vr[i]));
                }
            }
        }
        cache.calibrate_global_scales(seq_id, k_rot_max, v_rot_max);

        // Allocate paged slots sequentially for deterministic block_table population, then pack in parallel
        std::vector<int32_t> phys_slots(hist_len);
        for (int h_idx = 0; h_idx < hist_len; ++h_idx) {
            int32_t auto_slot = cache.allocate_historical_slot(seq_id, h_idx);
            phys_slots[h_idx] = (slot_mapping != nullptr) ? slot_mapping[sink_end + h_idx] : auto_slot;
        }

        #pragma omp parallel for if(hist_len >= 64)
        for (int h_idx = 0; h_idx < hist_len; ++h_idx) {
            const float* kr = k_rot_all.data() + static_cast<size_t>(h_idx) * nkvh * d;
            const float* vr = v_rot_all.data() + static_cast<size_t>(h_idx) * nkvh * d;
            cache.write_historical_token_from_fp32(seq_id, phys_slots[h_idx], kr, vr, true);
        }
    }

    // Pack Sink tokens in FP8 E4M3
    for (int t = 0; t < sink_end; ++t) {
        std::vector<float> kb(static_cast<size_t>(nkvh) * d);
        std::vector<float> vb(static_cast<size_t>(nkvh) * d);
        round_vec_bf16(k + static_cast<size_t>(t) * nkvh * d, kb.data(), nkvh * d);
        round_vec_bf16(v + static_cast<size_t>(t) * nkvh * d, vb.data(), nkvh * d);
        cache.write_sink_token(seq_id, t, kb.data(), vb.data());
    }

    // Pack Local Sliding Window tokens in FP8 E4M3
    for (int t = win_start; t < seq_len; ++t) {
        int ring_slot = (t - cfg.n_sink) % cfg.n_window;
        std::vector<float> kb(static_cast<size_t>(nkvh) * d);
        std::vector<float> vb(static_cast<size_t>(nkvh) * d);
        round_vec_bf16(k + static_cast<size_t>(t) * nkvh * d, kb.data(), nkvh * d);
        round_vec_bf16(v + static_cast<size_t>(t) * nkvh * d, vb.data(), nkvh * d);
        cache.write_window_token(seq_id, ring_slot, kb.data(), vb.data());
    }

    cache.set_lengths_after_prefill(seq_id, seq_len, hist_len);
}

// ==============================================================================
// FR-1.2 Path 2 — Chunked Prefill / Extend (kv_extend_workspace)
// ==============================================================================
void kv_extend_workspace(
    HybridPagedKVCache& cache,
    SharedFP8Workspace& shared_workspace,
    int seq_id,
    const float* q_chunk,
    const float* k_chunk,
    const float* v_chunk,
    float* out_chunk,
    int chunk_len,
    int num_q_heads)
{
    const auto& cfg     = cache.config();
    const int d         = cfg.head_dim;
    const int nkvh      = cfg.num_kv_heads;
    const int prefix_len = cache.seq_len(seq_id);
    const int hist_len   = cache.hist_len(seq_id);
    const int total_len  = prefix_len + chunk_len;

    // 1. Ensure single layer-shared FP8 workspace capacity (reused across all layers!)
    shared_workspace.ensure_capacity(total_len, nkvh, d);

    auto write_fp8_ws_token_head = [&](int t, int h, const float* kf, const float* vf) {
        size_t sc_idx = static_cast<size_t>(t) * nkvh + h;
        float k_amax = 0.0f, v_amax = 0.0f;
        for (int i = 0; i < d; ++i) {
            k_amax = std::max(k_amax, std::fabs(kf[i]));
            v_amax = std::max(v_amax, std::fabs(vf[i]));
        }
        float ks = (k_amax > 1e-12f) ? (k_amax / FP8_E4M3_MAX) : 1.0f;
        float vs = (v_amax > 1e-12f) ? (v_amax / FP8_E4M3_MAX) : 1.0f;
        shared_workspace.k_token_scale[sc_idx] = ks;
        shared_workspace.v_token_scale[sc_idx] = vs;
        uint8_t* kd = &shared_workspace.k_fp8[sc_idx * d];
        uint8_t* vd = &shared_workspace.v_fp8[sc_idx * d];
        for (int i = 0; i < d; ++i) {
            kd[i] = fp32_to_fp8_e4m3_rne(kf[i] / ks);
            vd[i] = fp32_to_fp8_e4m3_rne(vf[i] / vs);
        }
    };

    // 2. Gather cached Sink + Historical NVFP4 (dequantize + inverse FWHT) + Window into FP8 workspace
    const int sink_cnt = std::min(prefix_len, cfg.n_sink);
    for (int t = 0; t < sink_cnt; ++t) {
        float kb[128], vb[128];
        for (int h = 0; h < nkvh; ++h) {
            cache.read_sink_token_head(seq_id, t, h, kb, vb);
            write_fp8_ws_token_head(t, h, kb, vb);
        }
    }

    #pragma omp parallel for if(hist_len >= 32)
    for (int h_idx = 0; h_idx < hist_len; ++h_idx) {
        float kb[128], vb[128];
        int t = cfg.n_sink + h_idx;
        for (int h = 0; h < nkvh; ++h) {
            cache.read_historical_token_rotated_head(seq_id, h_idx, h, kb, vb);
            if (cfg.use_hadamard) {
                fwht_inplace(kb, d);
                fwht_inplace(vb, d);
            }
            write_fp8_ws_token_head(t, h, kb, vb);
        }
    }

    const int win_start = sink_cnt + hist_len;
    for (int t = win_start; t < prefix_len; ++t) {
        int ring_slot = (t - cfg.n_sink) % cfg.n_window;
        float kb[128], vb[128];
        for (int h = 0; h < nkvh; ++h) {
            cache.read_window_token_head(seq_id, ring_slot, h, kb, vb);
            write_fp8_ws_token_head(t, h, kb, vb);
        }
    }

    // 3. Concatenate active chunk's FP8 K/V tokens into shared workspace AND commit to Hybrid Cache
    for (int c = 0; c < chunk_len; ++c) {
        int t = prefix_len + c;
        std::vector<float> kb_all(static_cast<size_t>(nkvh) * d);
        std::vector<float> vb_all(static_cast<size_t>(nkvh) * d);
        for (int h = 0; h < nkvh; ++h) {
            round_vec_bf16(k_chunk + (static_cast<size_t>(c) * nkvh + h) * d, kb_all.data() + h * d, d);
            round_vec_bf16(v_chunk + (static_cast<size_t>(c) * nkvh + h) * d, vb_all.data() + h * d, d);
            write_fp8_ws_token_head(t, h, kb_all.data() + h * d, vb_all.data() + h * d);
        }
        cache.append_token(seq_id, kb_all.data(), vb_all.data());
    }

    // 4. Execute fused causal FlashAttention over [0 .. prefix_len + chunk_len) in FP8 workspace
    const int gqa_group = num_q_heads / nkvh;
    const float scale   = 1.0f / std::sqrt(static_cast<float>(d));

    #pragma omp parallel for collapse(2) if(chunk_len * num_q_heads >= 16)
    for (int c = 0; c < chunk_len; ++c) {
        for (int qh = 0; qh < num_q_heads; ++qh) {
            const int kvh    = qh / gqa_group;
            const int max_kv = prefix_len + c;
            float q_buf[128];
            round_vec_bf16(q_chunk + (static_cast<size_t>(c) * num_q_heads + qh) * d, q_buf, d);
            for (int i = 0; i < d; ++i) q_buf[i] *= scale;

            float m_prev = -FLT_MAX;
            float l_prev = 0.0f;
            float acc[128] = {0.0f};
            float k_dec[128], v_dec[128];

            for (int t = 0; t <= max_kv; ++t) {
                size_t sc_idx = static_cast<size_t>(t) * nkvh + kvh;
                float ks = shared_workspace.k_token_scale[sc_idx];
                float vs = shared_workspace.v_token_scale[sc_idx];
                const uint8_t* kp = &shared_workspace.k_fp8[sc_idx * d];
                const uint8_t* vp = &shared_workspace.v_fp8[sc_idx * d];
                for (int i = 0; i < d; ++i) {
                    k_dec[i] = fp8_e4m3_to_fp32(kp[i]) * ks;
                    v_dec[i] = fp8_e4m3_to_fp32(vp[i]) * vs;
                }
                float score = dot_product_simd(q_buf, k_dec, d);
                float m_new = std::max(m_prev, score);
                float alpha = std::exp(m_prev - m_new);
                float beta  = std::exp(score - m_new);
                l_prev = l_prev * alpha + beta;
                fmadd_vec_simd(acc, alpha, beta, v_dec, d);
                m_prev = m_new;
            }

            float inv_l = 1.0f / std::max(l_prev, 1e-20f);
            float* out_dst = out_chunk + (static_cast<size_t>(c) * num_q_heads + qh) * d;
            for (int i = 0; i < d; ++i) {
                out_dst[i] = bf16_to_fp32(fp32_to_bf16_rne(acc[i] * inv_l));
            }
        }
    }
}

// ==============================================================================
// FR-1.2 Path 3 — Fused Autoregressive Decode (kv_decode_fp4_fused)
// ==============================================================================
void kv_decode_fp4_fused(
    const HybridPagedKVCache& cache,
    const int* seq_ids,
    const float* q_decode,
    float* out_decode,
    float* attn_probs_last,
    int batch_size,
    int num_q_heads)
{
    // Critical path: ZERO recorded_device_synchronize() calls! (NFR-1.4)
    const auto& cfg     = cache.config();
    const int d         = cfg.head_dim;
    const int nkvh      = cfg.num_kv_heads;
    const int gqa_group = num_q_heads / nkvh;
    const float scale   = 1.0f / std::sqrt(static_cast<float>(d));

    #pragma omp parallel for collapse(2) if(batch_size * num_q_heads >= 8)
    for (int b = 0; b < batch_size; ++b) {
        for (int qh = 0; qh < num_q_heads; ++qh) {
            const int seq_id   = seq_ids[b];
            const int seq_len  = cache.seq_len(seq_id);
            const int hist_len = cache.hist_len(seq_id);
            const int kvh      = qh / gqa_group;

            float q_raw[128];
            float q_rot[128];
            round_vec_bf16(q_decode + (static_cast<size_t>(b) * num_q_heads + qh) * d, q_raw, d);
            for (int i = 0; i < d; ++i) {
                q_raw[i] *= scale;
                q_rot[i] = q_raw[i];
            }
            if (cfg.use_hadamard) {
                // Single FWHT per query head prior to scanning historical NVFP4 blocks!
                fwht_inplace(q_rot, d);
            }

            const bool record_probs = (attn_probs_last != nullptr && b == 0);
            std::vector<float> logits;
            if (record_probs) {
                logits.assign(seq_len, -FLT_MAX);
            }

            // 1. Historical NVFP4 Blocks (unpacked via PTX prmt/lop3 logic in rotated domain)
            float m_cur = -FLT_MAX;
            float l_cur = 0.0f;
            float acc[128] = {0.0f};
            float k_buf[128], v_buf[128];

            for (int h_idx = 0; h_idx < hist_len; ++h_idx) {
                cache.read_historical_token_rotated_head(seq_id, h_idx, kvh, k_buf, v_buf);
                float score = dot_product_simd(q_rot, k_buf, d);
                if (record_probs) {
                    logits[cfg.n_sink + h_idx] = score;
                }
                float m_new = std::max(m_cur, score);
                float alpha = std::exp(m_cur - m_new);
                float beta  = std::exp(score - m_new);
                l_cur = l_cur * alpha + beta;
                fmadd_vec_simd(acc, alpha, beta, v_buf, d);
                m_cur = m_new;
            }

            // Single epilogue FWHT to un-rotate accumulated historical V!
            if (cfg.use_hadamard && hist_len > 0) {
                fwht_inplace(acc, d);
            }

            // 2. Attention-Sink Partition (initial N_sink=4 tokens in FP8 E4M3)
            const int sink_cnt = std::min(seq_len, cfg.n_sink);
            for (int s = 0; s < sink_cnt; ++s) {
                cache.read_sink_token_head(seq_id, s, kvh, k_buf, v_buf);
                float score = dot_product_simd(q_raw, k_buf, d);
                if (record_probs) {
                    logits[s] = score;
                }
                float m_new = std::max(m_cur, score);
                float alpha = (l_cur > 0.0f) ? std::exp(m_cur - m_new) : 0.0f;
                float beta  = std::exp(score - m_new);
                l_cur = l_cur * alpha + beta;
                fmadd_vec_simd(acc, alpha, beta, v_buf, d);
                m_cur = m_new;
            }

            // 3. Local Sliding Window Partition (trailing N_win<=128 tokens in FP8 E4M3)
            const int win_start = sink_cnt + hist_len;
            for (int t = win_start; t < seq_len; ++t) {
                int ring_slot = (t - cfg.n_sink) % cfg.n_window;
                cache.read_window_token_head(seq_id, ring_slot, kvh, k_buf, v_buf);
                float score = dot_product_simd(q_raw, k_buf, d);
                if (record_probs) {
                    logits[t] = score;
                }
                float m_new = std::max(m_cur, score);
                float alpha = (l_cur > 0.0f) ? std::exp(m_cur - m_new) : 0.0f;
                float beta  = std::exp(score - m_new);
                l_cur = l_cur * alpha + beta;
                fmadd_vec_simd(acc, alpha, beta, v_buf, d);
                m_cur = m_new;
            }

            float inv_l = 1.0f / std::max(l_cur, 1e-20f);
            float* out_dst = out_decode + (static_cast<size_t>(b) * num_q_heads + qh) * d;
            for (int i = 0; i < d; ++i) {
                out_dst[i] = bf16_to_fp32(fp32_to_bf16_rne(acc[i] * inv_l));
            }

            if (record_probs) {
                float* p_dst = attn_probs_last + static_cast<size_t>(qh) * seq_len;
                for (int t = 0; t < seq_len; ++t) {
                    p_dst[t] = std::exp(logits[t] - m_cur) * inv_l;
                }
            }
        }
    }
}

// ==============================================================================
// FR-1.3 — Parallel Speculative Tree Verification & Acceptance Kernel
// ==============================================================================
SpeculativeTreeSpec build_canonical_draft_tree(int depth, int branching_factor) {
    SpeculativeTreeSpec spec{};
    depth = std::clamp(depth, 1, kSpecMaxDepth);
    branching_factor = std::clamp(branching_factor, 1, 4);

    // Node 0 is always root draft token (depth 1)
    spec.width           = 1;
    spec.max_depth       = 1;
    spec.parent_idx[0]   = -1;
    spec.depth[0]        = 1;
    spec.ancestor_mask[0] = 1ULL;
    spec.token_ids[0]    = 1000;

    // Breadth-first expansion capped cleanly at kSpecMaxWidth (64) and `depth` (<= 8)
    int queue_start = 0;
    int queue_end   = 1;
    while (queue_start < queue_end && spec.width < kSpecMaxWidth) {
        int parent = queue_start++;
        int next_d = spec.depth[parent] + 1;
        if (next_d > depth) continue;

        // Primary spine gets full branching; deeper side-branches decay so deep trees fit in W<=64
        int bf = (parent == 0 || spec.parent_idx[parent] == 0 || (parent < 8)) ? branching_factor : 1;
        for (int b = 0; b < bf && spec.width < kSpecMaxWidth; ++b) {
            int child = spec.width++;
            spec.parent_idx[child]    = parent;
            spec.depth[child]         = next_d;
            spec.ancestor_mask[child] = spec.ancestor_mask[parent] | (1ULL << child);
            spec.token_ids[child]     = 1000 + next_d * 100 + child;
            spec.max_depth            = std::max(spec.max_depth, next_d);
            queue_end = spec.width;
        }
    }
    return spec;
}

SpeculativeVerifyResult spec_tree_verify_and_commit(
    HybridPagedKVCache& cache,
    int seq_id,
    const SpeculativeTreeSpec& tree,
    const float* q_draft,
    const float* k_draft,
    const float* v_draft,
    float* out_draft_attn,
    const float* p_target_nodes,
    const float* p_draft_nodes,
    const float* uniform_rands,
    int bonus_token_candidate,
    int num_q_heads)
{
    const uint64_t sync_before = hal::get_host_sync_counter();

    const auto& cfg     = cache.config();
    const int d         = cfg.head_dim;
    const int nkvh      = cfg.num_kv_heads;
    const int gqa_group = num_q_heads / nkvh;
    const float scale   = 1.0f / std::sqrt(static_cast<float>(d));
    const int W         = std::clamp(tree.width, 0, kSpecMaxWidth);
    const int seq_len   = cache.seq_len(seq_id);
    const int hist_len  = cache.hist_len(seq_id);

    // 1. Single-Pass Bitwise-Masked Tree Attention over [cached_prefix || W draft nodes]
    #pragma omp parallel for collapse(2) if(W * num_q_heads >= 16)
    for (int qn = 0; qn < W; ++qn) {
        for (int qh = 0; qh < num_q_heads; ++qh) {
            const int kvh = qh / gqa_group;
            float q_raw[128], q_rot[128];
            round_vec_bf16(q_draft + (static_cast<size_t>(qn) * num_q_heads + qh) * d, q_raw, d);
            for (int i = 0; i < d; ++i) {
                q_raw[i] *= scale;
                q_rot[i] = q_raw[i];
            }
            if (cfg.use_hadamard) {
                fwht_inplace(q_rot, d);
            }

            float m_cur = -FLT_MAX;
            float l_cur = 0.0f;
            float acc[128] = {0.0f};
            float kb[128], vb[128];

            // Attend to cached historical NVFP4 prefix
            for (int h_idx = 0; h_idx < hist_len; ++h_idx) {
                cache.read_historical_token_rotated_head(seq_id, h_idx, kvh, kb, vb);
                float score = dot_product_simd(q_rot, kb, d);
                float m_new = std::max(m_cur, score);
                float alpha = (l_cur > 0.0f) ? std::exp(m_cur - m_new) : 0.0f;
                float beta  = std::exp(score - m_new);
                l_cur = l_cur * alpha + beta;
                fmadd_vec_simd(acc, alpha, beta, vb, d);
                m_cur = m_new;
            }
            if (cfg.use_hadamard && hist_len > 0) {
                fwht_inplace(acc, d);
            }

            // Attend to cached FP8 Sink + Window
            const int sink_cnt = std::min(seq_len, cfg.n_sink);
            for (int s = 0; s < sink_cnt; ++s) {
                cache.read_sink_token_head(seq_id, s, kvh, kb, vb);
                float score = dot_product_simd(q_raw, kb, d);
                float m_new = std::max(m_cur, score);
                float alpha = (l_cur > 0.0f) ? std::exp(m_cur - m_new) : 0.0f;
                float beta  = std::exp(score - m_new);
                l_cur = l_cur * alpha + beta;
                fmadd_vec_simd(acc, alpha, beta, vb, d);
                m_cur = m_new;
            }
            for (int t = sink_cnt + hist_len; t < seq_len; ++t) {
                int ring_slot = (t - cfg.n_sink) % cfg.n_window;
                cache.read_window_token_head(seq_id, ring_slot, kvh, kb, vb);
                float score = dot_product_simd(q_raw, kb, d);
                float m_new = std::max(m_cur, score);
                float alpha = (l_cur > 0.0f) ? std::exp(m_cur - m_new) : 0.0f;
                float beta  = std::exp(score - m_new);
                l_cur = l_cur * alpha + beta;
                fmadd_vec_simd(acc, alpha, beta, vb, d);
                m_cur = m_new;
            }

            // Attend to visible draft ancestors using 64-bit `ancestor_mask[qn]` + `warp_ffs64`!
            uint64_t mask_bits = tree.ancestor_mask[qn];
            while (mask_bits != 0ULL) {
                int kn = hal::warp_ffs64(mask_bits) - 1;
                mask_bits &= (mask_bits - 1ULL);
                if (kn >= W) break;

                round_vec_bf16(k_draft + (static_cast<size_t>(kn) * nkvh + kvh) * d, kb, d);
                round_vec_bf16(v_draft + (static_cast<size_t>(kn) * nkvh + kvh) * d, vb, d);
                float score = dot_product_simd(q_raw, kb, d);
                float m_new = std::max(m_cur, score);
                float alpha = (l_cur > 0.0f) ? std::exp(m_cur - m_new) : 0.0f;
                float beta  = std::exp(score - m_new);
                l_cur = l_cur * alpha + beta;
                fmadd_vec_simd(acc, alpha, beta, vb, d);
                m_cur = m_new;
            }

            float inv_l = 1.0f / std::max(l_cur, 1e-20f);
            float* dst = out_draft_attn + (static_cast<size_t>(qn) * num_q_heads + qh) * d;
            for (int i = 0; i < d; ++i) {
                dst[i] = bf16_to_fp32(fp32_to_bf16_rne(acc[i] * inv_l));
            }
        }
    }

    // 2. Warp-Level Parallel Rejection-Sampling Scan (__ballot_sync + __ffs equivalent)
    uint64_t node_accepted_bits = 0ULL;
    uint32_t warp_rej_ballot    = 0u;
    for (int n = 0; n < W; ++n) {
        float ratio = p_target_nodes[n] / std::max(p_draft_nodes[n], 1e-12f);
        bool accepted = (uniform_rands[n] <= std::min(1.0f, ratio));
        if (accepted) {
            node_accepted_bits |= (1ULL << n);
        } else if (n < 32) {
            warp_rej_ballot |= (1u << n);
        }
    }

    // Find deepest accepted root-to-node chain: all ancestors must be accepted!
    int best_leaf  = -1;
    int best_depth = 0;
    for (int n = 0; n < W; ++n) {
        uint64_t anc = tree.ancestor_mask[n];
        if ((anc & ~node_accepted_bits) == 0ULL) {
            if (tree.depth[n] > best_depth) {
                best_depth = tree.depth[n];
                best_leaf  = n;
            }
        }
    }

    SpeculativeVerifyResult res{};
    res.warp_ballot_mask = warp_rej_ballot;
    res.first_reject_ffs = hal::warp_ffs(warp_rej_ballot);
    res.bonus_token_id   = bonus_token_candidate;

    if (best_leaf >= 0) {
        uint64_t chain = tree.ancestor_mask[best_leaf];
        while (chain != 0ULL) {
            int nd = hal::warp_ffs64(chain) - 1;
            chain &= (chain - 1ULL);
            int d_idx = tree.depth[nd] - 1;
            if (d_idx >= 0 && d_idx < kSpecMaxDepth) {
                res.accepted_node_indices[d_idx] = nd;
                res.accepted_token_ids[d_idx]    = tree.token_ids[nd];
                res.num_accepted = std::max(res.num_accepted, d_idx + 1);
            }
        }
    }

    // 3. Commit accepted KV blocks directly into the paged Hybrid KV cache (ZERO host sync!)
    std::vector<float> kb_commit(static_cast<size_t>(nkvh) * d);
    std::vector<float> vb_commit(static_cast<size_t>(nkvh) * d);
    for (int a = 0; a < res.num_accepted; ++a) {
        int nd = res.accepted_node_indices[a];
        round_vec_bf16(k_draft + static_cast<size_t>(nd) * nkvh * d, kb_commit.data(), nkvh * d);
        round_vec_bf16(v_draft + static_cast<size_t>(nd) * nkvh * d, vb_commit.data(), nkvh * d);
        cache.append_token(seq_id, kb_commit.data(), vb_commit.data());
    }

    res.host_sync_calls_delta = hal::get_host_sync_counter() - sync_before;
    return res;
}

// ==============================================================================
// NFR-1.3 Empirical Numerical Fidelity Evaluator
// Compares TurboKV-FP4 AND Naive Uniform 4-bit against Full BF16 Ground Truth
// ==============================================================================
BenchmarkFidelityComparison evaluate_nfr13_fidelity(
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    EvalDistribution dist_type,
    uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 0.45f);

    const size_t q_elems  = static_cast<size_t>(num_q_heads) * head_dim;
    const size_t kv_elems = static_cast<size_t>(seq_len) * num_kv_heads * head_dim;
    std::vector<float> q_dec(q_elems);
    std::vector<float> k_all(kv_elems);
    std::vector<float> v_all(kv_elems);

    for (auto& x : q_dec) x = gauss(rng);
    for (auto& x : k_all) x = gauss(rng);
    for (auto& x : v_all) x = gauss(rng);

    // Inject realistic LLM structure (QuaRot / StreamingLLM empirical channel geometry):
    //   - Persistent massive channel bias + scale outliers on Key channels (30x-55x magnitude!)
    //   - StreamingLLM Attention Sinks on initial tokens t in [0..3]
    //   - GQA query heads attend coherently to Sinks + Semantic Needle / Reasoning hops + Local Window
    const int outlier_channels[4] = {3, 19, head_dim / 2 + 3, head_dim - 7};
    float outlier_bias = (dist_type == EvalDistribution::EXTREME_CHANNEL_OUTLIERS) ? 26.0f : 16.5f;
    int gqa = num_q_heads / num_kv_heads;

    // Give GQA query heads coherent semantic query component + per-head subspace variation
    for (int h = 0; h < num_kv_heads; ++h) {
        const float* base_q = q_dec.data() + (h * gqa) * head_dim;
        for (int g = 1; g < gqa; ++g) {
            float* qh_ptr = q_dec.data() + (h * gqa + g) * head_dim;
            for (int i = 0; i < head_dim; ++i) {
                qh_ptr[i] = 0.82f * base_q[i] + 0.35f * qh_ptr[i];
            }
        }
    }

    for (int t = 0; t < seq_len; ++t) {
        for (int h = 0; h < num_kv_heads; ++h) {
            float* kt = k_all.data() + (static_cast<size_t>(t) * num_kv_heads + h) * head_dim;
            float* vt = v_all.data() + (static_cast<size_t>(t) * num_kv_heads + h) * head_dim;
            const float* q_ref = q_dec.data() + (h * gqa) * head_dim;

            // Attention sink tokens t in {0, 1, 2, 3} carry strong baseline anchor alignment
            if (t < 4) {
                float sink_gain = (t == 0) ? 2.05f : 1.55f;
                for (int i = 0; i < head_dim; ++i) {
                    kt[i] = q_ref[i] * sink_gain + 0.12f * gauss(rng);
                    vt[i] *= 2.0f;
                }
            }

            // Local recency context alignment in trailing window [seq_len - 24, seq_len)
            if (t >= seq_len - 24) {
                for (int i = 0; i < head_dim; ++i) {
                    kt[i] += q_ref[i] * 1.25f;
                }
            }

            // Persistent channel-wise RoPE / massive-activation outliers across all K/V tokens
            for (int c_idx = 0; c_idx < 4; ++c_idx) {
                int oc = outlier_channels[c_idx];
                if (oc < head_dim) {
                    float sign = (c_idx & 1) ? -1.0f : 1.0f;
                    kt[oc] += sign * outlier_bias + 2.5f * kt[oc];
                    vt[oc] += sign * (outlier_bias * 0.35f);
                }
            }
        }
    }

    if (dist_type == EvalDistribution::NEEDLE_IN_A_HAYSTACK && seq_len > 256) {
        // Place semantic "needle" span (12 tokens) deep inside historical NVFP4 region!
        int needle_start = seq_len / 2;
        for (int offset = 0; offset < 12; ++offset) {
            int pos = needle_start + offset;
            for (int h = 0; h < num_kv_heads; ++h) {
                float* kt = k_all.data() + (static_cast<size_t>(pos) * num_kv_heads + h) * head_dim;
                float* vt = v_all.data() + (static_cast<size_t>(pos) * num_kv_heads + h) * head_dim;
                const float* q_ref = q_dec.data() + (h * gqa) * head_dim;
                for (int i = 0; i < head_dim; ++i) {
                    kt[i] = q_ref[i] * 1.95f + 0.12f * gauss(rng);
                    vt[i] = gauss(rng) * 1.6f;
                }
                for (int c_idx = 0; c_idx < 4; ++c_idx) {
                    int oc = outlier_channels[c_idx];
                    float sign = (c_idx & 1) ? -1.0f : 1.0f;
                    kt[oc] += sign * outlier_bias;
                    vt[oc] += sign * (outlier_bias * 0.35f);
                }
            }
        }
    } else if (dist_type == EvalDistribution::LONG_REASONING_CHAIN && seq_len > 256) {
        // Multi-hop reasoning spans (4 reasoning milestones x 6 tokens each across historical NVFP4 + window)
        int hops[4] = {seq_len / 5, (2 * seq_len) / 5, (3 * seq_len) / 4, seq_len - 36};
        for (int hop_idx = 0; hop_idx < 4; ++hop_idx) {
            int base_pos = std::clamp(hops[hop_idx], 8, seq_len - 12);
            for (int off = 0; off < 6; ++off) {
                int pos = base_pos + off;
                for (int h = 0; h < num_kv_heads; ++h) {
                    float* kt = k_all.data() + (static_cast<size_t>(pos) * num_kv_heads + h) * head_dim;
                    const float* q_ref = q_dec.data() + (h * gqa) * head_dim;
                    for (int i = 0; i < head_dim; ++i) {
                        kt[i] = q_ref[i] * 1.85f + 0.12f * gauss(rng);
                    }
                    for (int c_idx = 0; c_idx < 4; ++c_idx) {
                        int oc = outlier_channels[c_idx];
                        float sign = (c_idx & 1) ? -1.0f : 1.0f;
                        kt[oc] += sign * outlier_bias;
                    }
                }
            }
        }
    }

    // Measure raw vs Hadamard-rotated crest factor (max / rms) on a historical token
    BenchmarkFidelityComparison report{};
    {
        int sample_t = std::min(seq_len / 2, seq_len - 1);
        std::vector<float> sample(k_all.begin() + static_cast<size_t>(sample_t) * head_dim,
                                  k_all.begin() + static_cast<size_t>(sample_t + 1) * head_dim);
        double sq = 0.0, mx = 0.0;
        for (float v : sample) { sq += v * v; mx = std::max<double>(mx, std::fabs(v)); }
        report.raw_channel_crest_factor = mx / std::max(std::sqrt(sq / head_dim), 1e-9);

        fwht_inplace(sample.data(), head_dim);
        double sq_r = 0.0, mx_r = 0.0;
        for (float v : sample) { sq_r += v * v; mx_r = std::max<double>(mx_r, std::fabs(v)); }
        report.rotated_channel_crest_factor = mx_r / std::max(std::sqrt(sq_r / head_dim), 1e-9);
    }

    // 1. Ground-Truth BF16 Decode Output & Exact Softmax Distribution
    std::vector<float> out_bf16(q_elems, 0.0f);
    std::vector<float> probs_bf16(static_cast<size_t>(num_q_heads) * seq_len, 0.0f);
    reference_bf16_causal_attention(
        q_dec.data(), k_all.data(), v_all.data(),
        out_bf16.data(), probs_bf16.data(),
        1, seq_len, seq_len - 1, num_q_heads, num_kv_heads, head_dim);

    // 2. TurboKV-FP4: Prefill into Hybrid Cache (scrambled non-contiguous pages!) + Path 3 Decode
    HybridKVConfig cfg{};
    cfg.num_kv_heads = num_kv_heads;
    cfg.head_dim     = head_dim;
    cfg.page_size    = 16;
    cfg.n_sink       = 4;
    cfg.n_window     = 128;
    cfg.max_batch    = 2;
    cfg.max_blocks   = (seq_len / 16) + 64;
    cfg.use_hadamard = true;

    HybridPagedKVCache cache(cfg);
    cache.scramble_free_pool();

    // Populate KV cache via Path 1 epilogue (using dummy q for prefill write, since we only need KV packed)
    std::vector<float> dummy_q(static_cast<size_t>(seq_len) * num_q_heads * head_dim, 0.0f);
    std::vector<float> dummy_out(dummy_q.size(), 0.0f);
    // Note: for large seq_len in fast benchmark mode, pack KV directly using same exact Path 1 epilogue!
    if (seq_len <= 1024) {
        kv_prefill_write_fp4(cache, 0, dummy_q.data(), k_all.data(), v_all.data(), nullptr, dummy_out.data(), seq_len, num_q_heads);
    } else {
        // Fast exact KV population identical to Path 1 epilogue without O(L^2) dummy prefill attention
        const int sink_end  = std::min(seq_len, cfg.n_sink);
        const int win_start = (seq_len > cfg.n_sink + cfg.n_window) ? (seq_len - cfg.n_window) : sink_end;
        const int hist_len  = std::max(0, win_start - sink_end);
        float k_rmax = 0.0f, v_rmax = 0.0f;
        std::vector<float> kr(static_cast<size_t>(hist_len) * num_kv_heads * head_dim);
        std::vector<float> vr(static_cast<size_t>(hist_len) * num_kv_heads * head_dim);
        for (int h_idx = 0; h_idx < hist_len; ++h_idx) {
            int t = sink_end + h_idx;
            for (int h = 0; h < num_kv_heads; ++h) {
                float* kdst = kr.data() + (static_cast<size_t>(h_idx) * num_kv_heads + h) * head_dim;
                float* vdst = vr.data() + (static_cast<size_t>(h_idx) * num_kv_heads + h) * head_dim;
                round_vec_bf16(k_all.data() + (static_cast<size_t>(t) * num_kv_heads + h) * head_dim, kdst, head_dim);
                round_vec_bf16(v_all.data() + (static_cast<size_t>(t) * num_kv_heads + h) * head_dim, vdst, head_dim);
                fwht_inplace(kdst, head_dim);
                fwht_inplace(vdst, head_dim);
                for (int i = 0; i < head_dim; ++i) {
                    k_rmax = std::max(k_rmax, std::fabs(kdst[i]));
                    v_rmax = std::max(v_rmax, std::fabs(vdst[i]));
                }
            }
        }
        cache.calibrate_global_scales(0, k_rmax, v_rmax);
        for (int h_idx = 0; h_idx < hist_len; ++h_idx) {
            int32_t slot = cache.allocate_historical_slot(0, h_idx);
            cache.write_historical_token_from_fp32(
                0, slot,
                kr.data() + static_cast<size_t>(h_idx) * num_kv_heads * head_dim,
                vr.data() + static_cast<size_t>(h_idx) * num_kv_heads * head_dim,
                true);
        }
        std::vector<float> kb(static_cast<size_t>(num_kv_heads) * head_dim);
        std::vector<float> vb(static_cast<size_t>(num_kv_heads) * head_dim);
        for (int t = 0; t < sink_end; ++t) {
            round_vec_bf16(k_all.data() + static_cast<size_t>(t) * num_kv_heads * head_dim, kb.data(), num_kv_heads * head_dim);
            round_vec_bf16(v_all.data() + static_cast<size_t>(t) * num_kv_heads * head_dim, vb.data(), num_kv_heads * head_dim);
            cache.write_sink_token(0, t, kb.data(), vb.data());
        }
        for (int t = win_start; t < seq_len; ++t) {
            round_vec_bf16(k_all.data() + static_cast<size_t>(t) * num_kv_heads * head_dim, kb.data(), num_kv_heads * head_dim);
            round_vec_bf16(v_all.data() + static_cast<size_t>(t) * num_kv_heads * head_dim, vb.data(), num_kv_heads * head_dim);
            cache.write_window_token(0, (t - cfg.n_sink) % cfg.n_window, kb.data(), vb.data());
        }
        cache.set_lengths_after_prefill(0, seq_len, hist_len);
    }

    int seq_id_zero = 0;
    std::vector<float> out_tkv(q_elems, 0.0f);
    std::vector<float> probs_tkv(static_cast<size_t>(num_q_heads) * seq_len, 0.0f);
    kv_decode_fp4_fused(cache, &seq_id_zero, q_dec.data(), out_tkv.data(), probs_tkv.data(), 1, num_q_heads);

    // 3. Baseline Uniform 4-bit (No Hadamard, No Sink/Window, Single Per-Tensor Scale -> Industry Cliff!)
    float k_absmax = 0.0f, v_absmax = 0.0f;
    for (float x : k_all) k_absmax = std::max(k_absmax, std::fabs(x));
    for (float x : v_all) v_absmax = std::max(v_absmax, std::fabs(x));
    float k_unif_scale = std::max(k_absmax / NVFP4_E2M1_MAX, 1e-12f);
    float v_unif_scale = std::max(v_absmax / NVFP4_E2M1_MAX, 1e-12f);

    std::vector<float> k_unif(kv_elems);
    std::vector<float> v_unif(kv_elems);
    for (size_t i = 0; i < kv_elems; ++i) {
        k_unif[i] = nvfp4_e2m1_to_fp32(fp32_to_nvfp4_e2m1_rne(k_all[i] / k_unif_scale)) * k_unif_scale;
        v_unif[i] = nvfp4_e2m1_to_fp32(fp32_to_nvfp4_e2m1_rne(v_all[i] / v_unif_scale)) * v_unif_scale;
    }
    std::vector<float> out_unif(q_elems, 0.0f);
    std::vector<float> probs_unif(static_cast<size_t>(num_q_heads) * seq_len, 0.0f);
    reference_bf16_causal_attention(
        q_dec.data(), k_unif.data(), v_unif.data(),
        out_unif.data(), probs_unif.data(),
        1, seq_len, seq_len - 1, num_q_heads, num_kv_heads, head_dim);

    // Compute Cosine Similarity, RMSE, and Mean Head KL-Divergence D_KL(P_bf16 || Q_quant)
    auto compute_stats = [&](const std::vector<float>& cand_out,
                             const std::vector<float>& cand_probs,
                             double& cos_sim, double& kl_div, double& rmse) {
        double dot = 0.0, na = 0.0, nb = 0.0, sq_err = 0.0;
        for (size_t i = 0; i < q_elems; ++i) {
            double a = out_bf16[i];
            double b = cand_out[i];
            dot    += a * b;
            na     += a * a;
            nb     += b * b;
            sq_err += (a - b) * (a - b);
        }
        cos_sim = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-18);
        rmse    = std::sqrt(sq_err / static_cast<double>(q_elems));

        double total_kl = 0.0;
        for (int qh = 0; qh < num_q_heads; ++qh) {
            double head_kl = 0.0;
            const float* p = probs_bf16.data() + static_cast<size_t>(qh) * seq_len;
            const float* q = cand_probs.data() + static_cast<size_t>(qh) * seq_len;
            for (int t = 0; t < seq_len; ++t) {
                double pt = std::max<double>(p[t], 1e-12);
                double qt = std::max<double>(q[t], 1e-12);
                if (p[t] > 1e-9f) {
                    head_kl += pt * std::log(pt / qt);
                }
            }
            total_kl += std::max(0.0, head_kl);
        }
        kl_div = total_kl / static_cast<double>(num_q_heads);
    };

    compute_stats(out_tkv,  probs_tkv,  report.turbokv_cosine_sim,     report.turbokv_kl_div,     report.turbokv_rmse);
    compute_stats(out_unif, probs_unif, report.uniform_fp4_cosine_sim, report.uniform_fp4_kl_div, report.uniform_fp4_rmse);
    return report;
}

} // namespace turbokv
