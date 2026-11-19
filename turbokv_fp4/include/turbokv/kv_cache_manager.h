#pragma once

#include "turbokv/nvfp4_types.h"
#include "turbokv/walsh_hadamard.h"
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace turbokv {

// ==============================================================================
// FR-1.1: Two-Level Microscaled NVFP4 + Attention-Sink Hybrid Paged KV Layout
//
// Precision-Zone Partitioning per Sequence:
//   - Attention-Sink Zone: Initial N_sink = 4 tokens stored in unquantized FP8 (E4M3)
//     with exact per-token FP32 scale factor to preserve initial softmax sink anchors.
//   - Local Sliding Window: Trailing N_win = 128 tokens stored in unquantized FP8 (E4M3)
//     with per-token FP32 scale factor to preserve local recency precision.
//   - Historical Compressed Blocks: Remaining tokens [N_sink, seq_len - N_win) stored
//     in non-contiguous paged NVFP4 (E2M1) blocks (page_size = 16 tokens), using:
//       * Walsh-Hadamard outlier pre-conditioning across d in {64, 128}
//       * Per-16-element FP8 E4M3 micro-scales
//       * Per-tensor FP32 global scales (K_global_scale, V_global_scale)
// ==============================================================================

struct HybridKVConfig {
    int num_kv_heads  = 8;
    int head_dim      = 128;   // {64, 128}
    int page_size     = 16;    // tokens per paged KV block
    int n_sink        = 4;     // FR-1.1: N_sink = 4
    int n_window      = 128;   // FR-1.1: N_win  = 128
    int max_batch     = 128;
    int max_blocks    = 32768;
    bool use_hadamard = true;
};

struct MemoryFootprintReport {
    size_t seq_len               = 0;
    size_t bf16_bytes            = 0;
    size_t sink_fp8_bytes        = 0;
    size_t window_fp8_bytes      = 0;
    size_t nvfp4_payload_bytes   = 0;
    size_t micro_scale_fp8_bytes = 0;
    size_t global_scale_bytes    = 0;
    size_t block_table_bytes     = 0;
    size_t total_hybrid_bytes    = 0;
    double compression_ratio     = 1.0;
};

// Layer-Shared FP8 Workspace Buffer for FR-1.2 Path 2 (Chunked Prefill / Extend).
// Reused across all transformer layers so workspace HBM scales O(1) with num_layers!
struct SharedFP8Workspace {
    std::vector<uint8_t> k_fp8;
    std::vector<uint8_t> v_fp8;
    std::vector<float>   k_token_scale;
    std::vector<float>   v_token_scale;
    size_t               reuses_count = 0;

    void ensure_capacity(int max_tokens, int num_kv_heads, int head_dim) {
        size_t elem_cap = static_cast<size_t>(max_tokens) * num_kv_heads * head_dim;
        size_t scale_cap = static_cast<size_t>(max_tokens) * num_kv_heads;
        if (k_fp8.size() < elem_cap) {
            k_fp8.resize(elem_cap);
            v_fp8.resize(elem_cap);
        }
        if (k_token_scale.size() < scale_cap) {
            k_token_scale.resize(scale_cap);
            v_token_scale.resize(scale_cap);
        }
        ++reuses_count;
    }

    size_t allocated_bytes() const noexcept {
        return k_fp8.size() + v_fp8.size() +
               (k_token_scale.size() + v_token_scale.size()) * sizeof(float);
    }
};

class HybridPagedKVCache {
public:
    explicit HybridPagedKVCache(const HybridKVConfig& cfg)
        : cfg_(cfg)
    {
        if (cfg_.head_dim != 64 && cfg_.head_dim != 128) {
            throw std::invalid_argument("HybridPagedKVCache requires head_dim in {64, 128}");
        }
        if (cfg_.page_size <= 0) {
            throw std::invalid_argument("page_size must be positive");
        }

        // Packed uint4 words per token-head: 32 NVFP4 elements per uint4 (16 bytes)
        // head_dim=64 -> 2 uint4s (32 bytes); head_dim=128 -> 4 uint4s (64 bytes)
        uint4_per_head_       = cfg_.head_dim / 32;
        microblocks_per_head_ = cfg_.head_dim / NVFP4_MICRO_BLOCK_SIZE;

        size_t total_block_slots = static_cast<size_t>(cfg_.max_blocks) * cfg_.page_size * cfg_.num_kv_heads;
        k_nvfp4_pool_.resize(total_block_slots * uint4_per_head_);
        v_nvfp4_pool_.resize(total_block_slots * uint4_per_head_);
        k_micro_scales_.resize(total_block_slots * microblocks_per_head_, 0u);
        v_micro_scales_.resize(total_block_slots * microblocks_per_head_, 0u);

        // Free block stack (reversed so physical block allocation is non-trivial/realistic;
        // also supports shuffled non-contiguous page allocation for paged table tests)
        free_blocks_.reserve(cfg_.max_blocks);
        for (int b = cfg_.max_blocks - 1; b >= 0; --b) {
            free_blocks_.push_back(b);
        }

        seq_lengths_.assign(cfg_.max_batch, 0);
        hist_lengths_.assign(cfg_.max_batch, 0);
        block_tables_.resize(cfg_.max_batch);

        // Attention-Sink zone: [max_batch, n_sink, num_kv_heads, head_dim] in FP8 E4M3 + FP32 scale
        size_t sink_elems  = static_cast<size_t>(cfg_.max_batch) * cfg_.n_sink * cfg_.num_kv_heads * cfg_.head_dim;
        size_t sink_scales = static_cast<size_t>(cfg_.max_batch) * cfg_.n_sink * cfg_.num_kv_heads;
        k_sink_fp8_.assign(sink_elems, 0u);
        v_sink_fp8_.assign(sink_elems, 0u);
        k_sink_scale_.assign(sink_scales, 1.0f);
        v_sink_scale_.assign(sink_scales, 1.0f);

        // Sliding-Window ring buffer: [max_batch, n_window, num_kv_heads, head_dim] in FP8 E4M3 + FP32 scale
        size_t win_elems  = static_cast<size_t>(cfg_.max_batch) * cfg_.n_window * cfg_.num_kv_heads * cfg_.head_dim;
        size_t win_scales = static_cast<size_t>(cfg_.max_batch) * cfg_.n_window * cfg_.num_kv_heads;
        k_win_fp8_.assign(win_elems, 0u);
        v_win_fp8_.assign(win_elems, 0u);
        k_win_scale_.assign(win_scales, 1.0f);
        v_win_scale_.assign(win_scales, 1.0f);

        k_global_scale_.assign(cfg_.max_batch, 1.0f / 448.0f);
        v_global_scale_.assign(cfg_.max_batch, 1.0f / 448.0f);
    }

    const HybridKVConfig& config() const noexcept { return cfg_; }
    int seq_len(int seq_id) const noexcept { return seq_lengths_[seq_id]; }
    int hist_len(int seq_id) const noexcept { return hist_lengths_[seq_id]; }
    const std::vector<int32_t>& block_table(int seq_id) const noexcept { return block_tables_[seq_id]; }

    float k_global_scale(int seq_id) const noexcept { return k_global_scale_[seq_id]; }
    float v_global_scale(int seq_id) const noexcept { return v_global_scale_[seq_id]; }

    const uint4* k_nvfp4_pool() const noexcept { return k_nvfp4_pool_.data(); }
    const uint4* v_nvfp4_pool() const noexcept { return v_nvfp4_pool_.data(); }
    const uint8_t* k_micro_scales() const noexcept { return k_micro_scales_.data(); }
    const uint8_t* v_micro_scales() const noexcept { return v_micro_scales_.data(); }

    void reset_sequence(int seq_id) {
        for (int32_t blk : block_tables_[seq_id]) {
            free_blocks_.push_back(blk);
        }
        block_tables_[seq_id].clear();
        seq_lengths_[seq_id]  = 0;
        hist_lengths_[seq_id] = 0;
        k_global_scale_[seq_id] = 1.0f / 448.0f;
        v_global_scale_[seq_id] = 1.0f / 448.0f;
    }

    // Scrambles free-block order with a deterministic stride to stress non-contiguous paged indexing
    void scramble_free_pool() {
        std::vector<int32_t> reordered;
        reordered.reserve(free_blocks_.size());
        size_t n = free_blocks_.size();
        for (size_t step = 0; step < 7 && step < n; ++step) {
            for (size_t idx = step; idx < n; idx += 7) {
                reordered.push_back(free_blocks_[idx]);
            }
        }
        free_blocks_.swap(reordered);
    }

    // Calibrates per-tensor FP32 global scale from rotated absmax so micro-scales utilize full E4M3 range [0, 448]
    void calibrate_global_scales(int seq_id, float k_rotated_absmax, float v_rotated_absmax) noexcept {
        constexpr float kDenom = NVFP4_E2M1_MAX * FP8_E4M3_MAX; // 6.0 * 448.0 = 2688.0
        k_global_scale_[seq_id] = std::max(k_global_scale_[seq_id], std::max(k_rotated_absmax, 1e-6f) / kDenom);
        v_global_scale_[seq_id] = std::max(v_global_scale_[seq_id], std::max(v_rotated_absmax, 1e-6f) / kDenom);
    }

    // Allocates physical slot for historical index `hist_idx` (0-based offset after N_sink)
    int32_t allocate_historical_slot(int seq_id, int hist_idx) {
        int logical_block = hist_idx / cfg_.page_size;
        int block_offset  = hist_idx % cfg_.page_size;
        while (static_cast<int>(block_tables_[seq_id].size()) <= logical_block) {
            if (free_blocks_.empty()) {
                throw std::runtime_error("HybridPagedKVCache out of physical NVFP4 blocks");
            }
            int32_t phys_blk = free_blocks_.back();
            free_blocks_.pop_back();
            block_tables_[seq_id].push_back(phys_blk);
        }
        int32_t phys_blk = block_tables_[seq_id][logical_block];
        return phys_blk * cfg_.page_size + block_offset;
    }

    // Encodes a single historical token [num_kv_heads, head_dim] into physical slot `phys_slot`
    // Applies Walsh-Hadamard pre-conditioning + Two-Level NVFP4 packing into uint4 words.
    void write_historical_token_from_fp32(
        int seq_id,
        int32_t phys_slot,
        const float* k_token,
        const float* v_token,
        bool already_rotated = false) noexcept
    {
        float k_buf[128];
        float v_buf[128];
        const int d = cfg_.head_dim;

        for (int h = 0; h < cfg_.num_kv_heads; ++h) {
            std::memcpy(k_buf, k_token + h * d, d * sizeof(float));
            std::memcpy(v_buf, v_token + h * d, d * sizeof(float));
            if (cfg_.use_hadamard && !already_rotated) {
                fwht_inplace(k_buf, d);
                fwht_inplace(v_buf, d);
            }

            size_t head_slot = static_cast<size_t>(phys_slot) * cfg_.num_kv_heads + h;
            uint4* k_u4 = &k_nvfp4_pool_[head_slot * uint4_per_head_];
            uint4* v_u4 = &v_nvfp4_pool_[head_slot * uint4_per_head_];
            uint8_t* k_ms = &k_micro_scales_[head_slot * microblocks_per_head_];
            uint8_t* v_ms = &v_micro_scales_[head_slot * microblocks_per_head_];

            // Each uint4 holds 32 NVFP4 elements = 2 micro-blocks of 16 elements
            for (int u = 0; u < uint4_per_head_; ++u) {
                MicroBlockNVFP4 km0 = quantize_microblock_16(k_buf + u * 32,      k_global_scale_[seq_id]);
                MicroBlockNVFP4 km1 = quantize_microblock_16(k_buf + u * 32 + 16, k_global_scale_[seq_id]);
                k_u4[u] = make_uint4(km0.lo8, km0.hi8, km1.lo8, km1.hi8);
                k_ms[u * 2 + 0] = km0.micro_scale_e4m3;
                k_ms[u * 2 + 1] = km1.micro_scale_e4m3;

                MicroBlockNVFP4 vm0 = quantize_microblock_16(v_buf + u * 32,      v_global_scale_[seq_id]);
                MicroBlockNVFP4 vm1 = quantize_microblock_16(v_buf + u * 32 + 16, v_global_scale_[seq_id]);
                v_u4[u] = make_uint4(vm0.lo8, vm0.hi8, vm1.lo8, vm1.hi8);
                v_ms[u * 2 + 0] = vm0.micro_scale_e4m3;
                v_ms[u * 2 + 1] = vm1.micro_scale_e4m3;
            }
        }
    }

    // Writes a token into the FP8 E4M3 Sink zone (sink_idx in [0, n_sink))
    void write_sink_token(int seq_id, int sink_idx, const float* k_token, const float* v_token) noexcept {
        const int d = cfg_.head_dim;
        for (int h = 0; h < cfg_.num_kv_heads; ++h) {
            size_t sc_idx = (static_cast<size_t>(seq_id) * cfg_.n_sink + sink_idx) * cfg_.num_kv_heads + h;
            uint8_t* k_dst = &k_sink_fp8_[sc_idx * d];
            uint8_t* v_dst = &v_sink_fp8_[sc_idx * d];
            quantize_vec_fp8_e4m3(k_token + h * d, d, k_dst, k_sink_scale_[sc_idx]);
            quantize_vec_fp8_e4m3(v_token + h * d, d, v_dst, v_sink_scale_[sc_idx]);
        }
    }

    // Writes a token into the FP8 E4M3 Sliding Window ring slot (ring_slot = pos % n_window)
    void write_window_token(int seq_id, int ring_slot, const float* k_token, const float* v_token) noexcept {
        const int d = cfg_.head_dim;
        for (int h = 0; h < cfg_.num_kv_heads; ++h) {
            size_t sc_idx = (static_cast<size_t>(seq_id) * cfg_.n_window + ring_slot) * cfg_.num_kv_heads + h;
            uint8_t* k_dst = &k_win_fp8_[sc_idx * d];
            uint8_t* v_dst = &v_win_fp8_[sc_idx * d];
            quantize_vec_fp8_e4m3(k_token + h * d, d, k_dst, k_win_scale_[sc_idx]);
            quantize_vec_fp8_e4m3(v_token + h * d, d, v_dst, v_win_scale_[sc_idx]);
        }
    }

    void read_sink_token_head(int seq_id, int sink_idx, int kv_head, float* k_out, float* v_out) const noexcept {
        const int d = cfg_.head_dim;
        size_t sc_idx = (static_cast<size_t>(seq_id) * cfg_.n_sink + sink_idx) * cfg_.num_kv_heads + kv_head;
        const uint8_t* k_src = &k_sink_fp8_[sc_idx * d];
        const uint8_t* v_src = &v_sink_fp8_[sc_idx * d];
        float ks = k_sink_scale_[sc_idx];
        float vs = v_sink_scale_[sc_idx];
        for (int i = 0; i < d; ++i) {
            k_out[i] = fp8_e4m3_to_fp32(k_src[i]) * ks;
            v_out[i] = fp8_e4m3_to_fp32(v_src[i]) * vs;
        }
    }

    void read_window_token_head(int seq_id, int ring_slot, int kv_head, float* k_out, float* v_out) const noexcept {
        const int d = cfg_.head_dim;
        size_t sc_idx = (static_cast<size_t>(seq_id) * cfg_.n_window + ring_slot) * cfg_.num_kv_heads + kv_head;
        const uint8_t* k_src = &k_win_fp8_[sc_idx * d];
        const uint8_t* v_src = &v_win_fp8_[sc_idx * d];
        float ks = k_win_scale_[sc_idx];
        float vs = v_win_scale_[sc_idx];
        for (int i = 0; i < d; ++i) {
            k_out[i] = fp8_e4m3_to_fp32(k_src[i]) * ks;
            v_out[i] = fp8_e4m3_to_fp32(v_src[i]) * vs;
        }
    }

    // Unpacks historical NVFP4 token IN ROTATED HADAMARD DOMAIN (used directly by Path 3 Decode
    // so zero per-token inverse FWHT is needed inside the hot attention loop!)
    void read_historical_token_rotated_head(
        int seq_id,
        int hist_idx,
        int kv_head,
        float* k_rot_out,
        float* v_rot_out) const noexcept
    {
        int logical_block = hist_idx / cfg_.page_size;
        int block_offset  = hist_idx % cfg_.page_size;
        int32_t phys_blk  = block_tables_[seq_id][logical_block];
        int32_t phys_slot = phys_blk * cfg_.page_size + block_offset;

        size_t head_slot = static_cast<size_t>(phys_slot) * cfg_.num_kv_heads + kv_head;
        const uint4* k_u4 = &k_nvfp4_pool_[head_slot * uint4_per_head_];
        const uint4* v_u4 = &v_nvfp4_pool_[head_slot * uint4_per_head_];
        const uint8_t* k_ms = &k_micro_scales_[head_slot * microblocks_per_head_];
        const uint8_t* v_ms = &v_micro_scales_[head_slot * microblocks_per_head_];

        float kgs = k_global_scale_[seq_id];
        float vgs = v_global_scale_[seq_id];

        for (int u = 0; u < uint4_per_head_; ++u) {
            uint4 ku = k_u4[u];
            dequantize_microblock_16(ku.x, ku.y, k_ms[u * 2 + 0], kgs, k_rot_out + u * 32);
            dequantize_microblock_16(ku.z, ku.w, k_ms[u * 2 + 1], kgs, k_rot_out + u * 32 + 16);

            uint4 vu = v_u4[u];
            dequantize_microblock_16(vu.x, vu.y, v_ms[u * 2 + 0], vgs, v_rot_out + u * 32);
            dequantize_microblock_16(vu.z, vu.w, v_ms[u * 2 + 1], vgs, v_rot_out + u * 32 + 16);
        }
    }

    // Appends a single new token (used by Decode & Speculative Tree Commit), automatically
    // graduating the oldest sliding-window token into a paged historical NVFP4 block!
    void append_token(int seq_id, const float* k_token, const float* v_token) {
        int pos = seq_lengths_[seq_id];
        const int d = cfg_.head_dim;

        // Pre-calibrate global scale if this sequence hasn't calibrated historical blocks yet
        if (pos == cfg_.n_sink + cfg_.n_window) {
            float k_rot_max = 0.0f, v_rot_max = 0.0f;
            float kb[128], vb[128];
            for (int w = 0; w < cfg_.n_window; ++w) {
                for (int h = 0; h < cfg_.num_kv_heads; ++h) {
                    read_window_token_head(seq_id, w, h, kb, vb);
                    if (cfg_.use_hadamard) {
                        fwht_inplace(kb, d);
                        fwht_inplace(vb, d);
                    }
                    for (int i = 0; i < d; ++i) {
                        k_rot_max = std::max(k_rot_max, std::fabs(kb[i]));
                        v_rot_max = std::max(v_rot_max, std::fabs(vb[i]));
                    }
                }
            }
            // Apply 2.0x headroom for future decode tokens
            calibrate_global_scales(seq_id, k_rot_max * 2.0f, v_rot_max * 2.0f);
        }

        if (pos < cfg_.n_sink) {
            write_sink_token(seq_id, pos, k_token, v_token);
        } else if (pos < cfg_.n_sink + cfg_.n_window) {
            int ring_slot = (pos - cfg_.n_sink) % cfg_.n_window;
            write_window_token(seq_id, ring_slot, k_token, v_token);
        } else {
            // Evict oldest window token (at ring_slot = (pos - n_sink) % n_window) into historical NVFP4!
            int ring_slot = (pos - cfg_.n_sink) % cfg_.n_window;
            std::vector<float> evicted_k(static_cast<size_t>(cfg_.num_kv_heads) * d);
            std::vector<float> evicted_v(static_cast<size_t>(cfg_.num_kv_heads) * d);
            float k_rot_max = 0.0f, v_rot_max = 0.0f;
            for (int h = 0; h < cfg_.num_kv_heads; ++h) {
                read_window_token_head(seq_id, ring_slot, h,
                                       evicted_k.data() + h * d,
                                       evicted_v.data() + h * d);
                if (cfg_.use_hadamard) {
                    fwht_inplace(evicted_k.data() + h * d, d);
                    fwht_inplace(evicted_v.data() + h * d, d);
                }
                for (int i = 0; i < d; ++i) {
                    k_rot_max = std::max(k_rot_max, std::fabs(evicted_k[h * d + i]));
                    v_rot_max = std::max(v_rot_max, std::fabs(evicted_v[h * d + i]));
                }
            }
            // Note: do NOT retroactively mutate global_scale if historical blocks already exist;
            // micro-scales saturate cleanly up to 448x global_scale.
            if (hist_lengths_[seq_id] == 0) {
                calibrate_global_scales(seq_id, k_rot_max * 2.0f, v_rot_max * 2.0f);
            }
            int hist_idx = hist_lengths_[seq_id]++;
            int32_t phys_slot = allocate_historical_slot(seq_id, hist_idx);
            write_historical_token_from_fp32(seq_id, phys_slot, evicted_k.data(), evicted_v.data(), true);

            // Overwrite ring_slot with newest token in unquantized FP8
            write_window_token(seq_id, ring_slot, k_token, v_token);
        }
        seq_lengths_[seq_id] = pos + 1;
    }

    void set_lengths_after_prefill(int seq_id, int total_len, int hist_len) noexcept {
        seq_lengths_[seq_id]  = total_len;
        hist_lengths_[seq_id] = hist_len;
    }

    // NFR-1.1 Exact HBM Footprint Accounting for a Sequence of Length `seq_len`
    MemoryFootprintReport compute_footprint(size_t seq_len) const noexcept {
        MemoryFootprintReport r{};
        r.seq_len = seq_len;
        size_t kv_factors = 2ULL * cfg_.num_kv_heads * cfg_.head_dim;

        // Baseline BF16: 2 bytes per element for K and V
        r.bf16_bytes = seq_len * kv_factors * 2ULL;

        size_t sink_tokens = std::min<size_t>(seq_len, cfg_.n_sink);
        size_t rem_after_sink = (seq_len > sink_tokens) ? (seq_len - sink_tokens) : 0ULL;
        size_t win_tokens = std::min<size_t>(rem_after_sink, cfg_.n_window);
        size_t hist_tokens = (rem_after_sink > win_tokens) ? (rem_after_sink - win_tokens) : 0ULL;

        // Sink & Window: 1 byte FP8 per element + 4-byte FP32 per-token-head scale
        r.sink_fp8_bytes   = sink_tokens * kv_factors * 1ULL + sink_tokens * 2ULL * cfg_.num_kv_heads * sizeof(float);
        r.window_fp8_bytes = win_tokens  * kv_factors * 1ULL + win_tokens  * 2ULL * cfg_.num_kv_heads * sizeof(float);

        // Historical paged blocks (rounded up to page_size physical blocks!)
        size_t num_pages = (hist_tokens + cfg_.page_size - 1) / cfg_.page_size;
        size_t allocated_hist_tokens = num_pages * cfg_.page_size;

        // NVFP4 payload: 0.5 bytes per element (4 bits)
        r.nvfp4_payload_bytes = (allocated_hist_tokens * kv_factors) / 2ULL;
        // FP8 E4M3 micro-scales: 1 byte per 16 elements
        r.micro_scale_fp8_bytes = (allocated_hist_tokens * kv_factors) / NVFP4_MICRO_BLOCK_SIZE;
        // Per-tensor FP32 global scales (K & V) + block table int32 entries
        r.global_scale_bytes = 2ULL * sizeof(float);
        r.block_table_bytes  = num_pages * sizeof(int32_t);

        r.total_hybrid_bytes = r.sink_fp8_bytes + r.window_fp8_bytes +
                               r.nvfp4_payload_bytes + r.micro_scale_fp8_bytes +
                               r.global_scale_bytes + r.block_table_bytes;
        r.compression_ratio = (r.total_hybrid_bytes > 0)
            ? (static_cast<double>(r.bf16_bytes) / static_cast<double>(r.total_hybrid_bytes))
            : 1.0;
        return r;
    }

private:
    static void quantize_vec_fp8_e4m3(const float* src, int d, uint8_t* dst, float& out_scale) noexcept {
        float amax = 0.0f;
        for (int i = 0; i < d; ++i) {
            amax = std::max(amax, std::fabs(src[i]));
        }
        float scale = (amax > 1e-12f) ? (amax / FP8_E4M3_MAX) : 1.0f;
        float inv_scale = 1.0f / scale;
        out_scale = scale;
        for (int i = 0; i < d; ++i) {
            dst[i] = fp32_to_fp8_e4m3_rne(src[i] * inv_scale);
        }
    }

    HybridKVConfig cfg_;
    int uint4_per_head_       = 4;
    int microblocks_per_head_ = 8;

    std::vector<uint4>   k_nvfp4_pool_;
    std::vector<uint4>   v_nvfp4_pool_;
    std::vector<uint8_t> k_micro_scales_;
    std::vector<uint8_t> v_micro_scales_;
    std::vector<int32_t> free_blocks_;

    std::vector<int>                  seq_lengths_;
    std::vector<int>                  hist_lengths_;
    std::vector<std::vector<int32_t>> block_tables_;

    std::vector<uint8_t> k_sink_fp8_;
    std::vector<uint8_t> v_sink_fp8_;
    std::vector<float>   k_sink_scale_;
    std::vector<float>   v_sink_scale_;

    std::vector<uint8_t> k_win_fp8_;
    std::vector<uint8_t> v_win_fp8_;
    std::vector<float>   k_win_scale_;
    std::vector<float>   v_win_scale_;

    std::vector<float>   k_global_scale_;
    std::vector<float>   v_global_scale_;
};

} // namespace turbokv
