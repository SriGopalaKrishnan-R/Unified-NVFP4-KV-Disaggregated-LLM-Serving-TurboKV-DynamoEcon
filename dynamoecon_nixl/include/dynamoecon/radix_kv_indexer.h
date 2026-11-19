#pragma once

#include "dynamoecon/kv_events.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace dynamoecon {

// ==============================================================================
// FR-2.1: Lock-Striped Concurrent Radix-Tree KV Indexer with Recompute-Cost Eviction
//
// Eviction Watermark: 90% HBM capacity per worker.
// Tokconomics Retention Score:
//   S_retain(b) = (FLOPs_recompute(b) * P_reuse(b)) / (Size_bytes(b) * (1.0 + delta_t_idle))
// ==============================================================================

enum class EvictionPolicy : int {
    TOKCONOMICS_S_RETAIN = 0,
    STANDARD_LRU         = 1
};

struct RadixLookupResult {
    int      best_prefill_worker = -1;
    int      matched_tokens_best = 0;
    // Per-worker matched prefix tokens across up to 64 workers
    std::array<int, 64> worker_matched_tokens{};
};

struct EvictionStats {
    uint64_t total_evictions        = 0;
    uint64_t evicted_bytes          = 0;
    double   avg_evicted_s_retain   = 0.0;
    uint64_t total_lookups          = 0;
    uint64_t total_matched_tokens   = 0;
    uint64_t total_requested_tokens = 0;
};

class DYNAMOECON_API ConcurrentRadixKVIndexer {
public:
    static constexpr size_t kNumStripes = 64;

    ConcurrentRadixKVIndexer(
        int num_workers,
        uint64_t worker_hbm_capacity_bytes,
        int block_tokens = 16,
        EvictionPolicy policy = EvictionPolicy::TOKCONOMICS_S_RETAIN,
        KVEventBus* event_bus = nullptr);

    // Computes exact Tokconomics Retention Score S_retain(b) (FR-2.1)
    static double compute_retention_score(
        double flops_recompute,
        double p_reuse,
        uint64_t size_bytes,
        double delta_t_idle_s) noexcept;

    // Inserts or refreshes a prefix chain of `block_hashes` on `worker_id`.
    // Automatically triggers watermark eviction when worker HBM >= 90% capacity!
    void insert_prefix_blocks(
        int worker_id,
        const uint64_t* block_hashes,
        int num_blocks,
        uint64_t bytes_per_block,
        double base_flops_per_block,
        double now_s);

    // Concurrent lock-striped prefix lookup across candidate worker mask
    RadixLookupResult match_prefix(
        const uint64_t* block_hashes,
        int num_blocks,
        uint64_t candidate_worker_mask,
        double now_s,
        bool update_ema_on_hit = true);

    // Re-anchors KV blocks from a failed worker to `backup_worker_id` (used by NFR-2.3 failover)
    int migrate_worker_blocks(int dead_worker_id, int backup_worker_id, double now_s);

    uint64_t worker_used_bytes(int worker_id) const noexcept;
    double   worker_hbm_utilization(int worker_id) const noexcept;
    uint64_t worker_capacity_bytes() const noexcept { return worker_capacity_bytes_; }
    int      block_tokens() const noexcept { return block_tokens_; }
    EvictionStats stats() const noexcept;
    void set_eviction_policy(EvictionPolicy p) noexcept { policy_ = p; }
    EvictionPolicy eviction_policy() const noexcept { return policy_; }

private:
    struct RadixNode {
        uint64_t block_hash        = 0;
        uint64_t parent_hash       = 0;
        uint64_t worker_mask       = 0ULL; // Bit i set iff worker i holds this KV block
        uint64_t size_bytes        = 0;
        double   flops_recompute   = 0.0;  // Cumulative / positional prefill FLOPs
        double   p_reuse_ema       = 0.35; // Exponential moving average hit probability
        double   last_access_s     = 0.0;
        int      depth_blocks      = 1;
        std::unordered_map<uint64_t, std::unique_ptr<RadixNode>> children;
    };

    struct Stripe {
        mutable std::shared_mutex mu;
        std::unordered_map<uint64_t, std::unique_ptr<RadixNode>> roots;
        // Flat index within stripe for O(N_stripe_worker) fast watermark eviction & failover scan
        std::unordered_map<uint64_t, RadixNode*> by_hash;
    };

    size_t stripe_index(uint64_t root_hash) const noexcept {
        // Fibonacci / SplitMix64 hash mixing across 64 stripes
        uint64_t z = root_hash + 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return static_cast<size_t>((z ^ (z >> 31)) & (kNumStripes - 1));
    }

    void evict_if_above_watermark(int worker_id, double now_s);

    int num_workers_;
    uint64_t worker_capacity_bytes_;
    uint64_t watermark_90pct_bytes_;
    int block_tokens_;
    EvictionPolicy policy_;
    KVEventBus* event_bus_;

    std::array<Stripe, kNumStripes> stripes_;
    std::array<std::atomic<uint64_t>, 64> worker_bytes_{};

    std::atomic<uint64_t> total_evictions_{0};
    std::atomic<uint64_t> evicted_bytes_{0};
    std::atomic<uint64_t> total_lookups_{0};
    std::atomic<uint64_t> total_matched_tokens_{0};
    std::atomic<uint64_t> total_requested_tokens_{0};
    mutable std::mutex    evict_metric_mu_;
    double                sum_evicted_s_retain_ = 0.0;
};

} // namespace dynamoecon
