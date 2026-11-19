#pragma once

#include "dynamoecon/nixl_transfer_engine.h"
#include "dynamoecon/radix_kv_indexer.h"
#include <atomic>
#include <cstdint>
#include <vector>

namespace dynamoecon {

// ==============================================================================
// FR-2.2: Crossover-Aware Disaggregated Scheduler (Queueing + Transport Model)
//
// Solves the Disaggregation Crossover Equation across Prefill worker i and Decode worker j:
//   T_total(i, j) = max(T_queue_P(i) + T_prefill(L_p - L_m(i)), T_queue_D(j))
//                   + T_NIXL_stall(L_p) + T_decode_first(j)
//
// Dynamic Mode Switching:
//   Automatically bypasses disaggregation (executing in-place AGGREGATED_INPLACE)
//   when L_p < L_threshold OR when NIXL interconnect congestion makes local
//   recomputation faster than remote KV transfer!
// ==============================================================================

enum class ExecutionMode : int {
    DISAGGREGATED_NIXL = 0,
    AGGREGATED_INPLACE = 1
};

enum class RoutingPolicy : int {
    CROSSOVER_ADAPTIVE        = 0,
    STATIC_ROUND_ROBIN_DISAGG = 1
};

struct WorkerQueueState {
    int    worker_id        = 0;
    bool   is_prefill_pool  = true;   // true = Prefill specialist, false = Decode / Unified worker
    bool   healthy          = true;
    double queue_wait_ms    = 0.0;
    int    active_sessions  = 0;
    double prefill_tok_per_ms = 65.0; // e.g., 65,000 tok/s prefill throughput on H100/B200
    double decode_first_step_ms = 8.5;
};

struct RoutingDecision {
    ExecutionMode mode               = ExecutionMode::DISAGGREGATED_NIXL;
    int           prefill_worker_id  = 0;
    int           decode_worker_id   = 0;
    int           matched_prefix_len = 0;
    double        predicted_ttft_ms  = 0.0;
    double        predicted_nixl_stall_ms = 0.0;
    double        best_disagg_ttft_ms     = 0.0;
    double        best_agg_ttft_ms        = 0.0;
    double        routing_latency_us      = 0.0;
};

class DYNAMOECON_API CrossoverDisaggregatedScheduler {
public:
    CrossoverDisaggregatedScheduler(
        ConcurrentRadixKVIndexer& indexer,
        const LayerPipelinedNIXLEngine& nixl_engine,
        int num_prefill_workers = 16,
        int num_decode_workers  = 16,
        int l_threshold_tokens  = 512,
        uint64_t kv_bytes_per_token = 18432); // TurboKV-FP4 32-layer KV bytes/token (~18 KB vs 65 KB BF16!)

    RoutingDecision schedule_request(
        uint64_t request_id,
        int prompt_len,
        const uint64_t* block_hashes,
        int num_blocks,
        const std::vector<WorkerQueueState>& workers,
        double nixl_congestion_factor,
        double now_s,
        RoutingPolicy policy = RoutingPolicy::CROSSOVER_ADAPTIVE);

    void set_l_threshold(int tokens) noexcept { l_threshold_tokens_ = tokens; }
    int  l_threshold() const noexcept { return l_threshold_tokens_; }
    void set_kv_bytes_per_token(uint64_t b) noexcept { kv_bytes_per_token_ = b; }

private:
    ConcurrentRadixKVIndexer&       indexer_;
    const LayerPipelinedNIXLEngine& nixl_engine_;
    int                             num_prefill_workers_;
    int                             num_decode_workers_;
    int                             l_threshold_tokens_;
    uint64_t                        kv_bytes_per_token_;
    std::atomic<uint64_t>           rr_counter_{0};
};

} // namespace dynamoecon
