#pragma once

#include <cstdint>
#include <vector>

#include "dynamoecon/kv_events.h"

namespace dynamoecon {

// ==============================================================================
// FR-2.3: Layer-Pipelined Asynchronous NIXL Transfer Engine
// Emulates NVIDIA NIXL non-blocking layer-by-layer KV block streaming over
// NVLink (900-1800 GB/s) or UCX/RDMA InfiniBand (50-100 GB/s).
//
// Tracks the 4 canonical disaggregated timestamps per layer & request:
//   1. t_producer_ready  (layer l KV block computed on Prefill worker)
//   2. t_transfer_start  (NIXL RDMA/NVLink descriptor posted)
//   3. t_transfer_end    (KV block landed in Decode worker HBM)
//   4. t_consumer_ready  (Decode worker slot available)
// Exposed Transport Stall = max(0.0, t_transfer_end - t_consumer_ready)
// ==============================================================================

struct LayerTransferRecord {
    int    layer_id             = 0;
    double t_producer_ready_ms  = 0.0;
    double t_transfer_start_ms  = 0.0;
    double t_transfer_end_ms    = 0.0;
    double t_consumer_ready_ms  = 0.0;
    double exposed_stall_ms     = 0.0;
};

struct NIXLTransferSummary {
    uint64_t request_id             = 0;
    int      src_prefill_worker     = 0;
    int      dst_decode_worker      = 0;
    int      num_layers             = 32;
    uint64_t total_kv_bytes         = 0;
    double   link_bandwidth_gbps    = 400.0; // Effective GB/s
    double   congestion_factor      = 1.0;   // >= 1.0 (1.0 = uncongested)

    // Canonical request-level boundary timestamps (final layer completion vs consumer ready)
    double   t_producer_ready_ms    = 0.0;
    double   t_transfer_start_ms    = 0.0;
    double   t_transfer_end_ms      = 0.0;
    double   t_consumer_ready_ms    = 0.0;
    double   exposed_stall_ms       = 0.0;
    double   unoverlapped_pct       = 0.0;
    double   overlap_efficiency_pct = 0.0;

    std::vector<LayerTransferRecord> layers;
};

class DYNAMOECON_API LayerPipelinedNIXLEngine {
public:
    LayerPipelinedNIXLEngine(
        int num_layers = 32,
        double base_link_bw_gbps = 100.0, // e.g., 100 GB/s dual-rail NDR400 RDMA or 900 GB/s NVLink
        double descriptor_overhead_ms = 0.035);

    // Analytical stall predictor used by FR-2.2 Crossover Scheduler in O(1) time (< 50 ns!)
    double estimate_exposed_nixl_stall_ms(
        int uncached_tokens_on_decode,
        uint64_t kv_bytes_per_token_all_layers,
        double prefill_compute_ms,
        double prefill_queue_ms,
        double decode_queue_ms,
        double congestion_factor) const noexcept;

    // Full layer-by-layer NIXL pipeline execution & timestamp trace logger (FR-2.3)
    NIXLTransferSummary simulate_pipelined_transfer(
        uint64_t request_id,
        int src_prefill_worker,
        int dst_decode_worker,
        int tokens_transferred,
        uint64_t kv_bytes_per_token_all_layers,
        double t_request_arrival_ms,
        double prefill_queue_ms,
        double prefill_compute_ms,
        double decode_queue_ms,
        double congestion_factor,
        bool record_all_layers = true) const;

    void set_base_link_bw_gbps(double bw_gbps) noexcept { base_link_bw_gbps_ = bw_gbps; }
    double base_link_bw_gbps() const noexcept { return base_link_bw_gbps_; }
    int num_layers() const noexcept { return num_layers_; }

private:
    int    num_layers_;
    double base_link_bw_gbps_;
    double descriptor_overhead_ms_;
};

} // namespace dynamoecon
