#include "dynamoecon/nixl_transfer_engine.h"
#include <algorithm>
#include <cmath>

namespace dynamoecon {

LayerPipelinedNIXLEngine::LayerPipelinedNIXLEngine(
    int num_layers,
    double base_link_bw_gbps,
    double descriptor_overhead_ms)
    : num_layers_(std::max(1, num_layers))
    , base_link_bw_gbps_(std::max(1.0, base_link_bw_gbps))
    , descriptor_overhead_ms_(std::max(0.001, descriptor_overhead_ms))
{}

double LayerPipelinedNIXLEngine::estimate_exposed_nixl_stall_ms(
    int uncached_tokens_on_decode,
    uint64_t kv_bytes_per_token_all_layers,
    double prefill_compute_ms,
    double prefill_queue_ms,
    double decode_queue_ms,
    double congestion_factor) const noexcept
{
    if (uncached_tokens_on_decode <= 0) {
        return 0.0;
    }

    double safe_cong = std::max(1.0, congestion_factor);
    double eff_bw_bytes_per_ms = (base_link_bw_gbps_ / safe_cong) * 1e6; // 1 GB/s = 10^6 bytes/ms
    double total_bytes = static_cast<double>(uncached_tokens_on_decode) * static_cast<double>(kv_bytes_per_token_all_layers);

    double layer_compute_ms  = std::max(0.02, prefill_compute_ms) / static_cast<double>(num_layers_);
    double layer_transfer_ms = descriptor_overhead_ms_ + (total_bytes / static_cast<double>(num_layers_)) / eff_bw_bytes_per_ms;

    // Layer-pipelined completion time:
    //   - If layer_transfer_ms <= layer_compute_ms, each layer finishes streaming right after its compute
    //     plus 1 single layer_transfer_ms tail!
    //   - If layer_transfer_ms > layer_compute_ms (congested interconnect!), the NIC queue builds up across
    //     all num_layers_ layers: first layer starts at prefill_queue + layer_compute, followed by N * layer_transfer_ms!
    double t_prefill_start = prefill_queue_ms;
    double t_transfer_end  = 0.0;
    if (layer_transfer_ms <= layer_compute_ms) {
        t_transfer_end = t_prefill_start + prefill_compute_ms + layer_transfer_ms;
    } else {
        t_transfer_end = t_prefill_start + layer_compute_ms + static_cast<double>(num_layers_) * layer_transfer_ms;
    }

    // Consumer ready is when both Prefill finishes (or Decode worker queue clears, whichever is later — matching max() in FR-2.2!)
    double t_consumer_ready = std::max(prefill_queue_ms + prefill_compute_ms, decode_queue_ms);
    return std::max(0.0, t_transfer_end - t_consumer_ready);
}

NIXLTransferSummary LayerPipelinedNIXLEngine::simulate_pipelined_transfer(
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
    bool record_all_layers) const
{
    NIXLTransferSummary sum{};
    sum.request_id           = request_id;
    sum.src_prefill_worker   = src_prefill_worker;
    sum.dst_decode_worker    = dst_decode_worker;
    sum.num_layers           = num_layers_;
    sum.congestion_factor    = std::max(1.0, congestion_factor);
    sum.link_bandwidth_gbps  = base_link_bw_gbps_ / sum.congestion_factor;
    sum.total_kv_bytes       = static_cast<uint64_t>(std::max(0, tokens_transferred)) * kv_bytes_per_token_all_layers;

    double eff_bw_bytes_per_ms = sum.link_bandwidth_gbps * 1e6;
    double layer_bytes         = static_cast<double>(sum.total_kv_bytes) / static_cast<double>(num_layers_);
    double layer_compute_ms    = std::max(0.02, prefill_compute_ms) / static_cast<double>(num_layers_);
    double layer_wire_ms       = (tokens_transferred > 0)
        ? (descriptor_overhead_ms_ + layer_bytes / eff_bw_bytes_per_ms)
        : 0.0;

    double t_prefill_begin     = t_request_arrival_ms + prefill_queue_ms;
    double t_decode_slot_ready = t_request_arrival_ms + std::max(prefill_queue_ms + prefill_compute_ms, decode_queue_ms);
    double nic_free_at_ms      = t_prefill_begin;

    if (record_all_layers) {
        sum.layers.reserve(num_layers_);
    }

    double first_tx_start = 0.0;
    double last_prod_rdy  = t_prefill_begin;
    double last_tx_end    = t_prefill_begin;

    for (int l = 0; l < num_layers_; ++l) {
        double prod_ready = t_prefill_begin + static_cast<double>(l + 1) * layer_compute_ms;
        double tx_start   = std::max(prod_ready, nic_free_at_ms);
        double tx_end     = tx_start + layer_wire_ms;
        nic_free_at_ms    = tx_end;

        if (l == 0) first_tx_start = tx_start;
        last_prod_rdy = prod_ready;
        last_tx_end   = tx_end;

        if (record_all_layers) {
            LayerTransferRecord rec{};
            rec.layer_id            = l;
            rec.t_producer_ready_ms = prod_ready;
            rec.t_transfer_start_ms = tx_start;
            rec.t_transfer_end_ms   = tx_end;
            rec.t_consumer_ready_ms = t_decode_slot_ready;
            rec.exposed_stall_ms    = std::max(0.0, tx_end - t_decode_slot_ready);
            sum.layers.push_back(rec);
        }
    }

    sum.t_producer_ready_ms = last_prod_rdy;
    sum.t_transfer_start_ms = first_tx_start;
    sum.t_transfer_end_ms   = last_tx_end;
    sum.t_consumer_ready_ms = t_decode_slot_ready;
    sum.exposed_stall_ms    = std::max(0.0, last_tx_end - t_decode_slot_ready);

    double serial_wire_ms = static_cast<double>(num_layers_) * layer_wire_ms;
    if (serial_wire_ms > 1e-6) {
        sum.unoverlapped_pct       = std::clamp((sum.exposed_stall_ms / serial_wire_ms) * 100.0, 0.0, 100.0);
        sum.overlap_efficiency_pct = 100.0 - sum.unoverlapped_pct;
    } else {
        sum.unoverlapped_pct       = 0.0;
        sum.overlap_efficiency_pct = 100.0;
    }

    return sum;
}

} // namespace dynamoecon
