#pragma once

#include "dynamoecon/telemetry_hal.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace dynamoecon {

// ==============================================================================
// FR-2.4: Multi-Tier SLA Monetization & Hardware Power Telemetry Daemon
// (`econ_telemetryd`)
//
// Polls GPU telemetry at 50ms intervals via TelemetryHAL and classifies traffic
// into configurable monetization SLAs:
//   - Tier 1 (Interactive Reasoning): TTFT <= 150ms, ITL <= 20ms
//   - Tier 2 (Agentic Tool-Use):      TTFT <= 400ms, ITL <= 40ms
//   - Tier 3 (Batch Synthesis):       Throughput-optimized, best-effort latency
//
// Live Metrics Exported:
//   - SLA Goodput (%)
//   - Tokens per Watt (tok/s/W = tok/J)
//   - Effective Prefix Reuse (%)
//   - Cost per Million Goodput Tokens ($ / 1M)
// ==============================================================================

enum class SLATier : int {
    TIER1_INTERACTIVE_REASONING = 1, // TTFT <= 150ms, ITL <= 20ms
    TIER2_AGENTIC_TOOL_USE      = 2, // TTFT <= 400ms, ITL <= 40ms
    TIER3_BATCH_SYNTHESIS       = 3  // Best-effort throughput
};

struct SLATierThresholds {
    double tier1_max_ttft_ms = 150.0;
    double tier1_max_itl_ms  = 20.0;
    double tier2_max_ttft_ms = 400.0;
    double tier2_max_itl_ms  = 40.0;
    double electricity_usd_per_kwh = 0.12;
    double pue_factor              = 1.15;
};

struct CompletedRequestRecord {
    uint64_t request_id       = 0;
    SLATier  tier             = SLATier::TIER2_AGENTIC_TOOL_USE;
    int      prompt_tokens    = 0;
    int      cached_tokens    = 0;
    int      output_tokens    = 0;
    double   ttft_ms          = 0.0;
    double   itl_ms           = 0.0;
    double   nixl_stall_ms    = 0.0;
    bool     disaggregated    = true;
};

struct TokconomicsKPIReport {
    uint64_t total_requests              = 0;
    uint64_t sla_compliant_requests      = 0;
    uint64_t total_output_tokens         = 0;
    uint64_t goodput_output_tokens       = 0;

    double   sla_goodput_pct             = 0.0;
    double   tier1_goodput_pct           = 0.0;
    double   tier2_goodput_pct           = 0.0;
    double   tier3_goodput_pct           = 100.0;

    double   avg_ttft_ms                 = 0.0;
    double   p99_ttft_ms                 = 0.0;
    double   avg_itl_ms                  = 0.0;
    double   avg_exposed_nixl_stall_ms   = 0.0;

    double   cluster_power_watts         = 0.0;
    double   total_energy_joules         = 0.0;
    double   tokens_per_watt             = 0.0; // tokens / Joule = (tok/s) / Watt
    double   effective_prefix_reuse_pct  = 0.0;
    double   cost_per_1m_goodput_usd     = 0.0;
    uint64_t daemon_poll_ticks_50ms      = 0;
};

class DYNAMOECON_API EconTelemetryDaemon {
public:
    EconTelemetryDaemon(
        const TelemetryHAL& hal,
        int num_workers = 32,
        SLATierThresholds thresholds = {});
    ~EconTelemetryDaemon();

    // Evaluates whether a request satisfies its tier SLA bounds (FR-2.4)
    bool meets_sla(SLATier tier, double ttft_ms, double itl_ms) const noexcept;

    void record_request(const CompletedRequestRecord& rec);

    // Updates simulated/hardware worker activity & executes one 50ms telemetry poll sample
    void poll_cluster_once(
        const std::vector<double>& worker_u_sm,
        const std::vector<double>& worker_b_hbm,
        double interval_s = 0.050);

    // Starts/stops real-time 50ms C++20 background polling daemon thread
    void start_background_50ms_polling();
    void stop_background_polling();

    TokconomicsKPIReport snapshot_kpis(double simulated_duration_s = 0.0) const;
    std::vector<WorkerTelemetrySample> latest_worker_samples() const;
    std::string export_prometheus_metrics(double simulated_duration_s = 0.0) const;

private:
    const TelemetryHAL& hal_;
    int                 num_workers_;
    SLATierThresholds   thresholds_;

    mutable std::mutex  mu_;
    std::vector<WorkerTelemetrySample> latest_samples_;
    std::vector<double> ttft_samples_ms_;

    uint64_t total_reqs_         = 0;
    uint64_t good_reqs_          = 0;
    uint64_t tier_total_[4]      = {0, 0, 0, 0};
    uint64_t tier_good_[4]       = {0, 0, 0, 0};
    uint64_t total_prompt_tok_   = 0;
    uint64_t total_cached_tok_   = 0;
    uint64_t total_output_tok_   = 0;
    uint64_t goodput_output_tok_ = 0;
    double   sum_ttft_ms_        = 0.0;
    double   sum_itl_ms_         = 0.0;
    double   sum_nixl_stall_ms_  = 0.0;

    double   cluster_power_watts_  = 0.0;
    double   total_energy_joules_  = 0.0;
    double   accumulated_time_s_   = 0.0;
    uint64_t poll_ticks_           = 0;

    std::atomic<bool> running_{false};
    std::thread       poll_thread_;
};

} // namespace dynamoecon
