#include "dynamoecon/econ_telemetryd.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace dynamoecon {

EconTelemetryDaemon::EconTelemetryDaemon(
    const TelemetryHAL& hal,
    int num_workers,
    SLATierThresholds thresholds)
    : hal_(hal)
    , num_workers_(std::max(1, num_workers))
    , thresholds_(thresholds)
{
    latest_samples_.resize(num_workers_);
    for (int w = 0; w < num_workers_; ++w) {
        latest_samples_[w] = hal_.poll_worker(w, 0.05, 0.05);
    }
}

EconTelemetryDaemon::~EconTelemetryDaemon() {
    stop_background_polling();
}

bool EconTelemetryDaemon::meets_sla(SLATier tier, double ttft_ms, double itl_ms) const noexcept {
    switch (tier) {
        case SLATier::TIER1_INTERACTIVE_REASONING:
            return (ttft_ms <= thresholds_.tier1_max_ttft_ms) && (itl_ms <= thresholds_.tier1_max_itl_ms);
        case SLATier::TIER2_AGENTIC_TOOL_USE:
            return (ttft_ms <= thresholds_.tier2_max_ttft_ms) && (itl_ms <= thresholds_.tier2_max_itl_ms);
        case SLATier::TIER3_BATCH_SYNTHESIS:
        default:
            return (ttft_ms <= 5000.0) && (itl_ms <= 200.0);
    }
}

void EconTelemetryDaemon::record_request(const CompletedRequestRecord& rec) {
    bool ok = meets_sla(rec.tier, rec.ttft_ms, rec.itl_ms);
    int tidx = std::clamp(static_cast<int>(rec.tier), 1, 3);

    std::lock_guard<std::mutex> lk(mu_);
    ++total_reqs_;
    ++tier_total_[tidx];
    if (ok) {
        ++good_reqs_;
        ++tier_good_[tidx];
        goodput_output_tok_ += static_cast<uint64_t>(std::max(0, rec.output_tokens));
    }
    total_prompt_tok_  += static_cast<uint64_t>(std::max(0, rec.prompt_tokens));
    total_cached_tok_  += static_cast<uint64_t>(std::clamp(rec.cached_tokens, 0, rec.prompt_tokens));
    total_output_tok_  += static_cast<uint64_t>(std::max(0, rec.output_tokens));
    sum_ttft_ms_       += rec.ttft_ms;
    sum_itl_ms_        += rec.itl_ms;
    sum_nixl_stall_ms_ += rec.nixl_stall_ms;
    ttft_samples_ms_.push_back(rec.ttft_ms);
}

void EconTelemetryDaemon::poll_cluster_once(
    const std::vector<double>& worker_u_sm,
    const std::vector<double>& worker_b_hbm,
    double interval_s)
{
    std::vector<WorkerTelemetrySample> samples(num_workers_);
    double total_w = 0.0;
    for (int w = 0; w < num_workers_; ++w) {
        double usm  = (w < static_cast<int>(worker_u_sm.size())) ? worker_u_sm[w] : 0.15;
        double bhbm = (w < static_cast<int>(worker_b_hbm.size())) ? worker_b_hbm[w] : 0.20;
        samples[w] = hal_.poll_worker(w, usm, bhbm, bhbm * 180.0, bhbm * 180.0);
        total_w   += samples[w].board_power_watts;
    }

    std::lock_guard<std::mutex> lk(mu_);
    latest_samples_       = std::move(samples);
    cluster_power_watts_  = total_w;
    total_energy_joules_ += total_w * std::max(0.001, interval_s);
    accumulated_time_s_  += std::max(0.001, interval_s);
    ++poll_ticks_;
}

void EconTelemetryDaemon::start_background_50ms_polling() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;

    poll_thread_ = std::thread([this]() {
        std::vector<double> u_sm(num_workers_, 0.42);
        std::vector<double> b_hbm(num_workers_, 0.68);
        while (running_.load(std::memory_order_relaxed)) {
            poll_cluster_once(u_sm, b_hbm, 0.050);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });
}

void EconTelemetryDaemon::stop_background_polling() {
    bool expected = true;
    if (running_.compare_exchange_strong(expected, false)) {
        if (poll_thread_.joinable()) {
            poll_thread_.join();
        }
    }
}

TokconomicsKPIReport EconTelemetryDaemon::snapshot_kpis(double simulated_duration_s) const {
    std::lock_guard<std::mutex> lk(mu_);
    TokconomicsKPIReport r{};
    r.total_requests         = total_reqs_;
    r.sla_compliant_requests = good_reqs_;
    r.total_output_tokens    = total_output_tok_;
    r.goodput_output_tokens  = goodput_output_tok_;

    r.sla_goodput_pct = (total_reqs_ > 0)
        ? (100.0 * static_cast<double>(good_reqs_) / static_cast<double>(total_reqs_))
        : 0.0;
    r.tier1_goodput_pct = (tier_total_[1] > 0) ? (100.0 * tier_good_[1] / tier_total_[1]) : 100.0;
    r.tier2_goodput_pct = (tier_total_[2] > 0) ? (100.0 * tier_good_[2] / tier_total_[2]) : 100.0;
    r.tier3_goodput_pct = (tier_total_[3] > 0) ? (100.0 * tier_good_[3] / tier_total_[3]) : 100.0;

    if (total_reqs_ > 0) {
        r.avg_ttft_ms               = sum_ttft_ms_ / static_cast<double>(total_reqs_);
        r.avg_itl_ms                = sum_itl_ms_ / static_cast<double>(total_reqs_);
        r.avg_exposed_nixl_stall_ms = sum_nixl_stall_ms_ / static_cast<double>(total_reqs_);
    }

    if (!ttft_samples_ms_.empty()) {
        std::vector<double> sorted = ttft_samples_ms_;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = std::min(sorted.size() - 1, static_cast<size_t>(std::ceil(0.99 * sorted.size())) - 1);
        r.p99_ttft_ms = sorted[idx];
    }

    r.cluster_power_watts    = cluster_power_watts_;
    r.total_energy_joules    = total_energy_joules_;
    r.daemon_poll_ticks_50ms = poll_ticks_;

    double duration_s = (simulated_duration_s > 0.0) ? simulated_duration_s : std::max(0.050, accumulated_time_s_);
    double avg_power  = (total_energy_joules_ > 0.0) ? (total_energy_joules_ / std::max(0.050, accumulated_time_s_)) : std::max(1.0, cluster_power_watts_);
    double tok_per_s  = static_cast<double>(total_output_tok_) / duration_s;
    r.tokens_per_watt = tok_per_s / std::max(1.0, avg_power);

    r.effective_prefix_reuse_pct = (total_prompt_tok_ > 0)
        ? (100.0 * static_cast<double>(total_cached_tok_) / static_cast<double>(total_prompt_tok_))
        : 0.0;

    // Total Cluster Cost over `duration_s` = GPU Amortized Capex + PUE-adjusted Electricity Cost
    double capex_usd = static_cast<double>(num_workers_) * hal_.gpu_hourly_capex_usd() * (duration_s / 3600.0);
    double kwh_used  = (avg_power * duration_s) / 3.6e6;
    double power_usd = kwh_used * thresholds_.pue_factor * thresholds_.electricity_usd_per_kwh;
    double total_usd = capex_usd + power_usd;

    if (goodput_output_tok_ > 0) {
        r.cost_per_1m_goodput_usd = (total_usd / static_cast<double>(goodput_output_tok_)) * 1e6;
    } else {
        r.cost_per_1m_goodput_usd = 999.0;
    }

    return r;
}

std::vector<WorkerTelemetrySample> EconTelemetryDaemon::latest_worker_samples() const {
    std::lock_guard<std::mutex> lk(mu_);
    return latest_samples_;
}

std::string EconTelemetryDaemon::export_prometheus_metrics(double simulated_duration_s) const {
    auto kpi = snapshot_kpis(simulated_duration_s);
    std::ostringstream os;
    os << std::fixed << std::setprecision(4);
    os << "# HELP dynamoecon_sla_goodput_pct Percentage of requests meeting Tier SLA bounds\n"
       << "# TYPE dynamoecon_sla_goodput_pct gauge\n"
       << "dynamoecon_sla_goodput_pct " << kpi.sla_goodput_pct << "\n"
       << "dynamoecon_tier_goodput_pct{tier=\"1\"} " << kpi.tier1_goodput_pct << "\n"
       << "dynamoecon_tier_goodput_pct{tier=\"2\"} " << kpi.tier2_goodput_pct << "\n"
       << "dynamoecon_tier_goodput_pct{tier=\"3\"} " << kpi.tier3_goodput_pct << "\n"
       << "# HELP dynamoecon_tokens_per_watt Cluster energy efficiency in output tokens per Joule\n"
       << "# TYPE dynamoecon_tokens_per_watt gauge\n"
       << "dynamoecon_tokens_per_watt " << kpi.tokens_per_watt << "\n"
       << "# HELP dynamoecon_effective_prefix_reuse_pct Radix KV cache prefix token reuse ratio\n"
       << "# TYPE dynamoecon_effective_prefix_reuse_pct gauge\n"
       << "dynamoecon_effective_prefix_reuse_pct " << kpi.effective_prefix_reuse_pct << "\n"
       << "# HELP dynamoecon_cost_per_million_goodput_usd Monetization cost per 1M SLA-compliant tokens\n"
       << "# TYPE dynamoecon_cost_per_million_goodput_usd gauge\n"
       << "dynamoecon_cost_per_million_goodput_usd " << kpi.cost_per_1m_goodput_usd << "\n"
       << "# HELP dynamoecon_cluster_power_watts Instantaneous GPU cluster board power\n"
       << "# TYPE dynamoecon_cluster_power_watts gauge\n"
       << "dynamoecon_cluster_power_watts " << kpi.cluster_power_watts << "\n"
       << "# HELP dynamoecon_ttft_ms Time-to-First-Token summary\n"
       << "# TYPE dynamoecon_ttft_ms gauge\n"
       << "dynamoecon_ttft_ms{quantile=\"avg\"} " << kpi.avg_ttft_ms << "\n"
       << "dynamoecon_ttft_ms{quantile=\"p99\"} " << kpi.p99_ttft_ms << "\n";
    return os.str();
}

} // namespace dynamoecon
