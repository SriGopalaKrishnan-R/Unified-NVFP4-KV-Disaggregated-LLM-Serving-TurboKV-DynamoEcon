#include "gtest_lite.h"
#include "dynamoecon/crossover_scheduler.h"
#include "dynamoecon/econ_telemetryd.h"
#include "dynamoecon/failover_manager.h"
#include "dynamoecon/kv_events.h"
#include "dynamoecon/nixl_transfer_engine.h"
#include "dynamoecon/radix_kv_indexer.h"
#include "dynamoecon/telemetry_hal.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace dynamoecon;

extern "C" {
struct DynamoEconBenchmarkSummary {
    double scheduler_p50_us;
    double scheduler_p99_us;
    double scheduler_throughput_rps;
    double adaptive_sla_goodput_pct;
    double adaptive_tier1_goodput_pct;
    double adaptive_tier2_goodput_pct;
    double adaptive_tier3_goodput_pct;
    double adaptive_avg_ttft_ms;
    double adaptive_p99_ttft_ms;
    double adaptive_avg_itl_ms;
    double adaptive_nixl_stall_ms;
    double adaptive_tokens_per_watt;
    double adaptive_prefix_reuse_pct;
    double adaptive_cost_per_1m_usd;
    double adaptive_cluster_watts;
    int    adaptive_aggregated_bypasses;
    int    adaptive_disaggregated_routes;
    double baseline_sla_goodput_pct;
    double baseline_avg_ttft_ms;
    double baseline_p99_ttft_ms;
    double baseline_avg_itl_ms;
    double baseline_nixl_stall_ms;
    double baseline_tokens_per_watt;
    double baseline_prefix_reuse_pct;
    double baseline_cost_per_1m_usd;
    double goodput_improvement_pct;
    double cost_reduction_pct;
    double failover_latency_ms;
    int    failover_sessions_reanchored;
    int    failover_sessions_lost;
    int    failover_kv_blocks_migrated;
};
int dynamoecon_run_agentic_benchmark(
    int num_requests, int num_workers, int hw_profile_code,
    double arrival_rate_scale, double nixl_congestion, DynamoEconBenchmarkSummary* out);
}

// 1. FR-2.1 Concurrent Lock-Striped Radix Tree & S_retain Eviction @ 90% Watermark
TEST(DynamoEconNIXL, FR21_RadixTreeRetentionScoreEvictionAndEvents) {
    KVEventBus bus(256);
    // Capacity = 100 KB -> 90% watermark = 90 KB, block size = 10 KB
    ConcurrentRadixKVIndexer idx(8, 100000ULL, 16, EvictionPolicy::TOKCONOMICS_S_RETAIN, &bus);

    // Exact S_retain formula verification: (1e10 * 0.8) / (10000 * (1 + 3.0)) = 200,000.0
    double s_ret = ConcurrentRadixKVIndexer::compute_retention_score(1e10, 0.8, 10000ULL, 3.0);
    EXPECT_NEAR(s_ret, 200000.0, 1e-3);

    // Insert a high-FLOP, frequently reused agentic prefix (5 blocks = 50 KB) at t=1.0s, hit 4 times up to t=9.5s
    uint64_t high_val_prefix[5] = {101, 102, 103, 104, 105};
    idx.insert_prefix_blocks(0, high_val_prefix, 5, 10000ULL, 5.0e10, 1.0);
    for (int hit = 0; hit < 4; ++hit) {
        auto m = idx.match_prefix(high_val_prefix, 5, 1ULL, 2.0 + hit * 2.0, true);
        EXPECT_EQ(m.matched_tokens_best, 80);
    }

    // Insert low-FLOP ephemeral one-shot blocks (5 blocks = 50 KB -> total 100 KB >= 90% watermark!)
    for (int e = 0; e < 5; ++e) {
        uint64_t eph = 9000ULL + static_cast<uint64_t>(e);
        idx.insert_prefix_blocks(0, &eph, 1, 10000ULL, 1.0e8, 2.0);
    }

    // Watermark eviction MUST have fired, evicting ephemeral blocks while preserving high-value prefix!
    auto st = idx.stats();
    EXPECT_GT(st.total_evictions, 0ULL);
    EXPECT_LE(idx.worker_hbm_utilization(0), 0.90);

    auto check_high = idx.match_prefix(high_val_prefix, 5, 1ULL, 10.0, false);
    EXPECT_EQ(check_high.matched_tokens_best, 80); // High-S_retain prefix completely intact!
    EXPECT_GT(bus.total_published(), 5ULL);
}

// 2. FR-2.2 Crossover-Aware Disaggregated Scheduler & Dynamic Aggregated Mode Bypass
TEST(DynamoEconNIXL, FR22_CrossoverEquationAndDynamicModeSwitching) {
    ConcurrentRadixKVIndexer idx(32, 64ULL * 1024 * 1024 * 1024, 16);
    LayerPipelinedNIXLEngine nixl(32, 120.0);
    CrossoverDisaggregatedScheduler sched(idx, nixl, 16, 16, 512, 18432ULL);

    std::vector<WorkerQueueState> workers(32);
    for (int i = 0; i < 32; ++i) {
        workers[i].worker_id       = i;
        workers[i].is_prefill_pool = (i < 16);
        workers[i].healthy         = true;
        workers[i].queue_wait_ms   = 5.0;
    }

    uint64_t hashes[256];
    for (int b = 0; b < 256; ++b) hashes[b] = 0x5500ULL + static_cast<uint64_t>(b);

    // Case A: Short prompt L_p = 256 < L_threshold (512) -> bypasses disaggregation automatically!
    auto short_dec = sched.schedule_request(1, 256, hashes, 16, workers, 1.0, 1.0);
    EXPECT_EQ(short_dec.mode, ExecutionMode::AGGREGATED_INPLACE);
    EXPECT_EQ(short_dec.prefill_worker_id, short_dec.decode_worker_id);

    // Case B: Long prompt L_p = 4096 on healthy uncongested link -> selects Disaggregated NIXL!
    // Also verify queue-depth vs prefix-hit trade-off: worker 0 has prefix hit, but massive 900ms queue!
    idx.insert_prefix_blocks(0, hashes, 192, 294912ULL, 9e9, 1.0);
    workers[0].queue_wait_ms = 900.0; // Severe queue stall on highest cache-hit worker!
    auto long_dec = sched.schedule_request(2, 4096, hashes, 256, workers, 1.0, 2.0);
    EXPECT_EQ(long_dec.mode, ExecutionMode::DISAGGREGATED_NIXL);
    EXPECT_TRUE(long_dec.prefill_worker_id != 0); // Avoids congested cache-hit trap!

    // Case C: Severe NIXL interconnect congestion (congestion_factor = 140x -> <1 GB/s effective link)
    // makes remote KV streaming stall exceed local compute gap -> dynamically switches to In-Place Aggregated!
    auto cong_dec = sched.schedule_request(3, 4096, hashes, 256, workers, 140.0, 3.0);
    EXPECT_EQ(cong_dec.mode, ExecutionMode::AGGREGATED_INPLACE);
}

// 3. FR-2.3 Layer-Pipelined NIXL 4 Canonical Timestamps & Exposed Stall Calculation
TEST(DynamoEconNIXL, FR23_LayerPipelinedNIXLTimestampsAndOverlap) {
    LayerPipelinedNIXLEngine nixl(32, 100.0, 0.03);

    // Uncongested transfer: high compute-communication overlap
    auto fast = nixl.simulate_pipelined_transfer(10, 1, 18, 2048, 18432ULL, 0.0, 4.0, 28.0, 10.0, 1.0, true);
    ASSERT_EQ(static_cast<int>(fast.layers.size()), 32);
    for (const auto& lr : fast.layers) {
        EXPECT_LE(lr.t_producer_ready_ms, lr.t_transfer_start_ms);
        EXPECT_LT(lr.t_transfer_start_ms, lr.t_transfer_end_ms);
    }
    double expected_stall = std::max(0.0, fast.t_transfer_end_ms - fast.t_consumer_ready_ms);
    EXPECT_NEAR(fast.exposed_stall_ms, expected_stall, 1e-6);
    EXPECT_GT(fast.overlap_efficiency_pct, 80.0);

    // Congested transfer: exposed stall grows cleanly when t_transfer_end > t_consumer_ready
    auto slow = nixl.simulate_pipelined_transfer(11, 1, 18, 4096, 18432ULL, 0.0, 4.0, 28.0, 10.0, 12.0, true);
    EXPECT_GT(slow.exposed_stall_ms, fast.exposed_stall_ms);
    EXPECT_NEAR(slow.exposed_stall_ms, std::max(0.0, slow.t_transfer_end_ms - slow.t_consumer_ready_ms), 1e-6);
}

// 4. FR-2.4 TelemetryHAL Physics Model & 50ms SLA Monetization Daemon
TEST(DynamoEconNIXL, FR24_TelemetryHALAndMultiTierSLADaemon) {
    TelemetryHAL h100(HardwareProfile::H100_SXM5_700W, false);
    EXPECT_NEAR(h100.compute_physics_power_watts(0.0, 0.0), 70.0, 1e-5);
    EXPECT_NEAR(h100.compute_physics_power_watts(1.0, 1.0), 700.0, 1e-5);

    TelemetryHAL b200(HardwareProfile::B200_SXM6_1000W, false);
    EXPECT_NEAR(b200.compute_physics_power_watts(0.0, 0.0), 120.0, 1e-5);
    EXPECT_NEAR(b200.compute_physics_power_watts(1.0, 1.0), 1000.0, 1e-5);

    EconTelemetryDaemon daemon(b200, 8);
    // Tier 1: <=150ms TTFT, <=20ms ITL
    EXPECT_TRUE(daemon.meets_sla(SLATier::TIER1_INTERACTIVE_REASONING, 120.0, 16.0));
    EXPECT_FALSE(daemon.meets_sla(SLATier::TIER1_INTERACTIVE_REASONING, 180.0, 16.0));
    // Tier 2: <=400ms TTFT, <=40ms ITL
    EXPECT_TRUE(daemon.meets_sla(SLATier::TIER2_AGENTIC_TOOL_USE, 350.0, 32.0));
    EXPECT_FALSE(daemon.meets_sla(SLATier::TIER2_AGENTIC_TOOL_USE, 350.0, 48.0));

    daemon.start_background_50ms_polling();
    std::this_thread::sleep_for(std::chrono::milliseconds(130));
    daemon.stop_background_polling();

    daemon.record_request({1, SLATier::TIER1_INTERACTIVE_REASONING, 1000, 800, 100, 95.0, 12.0, 1.0, true});
    daemon.record_request({2, SLATier::TIER2_AGENTIC_TOOL_USE, 4000, 3000, 200, 520.0, 25.0, 8.0, true}); // violates Tier 2 TTFT
    auto kpi = daemon.snapshot_kpis(1.0);
    EXPECT_NEAR(kpi.sla_goodput_pct, 50.0, 1e-4);
    EXPECT_GT(kpi.daemon_poll_ticks_50ms, 0ULL);
    EXPECT_GT(kpi.tokens_per_watt, 0.0);

    std::string prom = daemon.export_prometheus_metrics(1.0);
    EXPECT_TRUE(prom.find("dynamoecon_cost_per_million_goodput_usd") != std::string::npos);
}

// 5. NFR-2.1, NFR-2.2 & NFR-2.3 End-to-End Verification Gate Suite
TEST(DynamoEconNIXL, NFR21_NFR22_NFR23_PerformanceGoodputAndFailoverGates) {
    // Multi-threaded 10,000+ RPS concurrent routing stress test across 32 workers (NFR-2.1)
    ConcurrentRadixKVIndexer idx(32, 64ULL * 1024 * 1024 * 1024, 16);
    LayerPipelinedNIXLEngine nixl(32, 120.0);
    CrossoverDisaggregatedScheduler sched(idx, nixl, 16, 16, 512, 18432ULL);

    std::vector<WorkerQueueState> workers(32);
    for (int i = 0; i < 32; ++i) {
        workers[i].worker_id       = i;
        workers[i].is_prefill_pool = (i < 16);
        workers[i].healthy         = true;
        workers[i].queue_wait_ms   = 5.0 + (i % 8);
    }
    uint64_t prefix[32];
    for (int b = 0; b < 32; ++b) prefix[b] = 0x7700ULL + static_cast<uint64_t>(b);
    idx.insert_prefix_blocks(4, prefix, 24, 294912ULL, 8e9, 1.0);

    const int kTotalReqs = 10000;
    const int kThreads   = 4;
    std::vector<double> latencies_us(kTotalReqs, 0.0);

    auto t_start = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int tid = 0; tid < kThreads; ++tid) {
        pool.emplace_back([&, tid]() {
            int begin = tid * (kTotalReqs / kThreads);
            int end   = begin + (kTotalReqs / kThreads);
            for (int r = begin; r < end; ++r) {
                auto d = sched.schedule_request(r, 512 + (r & 511), prefix, 32, workers, 1.0, 2.0);
                latencies_us[r] = d.routing_latency_us;
            }
        });
    }
    for (auto& th : pool) th.join();
    auto t_end = std::chrono::steady_clock::now();

    double elapsed_s = std::max(1e-6, std::chrono::duration<double>(t_end - t_start).count());
    double achieved_rps = static_cast<double>(kTotalReqs) / elapsed_s;
    std::sort(latencies_us.begin(), latencies_us.end());
    double p99_us = latencies_us[static_cast<size_t>(kTotalReqs * 0.99)];

    std::cout << "  [NFR-2.1] 10k Concurrent Routing: Throughput=" << achieved_rps
              << " req/s, P99=" << p99_us << " us (limit <= 500 us)\n";
    EXPECT_LE(p99_us, 500.0);
    EXPECT_GE(achieved_rps, 10000.0);

    // NFR-2.2 & NFR-2.3 Agentic Multi-Turn Benchmark & Sub-10ms Failover
    DynamoEconBenchmarkSummary sum{};
    ASSERT_EQ(dynamoecon_run_agentic_benchmark(640, 32, 1, 1.0, 1.2, &sum), 0);

    std::cout << "  [NFR-2.2] SLA Goodput: Adaptive=" << sum.adaptive_sla_goodput_pct
              << "% vs Baseline=" << sum.baseline_sla_goodput_pct
              << "% (Lift=+" << sum.goodput_improvement_pct << "%)\n";
    std::cout << "  [NFR-2.2] Cost/1M Goodput: Adaptive=$" << sum.adaptive_cost_per_1m_usd
              << " vs Baseline=$" << sum.baseline_cost_per_1m_usd
              << " (Reduction=-" << sum.cost_reduction_pct << "%)\n";
    std::cout << "  [NFR-2.3] Failover Latency=" << sum.failover_latency_ms
              << " ms, Re-anchored=" << sum.failover_sessions_reanchored
              << ", Lost=" << sum.failover_sessions_lost << "\n";

    EXPECT_GE(sum.goodput_improvement_pct, 35.0);
    EXPECT_GE(sum.cost_reduction_pct, 30.0);
    EXPECT_LE(sum.failover_latency_ms, 10.0);
    EXPECT_EQ(sum.failover_sessions_lost, 0);
    EXPECT_EQ(sum.failover_sessions_reanchored, 24);
}

#if !defined(USE_SYSTEM_GTEST)
int main() {
    return ::gtest_lite::run_all_tests();
}
#endif
