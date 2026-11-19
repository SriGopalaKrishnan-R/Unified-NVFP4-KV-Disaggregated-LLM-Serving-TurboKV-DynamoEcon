#include "dynamoecon/econ_telemetryd.h"
#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
    bool oneshot = (argc > 1 && std::string(argv[1]) == "--oneshot");

    dynamoecon::TelemetryHAL hal(dynamoecon::HardwareProfile::B200_SXM6_1000W, false);
    dynamoecon::EconTelemetryDaemon daemon(hal, 32);

    daemon.start_background_50ms_polling();
    std::this_thread::sleep_for(std::chrono::milliseconds(oneshot ? 160 : 500));

    // Seed representative multi-tier telemetry records
    daemon.record_request({1, dynamoecon::SLATier::TIER1_INTERACTIVE_REASONING, 1024, 768, 128, 92.4, 11.8, 1.4, true});
    daemon.record_request({2, dynamoecon::SLATier::TIER2_AGENTIC_TOOL_USE,      8192, 6144, 256, 218.0, 21.5, 4.2, true});
    daemon.record_request({3, dynamoecon::SLATier::TIER3_BATCH_SYNTHESIS,      16384, 8192, 512, 640.0, 34.0, 9.1, true});

    daemon.stop_background_polling();

    std::cout << "# NV-DEVTECH-SRS-2026-V3 | econ_telemetryd (" << hal.profile_name() << ")\n";
    std::cout << daemon.export_prometheus_metrics(1.0);
    return 0;
}
