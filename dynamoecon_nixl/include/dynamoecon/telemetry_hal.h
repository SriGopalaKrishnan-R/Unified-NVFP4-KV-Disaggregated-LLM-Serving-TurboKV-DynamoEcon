#pragma once

#include "dynamoecon/kv_events.h"
#include <cstdint>
#include <string>
#include <vector>

namespace dynamoecon {

// ==============================================================================
// FR-2.4: Dual-Mode TelemetryHAL
//   - Hardware Backend: Runtime dynamic C API bindings (`LoadLibrary` / `dlopen`)
//     to `libnvidia-ml` (NVML) and `libdcgm` for instantaneous board power (W),
//     SM active cycles, and NVLink TX/RX throughput.
//   - Simulation Backend: Physics-calibrated H100 SXM5 / B200 power & bandwidth model:
//       P(t) = P_idle + alpha * U_SM(t) + beta * B_HBM(t)
//     for deterministic CPU-only execution.
// ==============================================================================

enum class HardwareProfile : int {
    H100_SXM5_700W  = 0,
    B200_SXM6_1000W = 1
};

struct WorkerTelemetrySample {
    int      worker_id           = 0;
    double   board_power_watts   = 70.0;
    double   sm_utilization      = 0.0; // U_SM in [0.0, 1.0]
    double   hbm_bw_utilization  = 0.0; // Normalized [0.0, 1.0]
    double   hbm_bw_tbps         = 0.0; // Instantaneous HBM bandwidth (TB/s)
    double   nvlink_tx_gbps      = 0.0;
    double   nvlink_rx_gbps      = 0.0;
    uint64_t sm_active_cycles    = 0;
    bool     from_nvml_hardware  = false;
};

class DYNAMOECON_API TelemetryHAL {
public:
    explicit TelemetryHAL(
        HardwareProfile profile = HardwareProfile::B200_SXM6_1000W,
        bool prefer_hardware_nvml = false);
    ~TelemetryHAL();

    // Physics-calibrated instantaneous GPU power equation (FR-2.4):
    //   P(t) = P_idle + alpha * U_SM + beta * B_HBM_norm
    double compute_physics_power_watts(double u_sm, double b_hbm_norm) const noexcept;

    WorkerTelemetrySample poll_worker(
        int worker_id,
        double sim_u_sm,
        double sim_b_hbm_norm,
        double sim_nvlink_tx_gbps = 0.0,
        double sim_nvlink_rx_gbps = 0.0) const;

    bool hardware_nvml_active() const noexcept { return nvml_loaded_; }
    bool hardware_dcgm_active() const noexcept { return dcgm_loaded_; }
    HardwareProfile profile() const noexcept { return profile_; }
    const char* profile_name() const noexcept;
    double peak_hbm_tbps() const noexcept;
    double gpu_hourly_capex_usd() const noexcept;

private:
    HardwareProfile profile_;
    double p_idle_watts_ = 120.0;
    double alpha_sm_watts_ = 600.0;
    double beta_hbm_watts_ = 280.0;
    double peak_hbm_tbps_  = 8.0;

    bool  nvml_loaded_ = false;
    bool  dcgm_loaded_ = false;
    void* nvml_handle_ = nullptr;
    void* dcgm_handle_ = nullptr;

    // Function pointers resolved dynamically if libnvidia-ml is available
    using fn_nvmlDeviceGetHandleByIndex_t = int(*)(unsigned int, void**);
    using fn_nvmlDeviceGetPowerUsage_t    = int(*)(void*, unsigned int*);
    using fn_nvmlDeviceGetUtilization_t   = int(*)(void*, void*);
    fn_nvmlDeviceGetHandleByIndex_t p_get_handle_ = nullptr;
    fn_nvmlDeviceGetPowerUsage_t    p_get_power_  = nullptr;
    fn_nvmlDeviceGetUtilization_t   p_get_util_   = nullptr;
};

} // namespace dynamoecon
