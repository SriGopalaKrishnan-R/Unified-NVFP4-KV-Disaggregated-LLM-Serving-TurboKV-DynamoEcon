#include "dynamoecon/telemetry_hal.h"
#include <algorithm>
#include <cmath>

#if defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

namespace dynamoecon {

TelemetryHAL::TelemetryHAL(HardwareProfile profile, bool prefer_hardware_nvml)
    : profile_(profile)
{
    if (profile_ == HardwareProfile::H100_SXM5_700W) {
        p_idle_watts_   = 70.0;
        alpha_sm_watts_ = 430.0;
        beta_hbm_watts_ = 200.0;
        peak_hbm_tbps_  = 3.35;
    } else {
        // NVIDIA B200 SXM6 (1000W TDP)
        p_idle_watts_   = 120.0;
        alpha_sm_watts_ = 600.0;
        beta_hbm_watts_ = 280.0;
        peak_hbm_tbps_  = 8.00;
    }

    if (prefer_hardware_nvml) {
#if defined(_WIN32)
        HMODULE mod = LoadLibraryA("nvml.dll");
        if (mod) {
            using fn_nvmlInit_t = int(*)();
            auto p_init = reinterpret_cast<fn_nvmlInit_t>(reinterpret_cast<void*>(GetProcAddress(mod, "nvmlInit_v2")));
            p_get_handle_ = reinterpret_cast<fn_nvmlDeviceGetHandleByIndex_t>(reinterpret_cast<void*>(GetProcAddress(mod, "nvmlDeviceGetHandleByIndex_v2")));
            p_get_power_  = reinterpret_cast<fn_nvmlDeviceGetPowerUsage_t>(reinterpret_cast<void*>(GetProcAddress(mod, "nvmlDeviceGetPowerUsage")));
            p_get_util_   = reinterpret_cast<fn_nvmlDeviceGetUtilization_t>(reinterpret_cast<void*>(GetProcAddress(mod, "nvmlDeviceGetUtilizationRates")));
            if (p_init && p_get_handle_ && p_get_power_ && p_init() == 0) {
                nvml_handle_ = reinterpret_cast<void*>(mod);
                nvml_loaded_ = true;
            } else {
                FreeLibrary(mod);
            }
        }
#else
        void* mod = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (mod) {
            using fn_nvmlInit_t = int(*)();
            auto p_init = reinterpret_cast<fn_nvmlInit_t>(dlsym(mod, "nvmlInit_v2"));
            p_get_handle_ = reinterpret_cast<fn_nvmlDeviceGetHandleByIndex_t>(dlsym(mod, "nvmlDeviceGetHandleByIndex_v2"));
            p_get_power_  = reinterpret_cast<fn_nvmlDeviceGetPowerUsage_t>(dlsym(mod, "nvmlDeviceGetPowerUsage"));
            p_get_util_   = reinterpret_cast<fn_nvmlDeviceGetUtilization_t>(dlsym(mod, "nvmlDeviceGetUtilizationRates"));
            if (p_init && p_get_handle_ && p_get_power_ && p_init() == 0) {
                nvml_handle_ = mod;
                nvml_loaded_ = true;
            } else {
                dlclose(mod);
            }
        }
        void* dcgm = dlopen("libdcgm.so.3", RTLD_LAZY);
        if (dcgm) {
            dcgm_handle_ = dcgm;
            dcgm_loaded_ = true;
        }
#endif
    }
}

TelemetryHAL::~TelemetryHAL() {
#if defined(_WIN32)
    if (nvml_handle_) {
        FreeLibrary(reinterpret_cast<HMODULE>(nvml_handle_));
    }
#else
    if (nvml_handle_) dlclose(nvml_handle_);
    if (dcgm_handle_) dlclose(dcgm_handle_);
#endif
}

double TelemetryHAL::compute_physics_power_watts(double u_sm, double b_hbm_norm) const noexcept {
    double clamped_sm  = std::clamp(u_sm, 0.0, 1.0);
    double clamped_hbm = std::clamp(b_hbm_norm, 0.0, 1.0);
    // FR-2.4 Physics-calibrated power equation: P(t) = P_idle + alpha * U_SM + beta * B_HBM
    return p_idle_watts_ + alpha_sm_watts_ * clamped_sm + beta_hbm_watts_ * clamped_hbm;
}

WorkerTelemetrySample TelemetryHAL::poll_worker(
    int worker_id,
    double sim_u_sm,
    double sim_b_hbm_norm,
    double sim_nvlink_tx_gbps,
    double sim_nvlink_rx_gbps) const
{
    WorkerTelemetrySample s{};
    s.worker_id          = worker_id;
    s.sm_utilization     = std::clamp(sim_u_sm, 0.0, 1.0);
    s.hbm_bw_utilization = std::clamp(sim_b_hbm_norm, 0.0, 1.0);
    s.hbm_bw_tbps        = s.hbm_bw_utilization * peak_hbm_tbps_;
    s.nvlink_tx_gbps     = sim_nvlink_tx_gbps;
    s.nvlink_rx_gbps     = sim_nvlink_rx_gbps;
    s.sm_active_cycles   = static_cast<uint64_t>(s.sm_utilization * 1.98e9 * 0.050); // 1.98 GHz * 50ms

    if (nvml_loaded_ && p_get_handle_ && p_get_power_) {
        void* dev = nullptr;
        unsigned int mw = 0;
        if (p_get_handle_(static_cast<unsigned int>(worker_id), &dev) == 0 &&
            p_get_power_(dev, &mw) == 0) {
            s.board_power_watts  = static_cast<double>(mw) / 1000.0;
            s.from_nvml_hardware = true;
            if (p_get_util_) {
                struct NvmlUtil { unsigned int gpu; unsigned int memory; } util{};
                if (p_get_util_(dev, &util) == 0) {
                    s.sm_utilization     = static_cast<double>(util.gpu) / 100.0;
                    s.hbm_bw_utilization = static_cast<double>(util.memory) / 100.0;
                    s.hbm_bw_tbps        = s.hbm_bw_utilization * peak_hbm_tbps_;
                }
            }
            return s;
        }
    }

    s.board_power_watts  = compute_physics_power_watts(s.sm_utilization, s.hbm_bw_utilization);
    s.from_nvml_hardware = false;
    return s;
}

const char* TelemetryHAL::profile_name() const noexcept {
    return (profile_ == HardwareProfile::H100_SXM5_700W)
        ? "NVIDIA H100 SXM5 (700W, 3.35 TB/s HBM3)"
        : "NVIDIA B200 SXM6 (1000W, 8.0 TB/s HBM3e)";
}

double TelemetryHAL::peak_hbm_tbps() const noexcept {
    return peak_hbm_tbps_;
}

double TelemetryHAL::gpu_hourly_capex_usd() const noexcept {
    // Realistic amortized GPU cloud/cluster hourly rate per GPU ($2.85 H100, $4.25 B200)
    return (profile_ == HardwareProfile::H100_SXM5_700W) ? 2.85 : 4.25;
}

} // namespace dynamoecon
