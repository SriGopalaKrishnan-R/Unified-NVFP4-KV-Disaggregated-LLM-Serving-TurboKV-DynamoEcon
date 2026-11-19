#pragma once

#include "turbokv/kv_cache_manager.h"
#include <cstdint>
#include <string>
#include <vector>

namespace turbokv {

// ==============================================================================
// NFR-1.2 & FR-1.4: Analytical Roofline Bandwidth Model, Register Occupancy Audit,
// Nsight Compute (.ncu-rep compatible JSON/CSV) & GPGPU-Sim Memory Trace Emitter
// ==============================================================================

enum class GPUArchTarget : int {
    ADA_SM89_RTX4090      = 89,
    HOPPER_SM90_H100_SXM5 = 90,
    BLACKWELL_SM100_B200  = 100
};

struct RooflineMetrics {
    GPUArchTarget arch               = GPUArchTarget::BLACKWELL_SM100_B200;
    const char* arch_name            = "NVIDIA B200 (SM100)";
    int batch_size                   = 32;
    int seq_len                      = 65536;
    int num_q_heads                  = 64;
    int num_kv_heads                 = 8;
    int head_dim                     = 128;

    // Hardware specs
    double peak_hbm_bw_gbps          = 8000.0; // 8.0 TB/s B200, 3.35 TB/s H100 SXM5
    double peak_fp16_tflops          = 2250.0;
    int    sm_count                  = 148;
    int    registers_per_thread      = 64;     // <= 128 enforced by --maxrregcount=128 (FR-1.4)
    double achieved_occupancy_pct    = 75.0;   // >= 50% required by FR-1.4
    uint64_t shared_mem_bank_conflicts = 0;    // 0 required by FR-1.2 Path 3

    // Decode Path 3 transaction & Roofline accounting
    uint64_t total_bytes_read        = 0;
    uint64_t uint4_sector_txns       = 0;
    uint64_t total_flops             = 0;
    double arithmetic_intensity      = 0.0;    // FLOPs / Byte
    double estimated_kernel_us       = 0.0;
    double achieved_hbm_bw_gbps      = 0.0;
    double hbm_bw_utilization_pct    = 0.0;    // NFR-1.2: >= 80.0% at batch_size >= 32
    double speedup_vs_bf16_decode    = 0.0;
};

TURBOKV_API RooflineMetrics analyze_path3_decode_roofline(
    GPUArchTarget arch,
    int batch_size,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim);

// Emits Nsight Compute Roofline report (.ncu-rep.json) and GPGPU-Sim coalesced uint4 address trace
TURBOKV_API std::string export_ncu_roofline_json(const RooflineMetrics& m);
TURBOKV_API std::string generate_gpgpusim_decode_trace(
    const HybridPagedKVCache& cache,
    int seq_id,
    int max_warps_to_trace = 4);

} // namespace turbokv
