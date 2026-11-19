#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "turbokv/paged_attention.h"
#include "turbokv/roofline_profiler.h"
#include "turbokv/spec_tree_verify.h"

namespace py = pybind11;

PYBIND11_MODULE(turbokv_py, m) {
    m.doc() = "TurboKV-FP4 Dual-Level Microscaled NVFP4 KV-Cache & Speculative Tree Engine (SGLang / TensorRT-LLM Compatible)";

    py::class_<turbokv::HybridKVConfig>(m, "HybridKVConfig")
        .def(py::init<>())
        .def_readwrite("num_kv_heads", &turbokv::HybridKVConfig::num_kv_heads)
        .def_readwrite("head_dim", &turbokv::HybridKVConfig::head_dim)
        .def_readwrite("page_size", &turbokv::HybridKVConfig::page_size)
        .def_readwrite("n_sink", &turbokv::HybridKVConfig::n_sink)
        .def_readwrite("n_window", &turbokv::HybridKVConfig::n_window)
        .def_readwrite("use_hadamard", &turbokv::HybridKVConfig::use_hadamard);

    py::class_<turbokv::MemoryFootprintReport>(m, "MemoryFootprintReport")
        .def_readonly("seq_len", &turbokv::MemoryFootprintReport::seq_len)
        .def_readonly("bf16_bytes", &turbokv::MemoryFootprintReport::bf16_bytes)
        .def_readonly("total_hybrid_bytes", &turbokv::MemoryFootprintReport::total_hybrid_bytes)
        .def_readonly("compression_ratio", &turbokv::MemoryFootprintReport::compression_ratio);

    py::class_<turbokv::HybridPagedKVCache>(m, "HybridPagedKVCache")
        .def(py::init<const turbokv::HybridKVConfig&>())
        .def("compute_footprint", &turbokv::HybridPagedKVCache::compute_footprint);

    py::enum_<turbokv::GPUArchTarget>(m, "GPUArchTarget")
        .value("ADA_SM89", turbokv::GPUArchTarget::ADA_SM89_RTX4090)
        .value("HOPPER_SM90", turbokv::GPUArchTarget::HOPPER_SM90_H100_SXM5)
        .value("BLACKWELL_SM100", turbokv::GPUArchTarget::BLACKWELL_SM100_B200);

    py::class_<turbokv::RooflineMetrics>(m, "RooflineMetrics")
        .def_readonly("peak_hbm_bw_gbps", &turbokv::RooflineMetrics::peak_hbm_bw_gbps)
        .def_readonly("achieved_hbm_bw_gbps", &turbokv::RooflineMetrics::achieved_hbm_bw_gbps)
        .def_readonly("hbm_bw_utilization_pct", &turbokv::RooflineMetrics::hbm_bw_utilization_pct)
        .def_readonly("estimated_kernel_us", &turbokv::RooflineMetrics::estimated_kernel_us)
        .def_readonly("speedup_vs_bf16_decode", &turbokv::RooflineMetrics::speedup_vs_bf16_decode);

    m.def("analyze_path3_decode_roofline", &turbokv::analyze_path3_decode_roofline);
    m.def("export_ncu_roofline_json", &turbokv::export_ncu_roofline_json);
}
