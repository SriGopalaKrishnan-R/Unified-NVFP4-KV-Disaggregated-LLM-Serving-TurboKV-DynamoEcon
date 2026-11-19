"""FastAPI Control Plane, ZeroMQ KV-Event Bus & Prometheus Observability Server (Section 3.4)."""

from __future__ import annotations

import json
import math
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import CollectorRegistry, Gauge, generate_latest
from pydantic import BaseModel, Field

from control_plane.bridge import get_bridge
from control_plane.kv_events_zmq import DynamoZMQKVEventBus
from control_plane.replay_benchmark import (
    ARTIFACTS_DIR,
    generate_pareto_frontiers,
    materialize_verification_artifacts,
)

# Dedicated Prometheus registry so repeated imports in tests never collide
PROM_REGISTRY = CollectorRegistry()
G_COMPRESSION = Gauge("turbokv_compression_ratio", "KV-cache compression ratio vs BF16", registry=PROM_REGISTRY)
G_HBM_UTIL = Gauge("turbokv_hbm_bw_utilization_pct", "Path 3 decode HBM Roofline utilization %", registry=PROM_REGISTRY)
G_COSINE = Gauge("turbokv_cosine_similarity", "Attention output cosine similarity vs BF16", registry=PROM_REGISTRY)
G_KL_DIV = Gauge("turbokv_kl_divergence", "Attention softmax KL divergence vs BF16", registry=PROM_REGISTRY)
G_SLA_GOODPUT = Gauge("dynamoecon_sla_goodput_pct", "Cluster multi-tier SLA goodput %", registry=PROM_REGISTRY)
G_TOK_PER_WATT = Gauge("dynamoecon_tokens_per_watt", "Cluster output tokens per Watt", registry=PROM_REGISTRY)
G_PREFIX_REUSE = Gauge("dynamoecon_effective_prefix_reuse_pct", "Effective Radix KV prefix reuse %", registry=PROM_REGISTRY)
G_COST_1M = Gauge("dynamoecon_cost_per_million_goodput_usd", "USD cost per 1M SLA-compliant tokens", registry=PROM_REGISTRY)
G_P99_ROUTER_US = Gauge("dynamoecon_router_p99_us", "P99 crossover scheduler latency in microseconds", registry=PROM_REGISTRY)
G_FAILOVER_MS = Gauge("dynamoecon_failover_latency_ms", "Last Decode heartbeat failover latency in ms", registry=PROM_REGISTRY)


class AppState:
    zmq_bus: DynamoZMQKVEventBus | None = None
    cached_verification: Dict[str, Any] | None = None
    cached_pareto: Dict[str, Any] | None = None
    failed_workers: set[int] = set()
    last_failover_event: Dict[str, Any] | None = None


STATE = AppState()


def _sync_prometheus_gauges(ver: Dict[str, Any]) -> None:
    tkv = ver["turbokv_b200"]
    dyn = ver["dynamoecon_benchmark"]
    G_COMPRESSION.set(tkv["compression_ratio"])
    G_HBM_UTIL.set(tkv["hbm_bw_utilization_pct"])
    G_COSINE.set(tkv["turbokv_cosine_sim"])
    G_KL_DIV.set(tkv["turbokv_kl_div"])
    G_SLA_GOODPUT.set(dyn["adaptive_sla_goodput_pct"])
    G_TOK_PER_WATT.set(dyn["adaptive_tokens_per_watt"])
    G_PREFIX_REUSE.set(dyn["adaptive_prefix_reuse_pct"])
    G_COST_1M.set(dyn["adaptive_cost_per_1m_usd"])
    G_P99_ROUTER_US.set(dyn["scheduler_p99_us"])
    G_FAILOVER_MS.set(dyn["failover_latency_ms"])


def _seed_initial_zmq_events(bus: DynamoZMQKVEventBus) -> None:
    bus.publish_event(
        "BlockStored",
        worker_id=2,
        block_hashes=["0xa6e00001", "0xa6e00002", "0xa6e00003", "0xa6e00004"],
        parent_hash="0x0",
        s_retain=184520.4,
    )
    bus.publish_event(
        "BlockStored",
        worker_id=18,
        block_hashes=["0xa6e10001", "0xa6e10002"],
        parent_hash="0xa6e00004",
        s_retain=142310.8,
    )
    bus.publish_event(
        "BlockRemoved",
        worker_id=7,
        block_hashes=["0xee00991f"],
        parent_hash="0xee00991e",
        s_retain=412.6,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    STATE.zmq_bus = DynamoZMQKVEventBus()
    _seed_initial_zmq_events(STATE.zmq_bus)
    ver_file = ARTIFACTS_DIR / "srs_verification_report.json"
    par_file = ARTIFACTS_DIR / "aiperf_pareto_frontier.json"
    if ver_file.exists() and par_file.exists():
        STATE.cached_verification = json.loads(ver_file.read_text())
        STATE.cached_pareto = json.loads(par_file.read_text())
    else:
        STATE.cached_verification = materialize_verification_artifacts()
        STATE.cached_pareto = json.loads(par_file.read_text())
    _sync_prometheus_gauges(STATE.cached_verification)
    yield
    if STATE.zmq_bus is not None:
        STATE.zmq_bus.close()
        STATE.zmq_bus = None


app = FastAPI(
    title="NV-DEVTECH-SRS-2026-V3 Tokconomics Control Plane",
    version="3.0.0",
    description="TurboKV-FP4 & DynamoEcon-NIXL Co-Design Telemetry & Orchestration API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TurboKVEvalRequest(BaseModel):
    seq_len: int = Field(default=65536, ge=256, le=131072)
    batch_size: int = Field(default=32, ge=1, le=256)
    head_dim: int = Field(default=128)
    arch_code: int = Field(default=100)
    dist_code: int = Field(default=0, ge=0, le=2)
    spec_depth: int = Field(default=6, ge=1, le=8)
    spec_branch: int = Field(default=3, ge=1, le=4)
    accept_rate_bias: float = Field(default=0.92, ge=0.1, le=1.0)


class CrossoverRequest(BaseModel):
    prompt_len: int = Field(default=4096, ge=64, le=131072)
    matched_prefix_len: int = Field(default=2048, ge=0, le=131072)
    nixl_congestion: float = Field(default=1.2, ge=1.0, le=200.0)
    link_bw_gbps: float = Field(default=100.0, ge=5.0, le=1800.0)
    l_threshold: int = Field(default=512, ge=64, le=8192)


class BenchmarkRunRequest(BaseModel):
    num_requests: int = Field(default=640, ge=64, le=5000)
    num_workers: int = Field(default=32, ge=4, le=64)
    hw_profile_code: int = Field(default=1, ge=0, le=1)
    arrival_rate_scale: float = Field(default=1.0, ge=0.3, le=2.5)
    nixl_congestion: float = Field(default=1.2, ge=1.0, le=50.0)


class FailoverInjectRequest(BaseModel):
    decode_worker_id: int = Field(default=19, ge=16, le=31)
    reset_all_healthy: bool = Field(default=False)


@app.get("/health")
def health() -> Dict[str, Any]:
    bridge = get_bridge()
    return {
        "status": "ok",
        "document_id": "NV-DEVTECH-SRS-2026-V3",
        "native_turbokv_dll": bridge.tkv_path,
        "native_dynamoecon_dll": bridge.dyn_path,
        "zmq_kv_events_endpoint": STATE.zmq_bus.endpoint if STATE.zmq_bus else None,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    if STATE.cached_verification is None:
        STATE.cached_verification = materialize_verification_artifacts()
        _sync_prometheus_gauges(STATE.cached_verification)
    return generate_latest(PROM_REGISTRY).decode("utf-8")


@app.get("/api/v1/verification-gates")
def get_verification_gates(refresh: bool = False) -> Dict[str, Any]:
    if refresh or STATE.cached_verification is None:
        STATE.cached_verification = materialize_verification_artifacts()
        _sync_prometheus_gauges(STATE.cached_verification)
    return STATE.cached_verification


@app.post("/api/v1/turbokv/evaluate")
def evaluate_turbokv(req: TurboKVEvalRequest) -> Dict[str, Any]:
    bridge = get_bridge()
    res = bridge.evaluate_turbokv(
        seq_len=req.seq_len,
        batch_size=req.batch_size,
        head_dim=64 if req.head_dim == 64 else 128,
        arch_code=req.arch_code if req.arch_code in (89, 90, 100) else 100,
        dist_code=req.dist_code,
        spec_depth=req.spec_depth,
        spec_branch=req.spec_branch,
        accept_rate_bias=req.accept_rate_bias,
    )
    G_COMPRESSION.set(res["compression_ratio"])
    G_HBM_UTIL.set(res["hbm_bw_utilization_pct"])
    G_COSINE.set(res["turbokv_cosine_sim"])
    G_KL_DIV.set(res["turbokv_kl_div"])
    return res


@app.post("/api/v1/dynamoecon/crossover")
def evaluate_crossover(req: CrossoverRequest) -> Dict[str, Any]:
    bridge = get_bridge()
    matched = min(req.prompt_len, req.matched_prefix_len)
    res = bridge.probe_crossover(
        prompt_len=req.prompt_len,
        matched_prefix_len=matched,
        nixl_congestion=req.nixl_congestion,
        link_bw_gbps=req.link_bw_gbps,
        l_threshold=req.l_threshold,
    )
    if STATE.zmq_bus is not None:
        STATE.zmq_bus.publish_event(
            "BlockStored",
            worker_id=res["prefill_worker_id"],
            block_hashes=[f"0xc0de{req.prompt_len:04x}"],
            parent_hash="0xa6e00001",
            s_retain=98450.0 + req.prompt_len * 4.2,
        )
    return res


@app.post("/api/v1/dynamoecon/benchmark")
def run_benchmark(req: BenchmarkRunRequest) -> Dict[str, Any]:
    bridge = get_bridge()
    res = bridge.run_agentic_benchmark(
        num_requests=req.num_requests,
        num_workers=req.num_workers,
        hw_profile_code=req.hw_profile_code,
        arrival_rate_scale=req.arrival_rate_scale,
        nixl_congestion=req.nixl_congestion,
    )
    G_SLA_GOODPUT.set(res["adaptive_sla_goodput_pct"])
    G_TOK_PER_WATT.set(res["adaptive_tokens_per_watt"])
    G_PREFIX_REUSE.set(res["adaptive_prefix_reuse_pct"])
    G_COST_1M.set(res["adaptive_cost_per_1m_usd"])
    G_P99_ROUTER_US.set(res["scheduler_p99_us"])
    if STATE.zmq_bus is not None:
        STATE.zmq_bus.publish_event(
            "BlockStored",
            worker_id=1,
            block_hashes=["0xa6e00101", "0xa6e00102"],
            parent_hash="0xa6e00004",
            s_retain=215400.0,
        )
    return res


@app.post("/api/v1/dynamoecon/failover")
def inject_failover(req: FailoverInjectRequest) -> Dict[str, Any]:
    if req.reset_all_healthy:
        STATE.failed_workers.clear()
        STATE.last_failover_event = None
        return {"status": "cluster_restored", "failed_workers": []}

    bridge = get_bridge()
    bench = bridge.run_agentic_benchmark(
        num_requests=128, num_workers=32, hw_profile_code=1, arrival_rate_scale=1.0, nixl_congestion=1.1
    )
    STATE.failed_workers.add(req.decode_worker_id)
    backup_id = 16 + ((req.decode_worker_id - 15) % 16)
    if backup_id in STATE.failed_workers:
        backup_id = next((w for w in range(16, 32) if w not in STATE.failed_workers), 16)

    ev = {
        "failed_decode_worker": req.decode_worker_id,
        "backup_decode_worker": backup_id,
        "failover_latency_ms": round(bench["failover_latency_ms"], 4),
        "sessions_reanchored": bench["failover_sessions_reanchored"],
        "sessions_lost": bench["failover_sessions_lost"],
        "kv_blocks_migrated": bench["failover_kv_blocks_migrated"],
        "nfr23_passed": bench["failover_latency_ms"] <= 10.0 and bench["failover_sessions_lost"] == 0,
        "timestamp": time.time(),
    }
    STATE.last_failover_event = ev
    G_FAILOVER_MS.set(ev["failover_latency_ms"])

    if STATE.zmq_bus is not None:
        STATE.zmq_bus.publish_event(
            "BlockStored",
            worker_id=backup_id,
            block_hashes=[f"0xf0{req.decode_worker_id:02d}0001", f"0xf0{req.decode_worker_id:02d}0002"],
            parent_hash="0xa6e00001",
            s_retain=310000.0,
        )
    return ev


@app.get("/api/v1/pareto")
def get_pareto(hw_profile_code: int = 1, refresh: bool = False) -> Dict[str, Any]:
    if refresh or STATE.cached_pareto is None or hw_profile_code != 1:
        pareto = generate_pareto_frontiers(hw_profile_code=hw_profile_code)
        if hw_profile_code == 1:
            STATE.cached_pareto = pareto
        return pareto
    return STATE.cached_pareto


@app.get("/api/v1/kv-events")
def get_kv_events(limit: int = 25) -> Dict[str, Any]:
    if STATE.zmq_bus is None:
        return {"endpoint": None, "events": []}
    return {
        "endpoint": STATE.zmq_bus.endpoint,
        "topic": STATE.zmq_bus.topic,
        "events": STATE.zmq_bus.recent_events(limit=limit),
    }


@app.get("/api/v1/cluster-telemetry")
def get_cluster_telemetry() -> Dict[str, Any]:
    t = time.time()
    workers: List[Dict[str, Any]] = []
    total_power = 0.0
    for wid in range(32):
        is_prefill = wid < 16
        healthy = wid not in STATE.failed_workers
        phase = t * 0.8 + wid * 0.45
        if not healthy:
            u_sm, b_hbm, q_ms, hbm_used_pct = 0.0, 0.0, 0.0, 0.0
            watts = 0.0
        else:
            u_sm = (0.64 if is_prefill else 0.44) + 0.14 * abs(math.sin(phase))
            b_hbm = (0.48 if is_prefill else 0.82) + 0.09 * abs(math.cos(phase))
            watts = 120.0 + 600.0 * min(1.0, u_sm) + 280.0 * min(1.0, b_hbm)
            q_ms = (5.5 if is_prefill else 3.8) + (9.5 if wid % 4 == 0 else 2.2) * abs(math.sin(phase * 0.7))
            hbm_used_pct = 74.0 + 14.5 * abs(math.sin(wid * 0.9 + t * 0.2))
        total_power += watts
        workers.append(
            {
                "worker_id": wid,
                "pool": "PREFILL" if is_prefill else "DECODE",
                "healthy": healthy,
                "sm_util_pct": round(u_sm * 100.0, 1),
                "hbm_bw_pct": round(b_hbm * 100.0, 1),
                "hbm_bw_tbps": round(b_hbm * 8.0, 2),
                "power_watts": round(watts, 1),
                "queue_wait_ms": round(q_ms, 1),
                "radix_hbm_used_pct": round(hbm_used_pct, 1),
            }
        )

    ver = STATE.cached_verification or materialize_verification_artifacts()
    return {
        "poll_interval_ms": 50,
        "telemetry_backend": "Physics-Calibrated B200 SXM6 Simulation HAL (NVML C-API Ready)",
        "cluster_power_kw": round(total_power / 1000.0, 2),
        "failed_workers": sorted(STATE.failed_workers),
        "last_failover": STATE.last_failover_event,
        "kpis": ver["dynamoecon_benchmark"],
        "workers": workers,
    }


# Mount production UI bundle at `/` if built (`GPU_PJ/ui/dist`)
_UI_DIST = ARTIFACTS_DIR.parent / "ui" / "dist"
if _UI_DIST.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_UI_DIST), html=True), name="ui_static")

