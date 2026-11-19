"""Pytest Verification Suite for Control Plane, Native C++ Bridge, ZeroMQ KV Events & Prometheus."""

from __future__ import annotations

from fastapi.testclient import TestClient

from control_plane.app import app
from control_plane.bridge import get_bridge
from control_plane.kv_events_zmq import DynamoZMQKVEventBus
from control_plane.replay_benchmark import (
    generate_pareto_frontiers,
    materialize_verification_artifacts,
)


def test_native_bridge_and_all_srs_gates() -> None:
    rep = materialize_verification_artifacts()
    assert rep["document_id"] == "NV-DEVTECH-SRS-2026-V3"
    assert rep["all_gates_passed"] is True
    for gate in rep["gates"]:
        assert gate["passed"] is True, f"SRS Gate failed: {gate}"


def test_zmq_kv_events_bus_roundtrip() -> None:
    import time

    bus = DynamoZMQKVEventBus(topic="dynamo.test.kv")
    try:
        bus.publish_event(
            "BlockStored",
            worker_id=4,
            block_hashes=["0x1111", "0x2222"],
            parent_hash="0x0",
            s_retain=12345.6,
        )
        time.sleep(0.08)
        events = bus.recent_events()
        assert len(events) >= 1
        assert events[-1]["event"]["type"] == "BlockStored"
        assert events[-1]["worker_id"] == 4
    finally:
        bus.close()


def test_fastapi_endpoints_and_prometheus_metrics() -> None:
    with TestClient(app) as client:
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "ok"

        r_metrics = client.get("/metrics")
        assert r_metrics.status_code == 200
        text = r_metrics.text
        assert "turbokv_compression_ratio" in text
        assert "dynamoecon_sla_goodput_pct" in text
        assert "dynamoecon_cost_per_million_goodput_usd" in text

        r_tkv = client.post(
            "/api/v1/turbokv/evaluate",
            json={
                "seq_len": 65536,
                "batch_size": 32,
                "head_dim": 128,
                "arch_code": 100,
                "dist_code": 0,
                "spec_depth": 6,
                "spec_branch": 3,
                "accept_rate_bias": 0.92,
            },
        )
        assert r_tkv.status_code == 200
        tkv_data = r_tkv.json()
        assert tkv_data["compression_ratio"] >= 3.2
        assert tkv_data["hbm_bw_utilization_pct"] >= 80.0
        assert tkv_data["turbokv_cosine_sim"] >= 0.997
        assert tkv_data["turbokv_kl_div"] <= 0.015
        assert tkv_data["host_sync_calls"] == 0

        # Short prompt < L_threshold must bypass disaggregation
        r_short = client.post(
            "/api/v1/dynamoecon/crossover",
            json={
                "prompt_len": 256,
                "matched_prefix_len": 64,
                "nixl_congestion": 1.0,
                "link_bw_gbps": 100.0,
                "l_threshold": 512,
            },
        )
        assert r_short.status_code == 200
        assert r_short.json()["mode_label"] == "AGGREGATED_INPLACE"

        # Failover drill must complete in <= 10ms with 0 lost requests
        r_fo = client.post(
            "/api/v1/dynamoecon/failover",
            json={"decode_worker_id": 21, "reset_all_healthy": False},
        )
        assert r_fo.status_code == 200
        fo = r_fo.json()
        assert fo["nfr23_passed"] is True
        assert fo["failover_latency_ms"] <= 10.0
        assert fo["sessions_lost"] == 0

        r_pareto = client.get("/api/v1/pareto")
        assert r_pareto.status_code == 200
        assert len(r_pareto.json()["series"]) == 3
