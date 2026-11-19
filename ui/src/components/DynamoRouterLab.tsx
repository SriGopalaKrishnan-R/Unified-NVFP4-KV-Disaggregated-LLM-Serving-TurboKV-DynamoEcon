import React, { useEffect, useState } from 'react';
import { ClusterTelemetryResponse, CrossoverResponse } from '../types';

export const DynamoRouterLab: React.FC = () => {
  const [promptLen, setPromptLen] = useState<number>(4096);
  const [matchedLen, setMatchedLen] = useState<number>(2048);
  const [congestion, setCongestion] = useState<number>(1.2);
  const [linkBw, setLinkBw] = useState<number>(100.0);
  const [threshold, setThreshold] = useState<number>(512);

  const [probe, setProbe] = useState<CrossoverResponse | null>(null);
  const [cluster, setCluster] = useState<ClusterTelemetryResponse | null>(null);
  const [failoverBusy, setFailoverBusy] = useState<boolean>(false);

  const runCrossover = async (overrides?: Partial<{
    prompt_len: number;
    matched_prefix_len: number;
    nixl_congestion: number;
    link_bw_gbps: number;
    l_threshold: number;
  }>) => {
    const payload = {
      prompt_len: overrides?.prompt_len ?? promptLen,
      matched_prefix_len: Math.min(overrides?.prompt_len ?? promptLen, overrides?.matched_prefix_len ?? matchedLen),
      nixl_congestion: overrides?.nixl_congestion ?? congestion,
      link_bw_gbps: overrides?.link_bw_gbps ?? linkBw,
      l_threshold: overrides?.l_threshold ?? threshold,
    };
    const res = await fetch('/api/v1/dynamoecon/crossover', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      setProbe(await res.json());
    }
  };

  const fetchCluster = async () => {
    const res = await fetch('/api/v1/cluster-telemetry');
    if (res.ok) {
      setCluster(await res.json());
    }
  };

  useEffect(() => {
    runCrossover();
    fetchCluster();
    const timer = setInterval(fetchCluster, 2000);
    return () => clearInterval(timer);
  }, []);

  const triggerFailover = async (decodeWorkerId: number, reset = false) => {
    setFailoverBusy(true);
    try {
      await fetch('/api/v1/dynamoecon/failover', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decode_worker_id: decodeWorkerId, reset_all_healthy: reset }),
      });
      await fetchCluster();
    } finally {
      setFailoverBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Crossover Router Controls */}
      <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <span className="inline-block w-2.5 h-2.5 rounded-full bg-[#00d4ff]" />
              Project 2: DynamoEcon-NIXL Crossover Scheduler & Layer-Pipelined RDMA Plane
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              T_total(i,j) = max(T_queue_P(i) + T_prefill(L_p - L_m), T_queue_D(j)) + T_NIXL_stall(L_p) + T_decode_first
            </p>
          </div>
          {probe && (
            <span
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold border ${
                probe.selected_mode === 0
                  ? 'bg-[#00d4ff]/15 border-[#00d4ff]/50 text-[#00d4ff]'
                  : 'bg-amber-500/15 border-amber-400/50 text-amber-300'
              }`}
            >
              ACTIVE ROUTE: {probe.mode_label} (P{probe.prefill_worker_id} → D{probe.decode_worker_id} in {probe.routing_latency_us.toFixed(1)} µs)
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 text-xs">
          <div>
            <label className="block text-slate-400 mb-1 font-mono">Prompt L_p: {promptLen} tok</label>
            <input
              type="range"
              min={128}
              max={16384}
              step={128}
              value={promptLen}
              onChange={(e) => {
                const v = Number(e.target.value);
                setPromptLen(v);
                const clampedM = Math.min(v, matchedLen);
                setMatchedLen(clampedM);
                runCrossover({ prompt_len: v, matched_prefix_len: clampedM });
              }}
              className="w-full accent-[#00d4ff]"
            />
          </div>

          <div>
            <label className="block text-slate-400 mb-1 font-mono">Radix Prefix L_m: {matchedLen} tok</label>
            <input
              type="range"
              min={0}
              max={promptLen}
              step={64}
              value={matchedLen}
              onChange={(e) => {
                const v = Number(e.target.value);
                setMatchedLen(v);
                runCrossover({ matched_prefix_len: v });
              }}
              className="w-full accent-[#76b900]"
            />
          </div>

          <div>
            <label className="block text-slate-400 mb-1 font-mono">NIXL Congestion: {congestion.toFixed(1)}x</label>
            <input
              type="range"
              min={1.0}
              max={150.0}
              step={1.5}
              value={congestion}
              onChange={(e) => {
                const v = Number(e.target.value);
                setCongestion(v);
                runCrossover({ nixl_congestion: v });
              }}
              className="w-full accent-amber-400"
            />
          </div>

          <div>
            <label className="block text-slate-400 mb-1 font-mono">Fabric Link: {linkBw} GB/s</label>
            <select
              value={linkBw}
              onChange={(e) => {
                const v = Number(e.target.value);
                setLinkBw(v);
                runCrossover({ link_bw_gbps: v });
              }}
              className="w-full bg-[#070a0f] border border-slate-700 rounded px-2.5 py-1.5 text-slate-100 font-mono"
            >
              <option value={900}>NVLink 4 (900 GB/s)</option>
              <option value={1800}>NVLink 5 B200 (1800 GB/s)</option>
              <option value={100}>Dual NDR400 RDMA (100 GB/s)</option>
              <option value={25}>Congested Pod IB (25 GB/s)</option>
            </select>
          </div>

          <div>
            <label className="block text-slate-400 mb-1 font-mono">Bypass L_threshold: {threshold}</label>
            <select
              value={threshold}
              onChange={(e) => {
                const v = Number(e.target.value);
                setThreshold(v);
                runCrossover({ l_threshold: v });
              }}
              className="w-full bg-[#070a0f] border border-slate-700 rounded px-2.5 py-1.5 text-slate-100 font-mono"
            >
              <option value={256}>256 tokens</option>
              <option value={512}>512 tokens (Default)</option>
              <option value={1024}>1,024 tokens</option>
            </select>
          </div>
        </div>

        {/* Quick Preset Scenario Buttons */}
        <div className="flex flex-wrap gap-2 mt-4 pt-3 border-t border-slate-800/80 text-xs">
          <button
            onClick={() => {
              setPromptLen(256);
              setMatchedLen(128);
              setCongestion(1.0);
              runCrossover({ prompt_len: 256, matched_prefix_len: 128, nixl_congestion: 1.0 });
            }}
            className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 font-mono"
          >
            Preset 1: Short Prompt (L_p=256 &lt; L_thresh → Aggregated Bypass)
          </button>
          <button
            onClick={() => {
              setPromptLen(8192);
              setMatchedLen(6144);
              setCongestion(1.2);
              setLinkBw(100);
              runCrossover({ prompt_len: 8192, matched_prefix_len: 6144, nixl_congestion: 1.2, link_bw_gbps: 100 });
            }}
            className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 font-mono"
          >
            Preset 2: Long Agentic Prefix (L_p=8192, 75% Hit → Disaggregated NIXL)
          </button>
          <button
            onClick={() => {
              setPromptLen(4096);
              setMatchedLen(0);
              setCongestion(140.0);
              setLinkBw(100);
              runCrossover({ prompt_len: 4096, matched_prefix_len: 0, nixl_congestion: 140.0, link_bw_gbps: 100 });
            }}
            className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 font-mono"
          >
            Preset 3: Severe RDMA Congestion (140x → Dynamic Local Bypass)
          </button>
        </div>
      </div>

      {/* Layer-Pipelined 4-Timestamp NIXL Waterfall + Crossover Curve */}
      {probe && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-5">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold text-white">
                FR-2.3 Layer-Pipelined NIXL Streaming (4 Canonical Timestamps)
              </h3>
              <span className="text-xs font-mono text-[#76b900]">
                Overlap: {probe.overlap_efficiency_pct.toFixed(1)}% • Exposed Stall: {probe.exposed_nixl_stall_ms.toFixed(2)} ms
              </span>
            </div>

            <div className="grid grid-cols-4 gap-2 mb-3 text-[11px] font-mono">
              <div className="bg-[#070a0f] p-2 rounded border border-slate-800">
                <div className="text-slate-400">t_producer_ready</div>
                <div className="text-white font-semibold">{probe.t_producer_ready_ms.toFixed(2)} ms</div>
              </div>
              <div className="bg-[#070a0f] p-2 rounded border border-slate-800">
                <div className="text-slate-400">t_transfer_start</div>
                <div className="text-[#00d4ff] font-semibold">{probe.t_transfer_start_ms.toFixed(2)} ms</div>
              </div>
              <div className="bg-[#070a0f] p-2 rounded border border-slate-800">
                <div className="text-slate-400">t_transfer_end</div>
                <div className="text-[#76b900] font-semibold">{probe.t_transfer_end_ms.toFixed(2)} ms</div>
              </div>
              <div className="bg-[#070a0f] p-2 rounded border border-slate-800">
                <div className="text-slate-400">t_consumer_ready</div>
                <div className="text-amber-300 font-semibold">{probe.t_consumer_ready_ms.toFixed(2)} ms</div>
              </div>
            </div>

            <svg viewBox="0 0 520 200" className="w-full h-48 bg-[#070a0f] rounded-lg border border-slate-800 p-2">
              {(() => {
                const stages = probe.layer_waterfall;
                const maxT = Math.max(
                  probe.t_transfer_end_ms,
                  probe.t_consumer_ready_ms,
                  ...stages.map((s) => s.t_transfer_end_ms),
                  10
                ) * 1.08;
                const scaleX = (t: number) => 68 + (t / maxT) * 430;
                const consX = scaleX(probe.t_consumer_ready_ms);

                return (
                  <>
                    {stages.map((st, idx) => {
                      const y = 18 + idx * 20;
                      const prodX = scaleX(st.t_producer_ready_ms);
                      const txS = scaleX(st.t_transfer_start_ms);
                      const txE = Math.max(txS + 3, scaleX(st.t_transfer_end_ms));
                      return (
                        <g key={st.layer_group}>
                          <text x={8} y={y + 9} fill="#94a3b8" fontSize="9" fontFamily="monospace">
                            {st.layer_group}
                          </text>
                          {/* Prefill compute bar up to producer_ready */}
                          <rect x={68} y={y} width={Math.max(3, prodX - 68)} height={6} rx={1.5} fill="#334155" />
                          {/* NIXL async RDMA transfer bar */}
                          <rect x={txS} y={y + 6} width={txE - txS} height={6} rx={1.5} fill="#00d4ff" />
                        </g>
                      );
                    })}
                    {/* Consumer Ready Vertical Line */}
                    <line x1={consX} y1={12} x2={consX} y2={178} stroke="#f59e0b" strokeWidth={1.6} strokeDasharray="4 3" />
                    <text x={Math.min(380, consX + 4)} y={192} fill="#f59e0b" fontSize="9.5" fontFamily="monospace">
                      ▲ t_consumer_ready ({probe.t_consumer_ready_ms.toFixed(1)}ms)
                    </text>
                    <text x={68} y={192} fill="#00d4ff" fontSize="9.5" fontFamily="monospace">
                      ■ Layer-Pipelined NIXL RDMA Stream
                    </text>
                  </>
                );
              })()}
            </svg>
          </div>

          {/* Disaggregation vs Aggregated Crossover Curve */}
          <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold text-white">
                  FR-2.2 Disaggregation Crossover Curve (Disagg NIXL vs In-Place Aggregated)
                </h3>
                <span className="text-xs font-mono text-slate-300">
                  Disagg: {probe.best_disagg_ttft_ms.toFixed(1)}ms | Agg: {probe.best_agg_ttft_ms.toFixed(1)}ms
                </span>
              </div>
              <p className="text-xs text-slate-400 mb-3">
                Router evaluates queueing + NIXL exposed stall across all 32 workers in {probe.routing_latency_us.toFixed(1)} µs (NFR-2.1 ≤ 500 µs).
              </p>
            </div>

            <svg viewBox="0 0 520 195" className="w-full h-48 bg-[#070a0f] rounded-lg border border-slate-800 p-2">
              {(() => {
                const pts = probe.crossover_curve;
                const maxY = Math.max(...pts.flatMap((p) => [p.disagg_ttft_ms, p.agg_ttft_ms]), 20) * 1.1;
                const xAt = (i: number) => 48 + (i / Math.max(1, pts.length - 1)) * 445;
                const yAt = (v: number) => 165 - (v / maxY) * 140;

                const disaggPath = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${xAt(i)},${yAt(p.disagg_ttft_ms)}`).join(' ');
                const aggPath = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${xAt(i)},${yAt(p.agg_ttft_ms)}`).join(' ');

                return (
                  <>
                    {[40, 85, 130, 165].map((gy) => (
                      <line key={gy} x1={44} y1={gy} x2={498} y2={gy} stroke="#1e293b" strokeDasharray="3 3" />
                    ))}
                    <path d={aggPath} fill="none" stroke="#f59e0b" strokeWidth={2.2} />
                    <path d={disaggPath} fill="none" stroke="#00d4ff" strokeWidth={2.4} />
                    {pts.map((p, i) => (
                      <g key={p.prompt_len}>
                        <circle cx={xAt(i)} cy={yAt(p.agg_ttft_ms)} r={3.2} fill="#f59e0b" />
                        <circle cx={xAt(i)} cy={yAt(p.disagg_ttft_ms)} r={3.5} fill="#00d4ff" />
                        <text x={xAt(i)} y={182} textAnchor="middle" fill="#94a3b8" fontSize="8.5" fontFamily="monospace">
                          {p.prompt_len >= 1024 ? `${p.prompt_len / 1024}K` : p.prompt_len}
                        </text>
                      </g>
                    ))}
                    <text x={52} y={20} fill="#00d4ff" fontSize="10" fontFamily="monospace">● Disaggregated NIXL TTFT (ms)</text>
                    <text x={280} y={20} fill="#f59e0b" fontSize="10" fontFamily="monospace">● In-Place Aggregated TTFT (ms)</text>
                  </>
                );
              })()}
            </svg>

            <div className="mt-3 bg-[#070a0f] border border-slate-800 rounded-lg p-2.5 text-[11px] font-mono text-slate-300 flex justify-between">
              <span>FR-2.1 Tokconomics Eviction @ 90% HBM:</span>
              <span className="text-[#76b900]">S_retain(b) = [FLOPs_recompute · P_reuse] / [Size_bytes · (1 + Δt_idle)]</span>
            </div>
          </div>
        </div>
      )}

      {/* 32-Worker Cluster Grid & NFR-2.3 Sub-10ms Chaos Failover */}
      {cluster && (
        <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-5">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <div>
              <h3 className="text-sm font-semibold text-white">
                32-GPU Disaggregated Pod Telemetry & NFR-2.3 Sub-10ms Chaos Failover Drill
              </h3>
              <p className="text-xs text-slate-400">
                Click any <span className="text-[#76b900] font-mono">DECODE W16..W31</span> node below to simulate heartbeat loss and trigger live Radix KV session re-anchoring!
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-slate-300 bg-[#070a0f] border border-slate-800 px-3 py-1.5 rounded">
                Cluster Power: <strong className="text-[#76b900]">{cluster.cluster_power_kw.toFixed(1)} kW</strong> (50ms HAL Poll)
              </span>
              {cluster.failed_workers.length > 0 && (
                <button
                  onClick={() => triggerFailover(16, true)}
                  disabled={failoverBusy}
                  className="px-3 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold"
                >
                  Restore Healthy Cluster
                </button>
              )}
            </div>
          </div>

          {cluster.last_failover && (
            <div className="mb-4 p-3 rounded-lg bg-[#76b900]/10 border border-[#76b900]/40 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
              <span className="text-[#76b900] font-semibold">
                ✓ NFR-2.3 FAILOVER VERIFIED: Decode W{cluster.last_failover.failed_decode_worker} → Re-anchored onto W{cluster.last_failover.backup_decode_worker} in {cluster.last_failover.failover_latency_ms} ms (≤ 10.0 ms Gate)
              </span>
              <span className="text-slate-200">
                Sessions Re-Anchored: {cluster.last_failover.sessions_reanchored} • KV Blocks Migrated: {cluster.last_failover.kv_blocks_migrated} • Lost Requests: {cluster.last_failover.sessions_lost}
              </span>
            </div>
          )}

          <div className="grid grid-cols-4 sm:grid-cols-8 lg:grid-cols-16 gap-2">
            {cluster.workers.map((w) => {
              const isDecode = w.pool === 'DECODE';
              return (
                <button
                  key={w.worker_id}
                  disabled={!isDecode || failoverBusy || !w.healthy}
                  onClick={() => isDecode && triggerFailover(w.worker_id)}
                  className={`p-2 rounded-lg border text-left font-mono transition ${
                    !w.healthy
                      ? 'bg-rose-950/50 border-rose-600/70 text-rose-300'
                      : isDecode
                      ? 'bg-[#070a0f] border-slate-800 hover:border-[#76b900] cursor-pointer'
                      : 'bg-[#070a0f]/80 border-slate-800/80 cursor-default'
                  }`}
                  title={
                    isDecode
                      ? `Click to inject heartbeat failure on Decode Worker W${w.worker_id}`
                      : `Prefill Worker W${w.worker_id}`
                  }
                >
                  <div className="flex justify-between items-center text-[10px]">
                    <span className={isDecode ? 'text-[#76b900] font-bold' : 'text-[#00d4ff] font-bold'}>
                      {isDecode ? 'D' : 'P'}{w.worker_id}
                    </span>
                    <span className="text-slate-400">{w.healthy ? `${w.power_watts.toFixed(0)}W` : 'DEAD'}</span>
                  </div>
                  <div className="w-full h-1.5 bg-slate-800 rounded mt-1.5 overflow-hidden">
                    <div
                      style={{ width: `${w.healthy ? w.radix_hbm_used_pct : 0}%` }}
                      className={`h-full ${w.radix_hbm_used_pct > 86 ? 'bg-amber-400' : isDecode ? 'bg-[#76b900]' : 'bg-[#00d4ff]'}`}
                    />
                  </div>
                  <div className="text-[9px] text-slate-400 mt-1">
                    {w.healthy ? `Q:${w.queue_wait_ms}ms` : 'RE-ANCHORED'}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
