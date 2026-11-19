import React, { useEffect, useState } from 'react';
import { DynamoBenchmark, ParetoResponse, ZMQEventEnvelope } from '../types';

interface Props {
  initialBench: DynamoBenchmark;
}

export const TokconomicsParetoView: React.FC<Props> = ({ initialBench }) => {
  const [bench, setBench] = useState<DynamoBenchmark>(initialBench);
  const [pareto, setPareto] = useState<ParetoResponse | null>(null);
  const [events, setEvents] = useState<ZMQEventEnvelope[]>([]);
  const [zmqEndpoint, setZmqEndpoint] = useState<string>('');
  const [promText, setPromText] = useState<string>('');
  const [runningReplay, setRunningReplay] = useState<boolean>(false);
  const [loadScale, setLoadScale] = useState<number>(1.0);

  const loadTelemetry = async () => {
    const [pRes, eRes, mRes] = await Promise.all([
      fetch('/api/v1/pareto'),
      fetch('/api/v1/kv-events?limit=12'),
      fetch('/metrics'),
    ]);
    if (pRes.ok) setPareto(await pRes.json());
    if (eRes.ok) {
      const ej = await eRes.json();
      setEvents(ej.events || []);
      setZmqEndpoint(ej.endpoint || '');
    }
    if (mRes.ok) setPromText(await mRes.text());
  };

  useEffect(() => {
    loadTelemetry();
  }, []);

  const triggerReplay = async () => {
    setRunningReplay(true);
    try {
      const res = await fetch('/api/v1/dynamoecon/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_requests: 640,
          num_workers: 32,
          hw_profile_code: 1,
          arrival_rate_scale: loadScale,
          nixl_congestion: 1.2,
        }),
      });
      if (res.ok) {
        setBench(await res.json());
        await loadTelemetry();
      }
    } finally {
      setRunningReplay(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header & Live Replay Trigger */}
      <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-400" />
            Real-Time Tokconomics Monetization, aiperf Pareto Frontier & Dynamo KV-Events
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Multi-Turn Agentic Workload Trace (75% System/Tool Prefix Sharing, Bursty Arrivals) • ZeroMQ {zmqEndpoint}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="text-xs font-mono">
            <span className="text-slate-400 mr-2">Offered Load: {loadScale.toFixed(2)}x</span>
            <input
              type="range"
              min={0.5}
              max={1.8}
              step={0.1}
              value={loadScale}
              onChange={(e) => setLoadScale(Number(e.target.value))}
              className="accent-[#76b900] align-middle"
            />
          </div>
          <button
            onClick={triggerReplay}
            disabled={runningReplay}
            className="px-4 py-2 rounded-lg bg-[#76b900] hover:bg-[#8be200] text-black font-semibold text-xs transition disabled:opacity-50"
          >
            {runningReplay ? 'REPLAYING 640 AGENTIC TURNS...' : 'RUN AIPERF REPLAY TRACE'}
          </button>
        </div>
      </div>

      {/* 4 Monetization & SLA Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400">Multi-Tier SLA Goodput (NFR-2.2)</div>
          <div className="text-2xl font-bold font-mono text-[#76b900] mt-1">
            {bench.adaptive_sla_goodput_pct.toFixed(1)}%
            <span className="text-xs text-emerald-300 ml-2">+{bench.goodput_improvement_pct.toFixed(1)}% Lift</span>
          </div>
          <div className="text-xs text-slate-400 font-mono mt-1">
            Static RR Baseline: {bench.baseline_sla_goodput_pct.toFixed(1)}%
          </div>
        </div>

        <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400">Cost / 1M Goodput Tokens (NFR-2.2)</div>
          <div className="text-2xl font-bold font-mono text-[#00d4ff] mt-1">
            ${bench.adaptive_cost_per_1m_usd.toFixed(3)}
            <span className="text-xs text-emerald-300 ml-2">-{bench.cost_reduction_pct.toFixed(1)}%</span>
          </div>
          <div className="text-xs text-slate-400 font-mono mt-1">
            Static RR Baseline: ${bench.baseline_cost_per_1m_usd.toFixed(3)} / 1M
          </div>
        </div>

        <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400">Effective Prefix Reuse</div>
          <div className="text-2xl font-bold font-mono text-white mt-1">
            {bench.adaptive_prefix_reuse_pct.toFixed(1)}%
          </div>
          <div className="text-xs text-slate-400 font-mono mt-1">
            Baseline LRU Reuse: {bench.baseline_prefix_reuse_pct.toFixed(1)}%
          </div>
        </div>

        <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400">Energy Efficiency & Routing</div>
          <div className="text-2xl font-bold font-mono text-amber-300 mt-1">
            {bench.adaptive_tokens_per_watt.toFixed(3)} <span className="text-sm">tok/J</span>
          </div>
          <div className="text-xs text-slate-400 font-mono mt-1">
            Bypasses: {bench.adaptive_aggregated_bypasses} • Disagg: {bench.adaptive_disaggregated_routes}
          </div>
        </div>
      </div>

      {/* Pareto Frontier SVG + Multi-Tier SLA Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-[#0e141f] border border-slate-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-semibold text-white">
              aiperf Latency (P99 TTFT ms) vs. SLA Goodput Throughput (kTok/s) Pareto Frontier
            </h3>
            <span className="text-xs font-mono text-slate-400">{pareto?.hardware_profile}</span>
          </div>

          {pareto && (
            <svg viewBox="0 0 600 230" className="w-full h-56 bg-[#070a0f] rounded-lg border border-slate-800 p-2">
              {(() => {
                const allPts = pareto.series.flatMap((s) => s.points);
                const maxX = Math.max(...allPts.map((p) => p.goodput_ktok_per_s), 50) * 1.1;
                const maxY = Math.min(900, Math.max(...allPts.map((p) => p.p99_ttft_ms), 250) * 1.08);

                const sx = (x: number) => 52 + (x / maxX) * 520;
                const sy = (y: number) => 195 - (Math.min(y, maxY) / maxY) * 165;

                const colors: Record<string, string> = {
                  turbokv_dynamoecon: '#76b900',
                  bf16_dynamoecon: '#00d4ff',
                  baseline_static_rr: '#f43f5e',
                };

                const tier2LineY = sy(400);

                return (
                  <>
                    {[45, 95, 145, 195].map((gy) => (
                      <line key={gy} x1={48} y1={gy} x2={578} y2={gy} stroke="#1e293b" strokeDasharray="3 3" />
                    ))}
                    {tier2LineY > 25 && tier2LineY < 195 && (
                      <>
                        <line x1={48} y1={tier2LineY} x2={578} y2={tier2LineY} stroke="#f59e0b" strokeDasharray="5 4" />
                        <text x={430} y={tier2LineY - 4} fill="#f59e0b" fontSize="8.5" fontFamily="monospace">
                          Tier-2 SLA Ceiling (400ms)
                        </text>
                      </>
                    )}
                    {pareto.series.map((ser) => {
                      const col = colors[ser.id] || '#94a3b8';
                      const d = ser.points
                        .map((pt, i) => `${i === 0 ? 'M' : 'L'}${sx(pt.goodput_ktok_per_s)},${sy(pt.p99_ttft_ms)}`)
                        .join(' ');
                      return (
                        <g key={ser.id}>
                          <path d={d} fill="none" stroke={col} strokeWidth={2.5} />
                          {ser.points.map((pt, idx) => (
                            <circle
                              key={idx}
                              cx={sx(pt.goodput_ktok_per_s)}
                              cy={sy(pt.p99_ttft_ms)}
                              r={3.8}
                              fill={col}
                            />
                          ))}
                        </g>
                      );
                    })}
                    <text x={55} y={20} fill="#76b900" fontSize="9.5" fontFamily="monospace">● DynamoEcon + TurboKV-FP4</text>
                    <text x={245} y={20} fill="#00d4ff" fontSize="9.5" fontFamily="monospace">● DynamoEcon + BF16</text>
                    <text x={410} y={20} fill="#f43f5e" fontSize="9.5" fontFamily="monospace">● Static RR + LRU</text>
                    <text x={240} y={218} fill="#94a3b8" fontSize="9.5" fontFamily="monospace">
                      SLA-Compliant Goodput Throughput (kTok/s) →
                    </text>
                    <text x={8} y={32} fill="#94a3b8" fontSize="9" fontFamily="monospace">P99 ms</text>
                  </>
                );
              })()}
            </svg>
          )}
        </div>

        {/* Tier 1 / 2 / 3 SLA Monetization Table */}
        <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-5 space-y-4">
          <h3 className="text-sm font-semibold text-white">FR-2.4 Multi-Tier SLA Compliance</h3>
          <div className="space-y-3 text-xs font-mono">
            <div className="p-3 rounded-lg bg-[#070a0f] border border-slate-800">
              <div className="flex justify-between">
                <span className="text-white font-semibold">Tier 1: Interactive Reasoning</span>
                <span className="text-[#76b900]">{bench.adaptive_tier1_goodput_pct.toFixed(1)}%</span>
              </div>
              <div className="text-slate-400 text-[11px] mt-0.5">Contract: TTFT ≤ 150ms, ITL ≤ 20ms</div>
            </div>

            <div className="p-3 rounded-lg bg-[#070a0f] border border-slate-800">
              <div className="flex justify-between">
                <span className="text-white font-semibold">Tier 2: Agentic Tool-Use</span>
                <span className="text-[#00d4ff]">{bench.adaptive_tier2_goodput_pct.toFixed(1)}%</span>
              </div>
              <div className="text-slate-400 text-[11px] mt-0.5">Contract: TTFT ≤ 400ms, ITL ≤ 40ms</div>
            </div>

            <div className="p-3 rounded-lg bg-[#070a0f] border border-slate-800">
              <div className="flex justify-between">
                <span className="text-white font-semibold">Tier 3: Batch Synthesis</span>
                <span className="text-emerald-300">{bench.adaptive_tier3_goodput_pct.toFixed(1)}%</span>
              </div>
              <div className="text-slate-400 text-[11px] mt-0.5">Throughput-optimized, best-effort</div>
            </div>

            <div className="p-3 rounded-lg bg-[#070a0f] border border-slate-800 flex justify-between">
              <span className="text-slate-400">Mean TTFT / ITL:</span>
              <span className="text-white">
                {bench.adaptive_avg_ttft_ms.toFixed(1)} ms / {bench.adaptive_avg_itl_ms.toFixed(1)} ms
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ZeroMQ KV-Events Feed + Prometheus /metrics Scrape */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-white">
              ZeroMQ `--kv-events-config` Pub/Sub Stream
            </h3>
            <span className="text-[11px] font-mono text-[#00d4ff]">{zmqEndpoint}</span>
          </div>
          <div className="bg-[#070a0f] border border-slate-800 rounded-lg p-3 max-h-48 overflow-y-auto space-y-1.5 font-mono text-[11px]">
            {events.map((ev) => (
              <div key={ev.event_id} className="flex items-center justify-between py-1 border-b border-slate-900">
                <span className={ev.event.type === 'BlockStored' ? 'text-[#76b900]' : 'text-amber-400'}>
                  [{ev.event.type}]
                </span>
                <span className="text-slate-300">W{ev.worker_id} • {ev.event.block_hashes.join(', ')}</span>
                <span className="text-slate-400">S_ret={ev.event.tokconomics_s_retain.toFixed(1)}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-white">Live Prometheus `/metrics` Scrape</h3>
            <span className="text-[11px] font-mono text-[#76b900]">econ_telemetryd + FastAPI Exporter</span>
          </div>
          <pre className="bg-[#070a0f] border border-slate-800 rounded-lg p-3 max-h-48 overflow-y-auto font-mono text-[10.5px] text-slate-300 leading-relaxed">
            {promText}
          </pre>
        </div>
      </div>
    </div>
  );
};
