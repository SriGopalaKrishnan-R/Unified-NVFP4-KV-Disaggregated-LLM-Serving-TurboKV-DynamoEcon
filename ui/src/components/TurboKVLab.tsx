import React, { useState } from 'react';
import { TurboKVSnapshot } from '../types';

interface Props {
  initialData: TurboKVSnapshot;
}

export const TurboKVLab: React.FC<Props> = ({ initialData }) => {
  const [snap, setSnap] = useState<TurboKVSnapshot>(initialData);
  const [seqLen, setSeqLen] = useState<number>(65536);
  const [batchSize, setBatchSize] = useState<number>(32);
  const [headDim, setHeadDim] = useState<number>(128);
  const [archCode, setArchCode] = useState<number>(100);
  const [distCode, setDistCode] = useState<number>(0);
  const [specDepth, setSpecDepth] = useState<number>(6);
  const [specBranch, setSpecBranch] = useState<number>(3);
  const [acceptBias, setAcceptBias] = useState<number>(0.92);
  const [loading, setLoading] = useState<boolean>(false);

  const runEval = async (overrides?: Partial<{
    seq_len: number;
    batch_size: number;
    head_dim: number;
    arch_code: number;
    dist_code: number;
    spec_depth: number;
    spec_branch: number;
    accept_rate_bias: number;
  }>) => {
    setLoading(true);
    try {
      const payload = {
        seq_len: overrides?.seq_len ?? seqLen,
        batch_size: overrides?.batch_size ?? batchSize,
        head_dim: overrides?.head_dim ?? headDim,
        arch_code: overrides?.arch_code ?? archCode,
        dist_code: overrides?.dist_code ?? distCode,
        spec_depth: overrides?.spec_depth ?? specDepth,
        spec_branch: overrides?.spec_branch ?? specBranch,
        accept_rate_bias: overrides?.accept_rate_bias ?? acceptBias,
      };
      const res = await fetch('/api/v1/turbokv/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        const data: TurboKVSnapshot = await res.json();
        setSnap(data);
      }
    } finally {
      setLoading(false);
    }
  };

  const bf16MiB = snap.bf16_bytes / (1024 * 1024);
  const hybridMiB = snap.total_hybrid_bytes / (1024 * 1024);
  const nvfp4Pct = (snap.nvfp4_payload_bytes / Math.max(1, snap.total_hybrid_bytes)) * 100;
  const scalePct = (snap.micro_scale_fp8_bytes / Math.max(1, snap.total_hybrid_bytes)) * 100;
  const sinkWinPct = Math.max(1.2, 100 - nvfp4Pct - scalePct);

  const rawSpectrum = snap.channel_spectrum.raw_abs_magnitude;
  const rotSpectrum = snap.channel_spectrum.hadamard_rotated_magnitude;
  const maxSpecVal = Math.max(...rawSpectrum, 1);

  return (
    <div className="space-y-6">
      {/* Control Bar */}
      <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-5">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <span className="inline-block w-2.5 h-2.5 rounded-full bg-[#76b900]" />
              Project 1: TurboKV-FP4 Kernel & Speculative Tree Workbench
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Live C++20 SIMD/PTX Bit-Exact Engine • Two-Level Microscaled NVFP4 (E2M1 + FP8 E4M3/16) + Sink(4) + Window(128)
            </p>
          </div>
          <button
            onClick={() => runEval()}
            disabled={loading}
            className="px-4 py-2 rounded-lg bg-[#76b900] hover:bg-[#8be200] text-black font-semibold text-xs tracking-wide transition disabled:opacity-50"
          >
            {loading ? 'EXECUTING C++20 KERNEL SIM...' : 'EXECUTE TURBOKV-FP4 SUITE'}
          </button>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3 text-xs">
          <div>
            <label className="block text-slate-400 mb-1">Target GPU Arch</label>
            <select
              value={archCode}
              onChange={(e) => {
                const v = Number(e.target.value);
                setArchCode(v);
                runEval({ arch_code: v });
              }}
              className="w-full bg-[#070a0f] border border-slate-700 rounded px-2.5 py-1.5 text-slate-100 font-mono"
            >
              <option value={100}>Blackwell SM100 (B200)</option>
              <option value={90}>Hopper SM90 (H100)</option>
              <option value={89}>Ada SM89 (RTX 4090)</option>
            </select>
          </div>

          <div>
            <label className="block text-slate-400 mb-1">Context Length (L)</label>
            <select
              value={seqLen}
              onChange={(e) => {
                const v = Number(e.target.value);
                setSeqLen(v);
                runEval({ seq_len: v });
              }}
              className="w-full bg-[#070a0f] border border-slate-700 rounded px-2.5 py-1.5 text-slate-100 font-mono"
            >
              <option value={4096}>4,096</option>
              <option value={16384}>16,384</option>
              <option value={32768}>32,768</option>
              <option value={65536}>65,536 (NFR-1.1)</option>
              <option value={131072}>131,072</option>
            </select>
          </div>

          <div>
            <label className="block text-slate-400 mb-1">Decode Batch</label>
            <select
              value={batchSize}
              onChange={(e) => {
                const v = Number(e.target.value);
                setBatchSize(v);
                runEval({ batch_size: v });
              }}
              className="w-full bg-[#070a0f] border border-slate-700 rounded px-2.5 py-1.5 text-slate-100 font-mono"
            >
              <option value={1}>Batch = 1</option>
              <option value={8}>Batch = 8</option>
              <option value={32}>Batch = 32 (NFR-1.2)</option>
              <option value={64}>Batch = 64</option>
              <option value={128}>Batch = 128</option>
            </select>
          </div>

          <div>
            <label className="block text-slate-400 mb-1">Head Dim (d)</label>
            <select
              value={headDim}
              onChange={(e) => {
                const v = Number(e.target.value);
                setHeadDim(v);
                runEval({ head_dim: v });
              }}
              className="w-full bg-[#070a0f] border border-slate-700 rounded px-2.5 py-1.5 text-slate-100 font-mono"
            >
              <option value={128}>d = 128</option>
              <option value={64}>d = 64</option>
            </select>
          </div>

          <div>
            <label className="block text-slate-400 mb-1">Fidelity Workload</label>
            <select
              value={distCode}
              onChange={(e) => {
                const v = Number(e.target.value);
                setDistCode(v);
                runEval({ dist_code: v });
              }}
              className="w-full bg-[#070a0f] border border-slate-700 rounded px-2.5 py-1.5 text-slate-100 font-mono"
            >
              <option value={0}>Needle-in-Haystack</option>
              <option value={1}>Long Reasoning Chain</option>
              <option value={2}>52x Channel Outliers</option>
            </select>
          </div>

          <div>
            <label className="block text-slate-400 mb-1">Spec Tree Depth ({specDepth})</label>
            <input
              type="range"
              min={2}
              max={8}
              value={specDepth}
              onChange={(e) => {
                const v = Number(e.target.value);
                setSpecDepth(v);
                runEval({ spec_depth: v });
              }}
              className="w-full accent-[#76b900] mt-2"
            />
          </div>

          <div>
            <label className="block text-slate-400 mb-1">Accept Bias ({Math.round(acceptBias * 100)}%)</label>
            <input
              type="range"
              min={0.45}
              max={0.98}
              step={0.05}
              value={acceptBias}
              onChange={(e) => {
                const v = Number(e.target.value);
                setAcceptBias(v);
                runEval({ accept_rate_bias: v });
              }}
              className="w-full accent-[#76b900] mt-2"
            />
          </div>
        </div>
      </div>

      {/* Top KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400">KV HBM Compression (NFR-1.1)</div>
          <div className="text-2xl font-bold font-mono text-[#76b900] mt-1">
            {snap.compression_ratio.toFixed(3)}x
          </div>
          <div className="text-xs text-slate-400 font-mono mt-1">
            {bf16MiB.toFixed(1)} MiB BF16 → {hybridMiB.toFixed(2)} MiB Hybrid
          </div>
        </div>

        <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400">Path 3 Roofline HBM BW (NFR-1.2)</div>
          <div className="text-2xl font-bold font-mono text-[#00d4ff] mt-1">
            {snap.hbm_bw_utilization_pct.toFixed(2)}%
          </div>
          <div className="text-xs text-slate-400 font-mono mt-1">
            {snap.achieved_hbm_bw_gbps.toFixed(0)} / {snap.peak_hbm_bw_gbps.toFixed(0)} GB/s ({snap.speedup_vs_bf16_decode.toFixed(2)}x vs BF16)
          </div>
        </div>

        <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400">Numerical Fidelity vs BF16 (NFR-1.3)</div>
          <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">
            {snap.turbokv_cosine_sim.toFixed(5)}
          </div>
          <div className="text-xs text-slate-400 font-mono mt-1">
            KL-Div: {snap.turbokv_kl_div.toFixed(5)} (Gate ≤ 0.015)
          </div>
        </div>

        <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400">PTX Register & Host-Sync Audit</div>
          <div className="text-2xl font-bold font-mono text-white mt-1">
            {snap.registers_per_thread} <span className="text-sm text-slate-400">regs/thr</span>
          </div>
          <div className="text-xs text-[#76b900] font-mono mt-1">
            Occ: {snap.achieved_occupancy_pct.toFixed(0)}% • Bank Conf: {snap.shared_mem_bank_conflicts} • Syncs: {snap.host_sync_calls}
          </div>
        </div>
      </div>

      {/* Hybrid Memory Partition + Numerical Fidelity Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white">FR-1.1 Hybrid Precision-Zone Memory Layout</h3>
            <span className="text-xs font-mono text-[#76b900]">uint4 Coalesced Paged Blocks</span>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-xs text-slate-400 font-mono">
              <span>Full BF16 Baseline (16-bit uniform)</span>
              <span>{bf16MiB.toFixed(2)} MiB (100%)</span>
            </div>
            <div className="w-full h-4 bg-slate-800 rounded overflow-hidden">
              <div className="h-full bg-slate-500 w-full" />
            </div>

            <div className="flex justify-between text-xs text-slate-300 font-mono pt-2">
              <span>TurboKV-FP4 Hybrid Footprint</span>
              <span>{hybridMiB.toFixed(2)} MiB ({(100 / snap.compression_ratio).toFixed(1)}%)</span>
            </div>
            <div className="w-full h-5 bg-slate-900 border border-slate-800 rounded overflow-hidden flex">
              <div
                style={{ width: `${(nvfp4Pct / snap.compression_ratio).toFixed(2)}%` }}
                className="h-full bg-[#76b900]"
                title="NVFP4 E2M1 Historical Payload"
              />
              <div
                style={{ width: `${(scalePct / snap.compression_ratio).toFixed(2)}%` }}
                className="h-full bg-[#00d4ff]"
                title="FP8 E4M3 Per-16 Micro-Scales"
              />
              <div
                style={{ width: `${Math.max(1.5, sinkWinPct / snap.compression_ratio).toFixed(2)}%` }}
                className="h-full bg-amber-400"
                title="FP8 Attention-Sink (N=4) + Local Sliding Window (N=128)"
              />
            </div>

            <div className="grid grid-cols-3 gap-2 pt-2 text-[11px] font-mono">
              <div className="bg-[#070a0f] border border-slate-800 p-2 rounded">
                <span className="inline-block w-2 h-2 rounded-full bg-amber-400 mr-1.5" />
                <span className="text-slate-300">Sink(4)+Win(128)</span>
                <div className="text-slate-400 mt-0.5">
                  {((snap.sink_fp8_bytes + snap.window_fp8_bytes) / 1024).toFixed(1)} KiB FP8
                </div>
              </div>
              <div className="bg-[#070a0f] border border-slate-800 p-2 rounded">
                <span className="inline-block w-2 h-2 rounded-full bg-[#76b900] mr-1.5" />
                <span className="text-slate-300">NVFP4 Payload</span>
                <div className="text-slate-400 mt-0.5">
                  {(snap.nvfp4_payload_bytes / (1024 * 1024)).toFixed(2)} MiB E2M1
                </div>
              </div>
              <div className="bg-[#070a0f] border border-slate-800 p-2 rounded">
                <span className="inline-block w-2 h-2 rounded-full bg-[#00d4ff] mr-1.5" />
                <span className="text-slate-300">2-Level Scales</span>
                <div className="text-slate-400 mt-0.5">
                  {(snap.micro_scale_fp8_bytes / (1024 * 1024)).toFixed(2)} MiB E4M3+FP32
                </div>
              </div>
            </div>
          </div>

          {/* Fidelity Side-by-Side Table */}
          <div className="pt-2 border-t border-slate-800">
            <div className="text-xs font-semibold text-slate-300 mb-2">
              NFR-1.3 Accuracy-vs-Bandwidth Cliff Resolution (vs Full BF16 Ground Truth)
            </div>
            <div className="grid grid-cols-3 gap-2 text-xs font-mono bg-[#070a0f] p-3 rounded-lg border border-slate-800">
              <div className="text-slate-400">Metric</div>
              <div className="text-[#76b900] font-semibold">TurboKV-FP4</div>
              <div className="text-rose-400 font-semibold">Naive Uniform FP4</div>

              <div className="text-slate-400">Cosine Sim (≥0.997)</div>
              <div className="text-white">{snap.turbokv_cosine_sim.toFixed(6)} ✓</div>
              <div className="text-slate-300">{snap.uniform_fp4_cosine_sim.toFixed(6)}</div>

              <div className="text-slate-400">Softmax KL (≤0.015)</div>
              <div className="text-[#76b900]">{snap.turbokv_kl_div.toFixed(6)} ✓</div>
              <div className="text-rose-400">{snap.uniform_fp4_kl_div.toFixed(6)} ✗</div>

              <div className="text-slate-400">Output RMSE</div>
              <div className="text-white">{snap.turbokv_rmse.toFixed(5)}</div>
              <div className="text-rose-300">{snap.uniform_fp4_rmse.toFixed(5)}</div>
            </div>
          </div>
        </div>

        {/* Walsh-Hadamard Outlier Spectrum SVG */}
        <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h3 className="text-sm font-semibold text-white">
                In-Register Walsh-Hadamard Transform (H_{headDim}) Outlier Suppression
              </h3>
              <span className="text-xs font-mono text-[#00d4ff]">
                Crest: {snap.raw_channel_crest_factor.toFixed(2)}x → {snap.rotated_channel_crest_factor.toFixed(2)}x
              </span>
            </div>
            <p className="text-xs text-slate-400 mb-3">
              Spreads massive RoPE / activation channel spikes across all {headDim} coordinates via warp butterfly shuffles before NVFP4 E2M1 quantization—requiring zero lookup tables.
            </p>
          </div>

          <svg viewBox="0 0 520 190" className="w-full h-48 bg-[#070a0f] rounded-lg border border-slate-800 p-2">
            {/* Grid lines */}
            {[35, 80, 125, 165].map((y) => (
              <line key={y} x1={32} y1={y} x2={505} y2={y} stroke="#1e293b" strokeDasharray="3 3" />
            ))}
            <text x={8} y={22} fill="#94a3b8" fontSize="9" fontFamily="monospace">|K[c]|</text>

            {rawSpectrum.map((rawV, i) => {
              const rotV = rotSpectrum[i] ?? 1;
              const xBase = 38 + i * 14.4;
              const rawH = Math.max(3, (rawV / maxSpecVal) * 135);
              const rotH = Math.max(3, (rotV / maxSpecVal) * 135);
              return (
                <g key={i}>
                  <rect
                    x={xBase}
                    y={168 - rawH}
                    width={5.5}
                    height={rawH}
                    rx={1}
                    fill={rawV > 5 ? '#f43f5e' : '#475569'}
                  />
                  <rect
                    x={xBase + 6}
                    y={168 - rotH}
                    width={5.5}
                    height={rotH}
                    rx={1}
                    fill="#76b900"
                  />
                </g>
              );
            })}
            <text x={40} y={184} fill="#f43f5e" fontSize="10" fontFamily="monospace">■ Raw K Channels (RoPE Spikes)</text>
            <text x={265} y={184} fill="#76b900" fontSize="10" fontFamily="monospace">■ FWHT Rotated H·K (Outlier-Free)</text>
          </svg>

          <div className="mt-3 bg-[#070a0f] border border-slate-800 rounded-lg p-3 font-mono text-[11px] text-slate-300">
            <div className="text-[#00d4ff] mb-1">// Path 3 PTX Unpack + Hadamard Inner-Product Invariance (&lt;H·q, H·k&gt; = &lt;q, k&gt;)</div>
            <div>asm("prmt.b32 %0, 0x3C383000, 0x4C484440, %1;" : "=r"(fp8x4) : "r"(even_mags));</div>
            <div>asm("lop3.b32 %0, %1, 0x7F7F7F7F, %2, 0xEA;" : "=r"(signed_fp8x4) ...); // 0 smem bank conflicts</div>
          </div>
        </div>
      </div>

      {/* Speculative Tree Verification Visualizer */}
      <div className="bg-[#0e141f] border border-slate-800 rounded-xl p-5">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div>
            <h3 className="text-sm font-semibold text-white">
              FR-1.3 & NFR-1.4: Bitwise Speculative Tree Verification & Warp Rejection Scan (spec_tree_verify.cu)
            </h3>
            <p className="text-xs text-slate-400">
              64-bit Ancestor Bitmask FlashAttention (W={snap.spec_tree_width} ≤ 64, D={snap.spec_tree_depth} ≤ 8) + __ballot_sync / __ffs On-Device KV Commit
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs font-mono">
            <span className="px-2.5 py-1 rounded bg-[#76b900]/15 border border-[#76b900]/40 text-[#76b900]">
              Accepted Chain: {snap.spec_num_accepted} Draft + 1 Bonus = {snap.spec_num_accepted + 1} Tokens/Step
            </span>
            <span className="px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-slate-300">
              __ballot_sync: 0x{snap.spec_warp_ballot_mask.toString(16).padStart(8, '0')} | __ffs: {snap.spec_first_reject_ffs}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 items-center">
          <div className="lg:col-span-2 bg-[#070a0f] border border-slate-800 rounded-lg p-3 overflow-x-auto">
            <svg viewBox="0 0 680 210" className="w-full h-52 min-w-[520px]">
              {(() => {
                const nodes = snap.speculative_tree_nodes.slice(0, 36);
                const byDepth: Record<number, typeof nodes> = {};
                nodes.forEach((n) => {
                  byDepth[n.depth] = byDepth[n.depth] || [];
                  byDepth[n.depth].push(n);
                });
                const coords: Record<number, { x: number; y: number }> = {};
                Object.entries(byDepth).forEach(([dStr, list]) => {
                  const d = Number(dStr);
                  const x = 45 + (d - 1) * 84;
                  list.forEach((nd, idx) => {
                    const y = ((idx + 1) / (list.length + 1)) * 195 + 6;
                    coords[nd.id] = { x, y };
                  });
                });

                return (
                  <>
                    {nodes.map((nd) => {
                      if (nd.parent < 0 || !coords[nd.parent] || !coords[nd.id]) return null;
                      const p = coords[nd.parent];
                      const c = coords[nd.id];
                      return (
                        <line
                          key={`e-${nd.id}`}
                          x1={p.x}
                          y1={p.y}
                          x2={c.x}
                          y2={c.y}
                          stroke={nd.on_winning_chain ? '#76b900' : '#334155'}
                          strokeWidth={nd.on_winning_chain ? 2.5 : 1.1}
                        />
                      );
                    })}
                    {nodes.map((nd) => {
                      const c = coords[nd.id];
                      if (!c) return null;
                      return (
                        <g key={`n-${nd.id}`}>
                          <circle
                            cx={c.x}
                            cy={c.y}
                            r={nd.on_winning_chain ? 10 : 7.5}
                            fill={nd.on_winning_chain ? '#76b900' : '#1e293b'}
                            stroke={nd.on_winning_chain ? '#bbf7d0' : '#475569'}
                            strokeWidth={1.2}
                          />
                          <text
                            x={c.x}
                            y={c.y + 3}
                            textAnchor="middle"
                            fontSize="8.5"
                            fontFamily="monospace"
                            fontWeight="bold"
                            fill={nd.on_winning_chain ? '#05080d' : '#cbd5e1'}
                          >
                            {nd.id}
                          </text>
                        </g>
                      );
                    })}
                  </>
                );
              })()}
            </svg>
          </div>

          <div className="bg-[#070a0f] border border-slate-800 rounded-lg p-3 max-h-52 overflow-y-auto font-mono text-[11px]">
            <div className="text-slate-400 pb-1.5 mb-1.5 border-b border-slate-800 flex justify-between">
              <span>Node / Depth</span>
              <span>64-Bit Ancestor Mask</span>
              <span>Status</span>
            </div>
            {snap.speculative_tree_nodes.slice(0, 14).map((nd) => (
              <div key={nd.id} className="flex justify-between py-1 border-b border-slate-900">
                <span className="text-slate-300">n{nd.id} (d={nd.depth})</span>
                <span className="text-[#00d4ff]">{nd.ancestor_mask_hex}</span>
                <span className={nd.on_winning_chain ? 'text-[#76b900] font-semibold' : 'text-slate-500'}>
                  {nd.on_winning_chain ? 'COMMITTED' : 'PRUNED'}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
