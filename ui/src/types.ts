export interface VerificationGate {
  id: string;
  project: string;
  title: string;
  target: string;
  measured: string;
  numeric_value: number;
  threshold: number;
  passed: boolean;
}

export interface SpecTreeNode {
  id: number;
  parent: number;
  depth: number;
  ancestor_mask_hex: string;
  accepted: boolean;
  on_winning_chain: boolean;
}

export interface TurboKVSnapshot {
  bf16_bytes: number;
  total_hybrid_bytes: number;
  sink_fp8_bytes: number;
  window_fp8_bytes: number;
  nvfp4_payload_bytes: number;
  micro_scale_fp8_bytes: number;
  compression_ratio: number;
  peak_hbm_bw_gbps: number;
  achieved_hbm_bw_gbps: number;
  hbm_bw_utilization_pct: number;
  estimated_kernel_us: number;
  speedup_vs_bf16_decode: number;
  registers_per_thread: number;
  achieved_occupancy_pct: number;
  shared_mem_bank_conflicts: number;
  turbokv_cosine_sim: number;
  turbokv_kl_div: number;
  turbokv_rmse: number;
  uniform_fp4_cosine_sim: number;
  uniform_fp4_kl_div: number;
  uniform_fp4_rmse: number;
  raw_channel_crest_factor: number;
  rotated_channel_crest_factor: number;
  spec_tree_width: number;
  spec_tree_depth: number;
  spec_num_accepted: number;
  spec_bonus_token_id: number;
  spec_warp_ballot_mask: number;
  spec_first_reject_ffs: number;
  host_sync_calls: number;
  arch_label: string;
  channel_spectrum: {
    raw_abs_magnitude: number[];
    hadamard_rotated_magnitude: number[];
  };
  speculative_tree_nodes: SpecTreeNode[];
}

export interface DynamoBenchmark {
  scheduler_p50_us: number;
  scheduler_p99_us: number;
  scheduler_throughput_rps: number;
  adaptive_sla_goodput_pct: number;
  adaptive_tier1_goodput_pct: number;
  adaptive_tier2_goodput_pct: number;
  adaptive_tier3_goodput_pct: number;
  adaptive_avg_ttft_ms: number;
  adaptive_p99_ttft_ms: number;
  adaptive_avg_itl_ms: number;
  adaptive_nixl_stall_ms: number;
  adaptive_tokens_per_watt: number;
  adaptive_prefix_reuse_pct: number;
  adaptive_cost_per_1m_usd: number;
  adaptive_cluster_watts: number;
  adaptive_aggregated_bypasses: number;
  adaptive_disaggregated_routes: number;
  baseline_sla_goodput_pct: number;
  baseline_avg_ttft_ms: number;
  baseline_p99_ttft_ms: number;
  baseline_avg_itl_ms: number;
  baseline_nixl_stall_ms: number;
  baseline_tokens_per_watt: number;
  baseline_prefix_reuse_pct: number;
  baseline_cost_per_1m_usd: number;
  goodput_improvement_pct: number;
  cost_reduction_pct: number;
  failover_latency_ms: number;
  failover_sessions_reanchored: number;
  failover_sessions_lost: number;
  failover_kv_blocks_migrated: number;
}

export interface VerificationResponse {
  document_id: string;
  all_gates_passed: boolean;
  gates: VerificationGate[];
  turbokv_b200: TurboKVSnapshot;
  turbokv_h100: TurboKVSnapshot;
  dynamoecon_benchmark: DynamoBenchmark;
}

export interface CrossoverWaterfallStage {
  layer_group: string;
  t_producer_ready_ms: number;
  t_transfer_start_ms: number;
  t_transfer_end_ms: number;
  t_consumer_ready_ms: number;
}

export interface CrossoverCurvePoint {
  prompt_len: number;
  disagg_ttft_ms: number;
  agg_ttft_ms: number;
  selected_mode: string;
}

export interface CrossoverResponse {
  selected_mode: number;
  mode_label: string;
  prefill_worker_id: number;
  decode_worker_id: number;
  matched_prefix_len: number;
  predicted_ttft_ms: number;
  best_disagg_ttft_ms: number;
  best_agg_ttft_ms: number;
  exposed_nixl_stall_ms: number;
  overlap_efficiency_pct: number;
  routing_latency_us: number;
  t_producer_ready_ms: number;
  t_transfer_start_ms: number;
  t_transfer_end_ms: number;
  t_consumer_ready_ms: number;
  layer_waterfall: CrossoverWaterfallStage[];
  crossover_curve: CrossoverCurvePoint[];
}

export interface ClusterWorker {
  worker_id: number;
  pool: 'PREFILL' | 'DECODE';
  healthy: boolean;
  sm_util_pct: number;
  hbm_bw_pct: number;
  hbm_bw_tbps: number;
  power_watts: number;
  queue_wait_ms: number;
  radix_hbm_used_pct: number;
}

export interface FailoverEvent {
  failed_decode_worker: number;
  backup_decode_worker: number;
  failover_latency_ms: number;
  sessions_reanchored: number;
  sessions_lost: number;
  kv_blocks_migrated: number;
  nfr23_passed: boolean;
}

export interface ClusterTelemetryResponse {
  poll_interval_ms: number;
  telemetry_backend: string;
  cluster_power_kw: number;
  failed_workers: number[];
  last_failover: FailoverEvent | null;
  kpis: DynamoBenchmark;
  workers: ClusterWorker[];
}

export interface ParetoPoint {
  load_scale: number;
  goodput_ktok_per_s: number;
  p99_ttft_ms: number;
  avg_ttft_ms: number;
  sla_goodput_pct: number;
  cost_per_1m_usd: number;
  tokens_per_watt: number;
}

export interface ParetoSeries {
  id: string;
  label: string;
  points: ParetoPoint[];
}

export interface ParetoResponse {
  schema: string;
  hardware_profile: string;
  series: ParetoSeries[];
}

export interface ZMQEventEnvelope {
  event_id: number;
  timestamp_ns: number;
  worker_id: number;
  event: {
    type: string;
    block_hashes: string[];
    parent_block_hash: string;
    token_block_size: number;
    tokconomics_s_retain: number;
  };
}
