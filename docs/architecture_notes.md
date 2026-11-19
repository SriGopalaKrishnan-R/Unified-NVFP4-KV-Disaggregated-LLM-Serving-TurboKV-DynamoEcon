# Architecture Notes

## Kernel Layer in turbokv_fp4

The KV cache memory layout splits every sequence into three zones so 4 bit compression does not hurt long reasoning accuracy.

* Initial 4 sink tokens stay in FP8 E4M3 with per head FP32 scales so attention sinks remain stable
* Trailing 128 window tokens stay in a ring buffered FP8 E4M3 partition for local recency accuracy
* Middle historical blocks pack into non contiguous NVFP4 E2M1 pages using one FP8 E4M3 micro scale per 16 values nested inside a global FP32 tensor scale
* Fast Walsh Hadamard rotation runs across head sizes 64 or 128 before quantization to flatten channel spikes without lookup tables

Three attention paths share this paged block table.

* Initial prefill computes causal attention directly on BF16 inputs and packs KV blocks in the epilogue
* Chunked extend unpacks cached prefix pages into one shared FP8 workspace reused across all layers
* Autoregressive decode reads packed 128 bit uint4 words and unpacks nibbles in registers via PTX prmt and lop3 instructions alongside warp shuffle reductions
* Speculative tree verification evaluates up to 64 draft nodes and depth 8 in one pass using a 64 bit ancestor mask and commits accepted tokens without host syncs

## Cluster Layer in dynamoecon_nixl

Disaggregated serving often slows down when cache hit routing ignores worker queues or when remote KV transfer stalls exceed local compute savings.

* Lock striped 64 bucket Radix Tree tracks cached prefix blocks across workers and evicts low value leaves at the 90 percent HBM mark using recompute FLOPs and reuse rate
* Crossover router compares queued remote prefill plus layer pipelined NIXL transfer stall against local aggregated execution and switches modes dynamically
* NIXL transfer model tracks producer ready and transfer start and transfer end and consumer ready timestamps across all 32 layers
* Telemetry daemon polls GPU power and utilization every 50 ms through runtime NVML hooks or calibrated H100 and B200 power equations while tracking multi tier SLA goodput
* Failover manager re anchors active decode sessions onto healthy workers in under 1 ms when a worker heartbeat drops
