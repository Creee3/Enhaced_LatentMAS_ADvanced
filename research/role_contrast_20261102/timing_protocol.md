# Timing plan: reuse LatentMAS rollout wall time and OBF component scopes

2026-11-02. Design only; no cloud job, benchmark repetition or historical result modification.

## Primary sources checked

- LatentMAS official run.py (main, read on2026-11-02): https://github.com/Gen-Verse/LatentMAS/blob/main/run.py . ModelWrapper construction precedes start_time=time.time(); total_time is recorded after the evaluation loop and before evaluate(preds); time_per_sample_sec=total_time/args.max_samples. This is loaded-model evaluation wall time, not a network-transfer measurement. The actual implementation commit must be pinned before a new measurement; a moving main URL is not a reproducibility lock.
- OBF revision36f43183e84d2fbf3b84c9bf4e3e06b7c442a523 run.py and methods/latent_mas.py: outer evaluation wall time plus per-sample latent_inference_time, text_inference_time, compression_time and compression_core_time. The method wrapper synchronizes CUDA immediately before and after latent, compression and final text scopes. compression_time is the outer measured compressor call; compression_core_time is the compressor-reported value with analysis-only diagnostics deducted, and does not replace the headline outer measurement.
- OBF paper Appendix H/I: https://arxiv.org/html/2604.13349v3 . Reports compression and end-to-end wall time separately. Its colocated single-process/single-device chain does not measure inter-device network transfer, as expressly noted in Appendix I.

## Adopted scopes

- Record model load/hash/alignment preparation separately; do not count startup, GPU admission waiting, manager queueing or experiment downtime as per-question model latency.
- Preserve existing monotonic timers; use device synchronization at both boundaries for CUDA work, following the upstream outer-scope pattern. Do not modify frozen snapshot source or overwrite old timing fields.
- Record producer latent inference, selector-required receiver probe, selector core including role grouping/objective evaluation/search/gather, Judger generation, and generated token count. The required probe is method overhead and cannot be deducted as an optional diagnostic.
- Separately time research-only held-out-query rollout, tensor checks, position-array archives and durable evidence I/O. Retain an instrumented wall scope if these operations remain on the observed path; do not call it a clean deployment timing.
- Keep OBF-style outer compression time and a separately scoped core timing. Avoid inferring core time by subtracting unsynchronized asynchronous diagnostic kernels.

## Shared prefix and cost attribution

The proposed T/R/R-shuffle experiment shares the original receiver first two handoffs, producer states and perhaps the receiver probe. Store those measured shared times once with their RID/source identifiers.

- Arm incremental latency measures the new final selection/Judger path.
- A sum of measured common-prefix time and arm-only components is a reconstructed standalone estimate; label it as reconstructed, not an independently measured end-to-end rollout.
- Actual wall time for executing all arms with shared work is experiment throughput/cost and is not the standalone latency of one method.
- Keep all real model requests charged exactly once in the shared ledger regardless of analytical attribution.

## Existing results

Current cloud_worker.case_arm_seconds starts after the shared Planner and initial planner cache clone; it is not full four-role end-to-end time. selection_seconds includes the receiver probe plus compress, and does not have independent start/end CUDA barriers. judge_seconds is a broad wrapper scope. Engine receipts synchronize on successful return, but start after ledger reservation and include some request/trace I/O; they are not pure GPU kernel time. Other proof hooks and position-array writes can affect the observed wall time.

Reuse historical answers and retain their original timing values as historical observations. Do not present old timing versus a new instrumented run as a controlled speedup on identical hardware/load. No full baseline rerun is authorized by this timing plan. A future dedicated timing subset would need an explicitly specified matched setup.

## Reporting

Primary effectiveness remains frozen-parser correctness and actual KV-byte retention. Timing becomes a recorded engineering metric: outer compression, required probe, selector core, Judger time and tokens, and either measured rollout wall time or explicitly reconstructed standalone time. State device, model/software locks, batch size, synchronization, diagnostic status, external load and sample count. Report mean seconds/sample for upstream compatibility and per-question variation; do not infer transfer latency/GPU/FLOP gains from KV bytes.
