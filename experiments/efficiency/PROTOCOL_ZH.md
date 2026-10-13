# v1.3 Same-Scene, Same-N, K20 Strict Qualification Contract

## This experiment seeks a fair runtime comparison, not artificially larger speedups.

The *same* observed UNIV scene window is extracted once using the ViTE official ETH-UCY loader. The window's `obs[N,8,2]` is saved locally, without future targets. Every method receives exactly the same target indices and coordinates at each N. ViTE's released checkpoint is UNIV-only; therefore UNIV is the sole common fold for this release. Other-fold timings must be kept in a separate analysis.

The planned workloads are N=1,2,4,8,16,32, and `all` targets of that same window (typically N=57). Native graph / neighbor construction is preserved per method. Truncating the target set changes social conditioning; the N-scaling study explicitly measures that dependence.

For every N and method, acceptance requires:

1. The trained checkpoint must load strictly, with SHA256 metadata. No untrained/random surrogate is accepted.
2. The canonical observation and target-subset SHA256 must match across methods.
3. The original network must produce exactly 20 future samples of 12 two-dimensional coordinates **for every target**.
4. The full model forward and required coordinate reconstruction must execute under FP32, no TF32, no gradients, same RTX 4090.
5. CUDA Event benchmarking: 50 warmups + 300 repeated GPU-ready full network measurements. Host synchronized wall clock: 60 repetitions from in-memory canonical observations through native preprocessing, transfer, full inference and decoding. Disk IO / model initialization excluded from both.
6. Original preprocessing can differ, by design. The physically identical source observation is required; models may apply their documented native input coordinate normalization. Input transforms must be documented.
7. For N=1, unsupported social graph models must be marked unsupported; N is not reduced silently, and padding does not count as predictions.
8. No comparison across unequal N, K, scene or timing scope. Do not divide a joint-scene inference time by N and call it single-agent latency.

## Special FlowChain gate

The existing official evaluation logs report `nsample=1`. Original FlowChain also separately advertises fast probability *update* which is not full K20 prediction. This archive does NOT pretend to know an unverified K20 interface. A source inventory (`FLOWCHAIN_SOURCE_AUDIT.json`) is generated for the exact installed repository, so that a fully native, same-input K20 adapter can be built. Until then it remains BLOCKED. A unified five-method conclusion is prohibited whenever `qualification.json` has `strict_all_five=false`.

This is necessary: a strict all-five benchmark is not established merely by including five rows in a table.

## Interpretation

Report `gpu_forward_median_ms` for on-device prepared tensor inference, and `prepared_pipeline_wall_median_ms` for cached in-memory preprocessing + transfers + model + decoding. The throughput is `1000 * number_of_targets / wall_median_ms`, not by arbitrary averaging across scenes. Median of per-fold median metrics is not pooled median, and this benchmark runs on UNIV rather than all five folds.

Accuracy should be evaluated separately on consistent held-out labels. This timing package does not use ground-truth futures, tune checkpoints or produce ADE/FDE. It is complementary evidence to Table VI, not an automatic replacement for that table.

User-controlled environment variables:

- `TMM_ROOT` — TMM project root (default `/root/autodl-tmp/TMM_Traj/source/tmm_v1`)
- `TMM_V13_SCENE_INDEX` — ViTE official UNIV sample index, default 0
- `TMM_V13_N` — e.g. `1,2,4,8,16,32,all`
- `TMM_V13_METHODS` — e.g. `proposed tutr eigen vite flow`
- `TMM_V13_GPU` — default 0

If the actual source uses different coordinate conventions between ETH-UCY implementations, ensure the model's native normalization and physical units are audited before claiming an accuracy–efficiency Pareto relationship. This benchmark isolates timing only.
