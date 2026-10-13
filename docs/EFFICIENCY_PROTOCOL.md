# Latency benchmark interpretation

There are **three distinct protocols** in the paper history:

1. **Original native batch-one model inference**: Proposed traj.+social 0.660 ms (median), Proposed visual 3.608 ms; Social-STGCNN, SocialVAE and EqMotion measured with native sampling/decoding. GPU memory and FLOPs follow that experiment.
2. **Unified single-target GPU timing (v1.2)**: Proposed ~0.675 ms vs TUTR ~1.702 ms mean of five fold-wise medians. FP32, warmup 50, iterations 300, K=20, 1 target.
3. **Matched UNIV K=20 GPU-resident forward / decoder timing (v1.3.2)**: same targets, N=1,2,4,8,16,32,57; CPU input preparation excluded. Model checkpoints vary by architecture. Do not claim a full online end-to-end speedup from protocol #3.

The v1.3.1 vectorized neighbor-preparation implementation was experimentally checked to preserve Proposed's predictions and improves prepared in-memory full-pipeline latency; this is distinct from protocol #3. Under v1.3.2, FlowChain sample diversity, weight restore and prepared-input hashes were checked, but the time for its original CPU preprocessing was not qualified. Keep `strict_all_five=false` for complete online system latency.

We distribute selected *experiment-owned* timing and aggregated summary only, not third-party implementations. Separate programs in `experiments/efficiency` perform benchmarks using the installed upstream repos and frozen checkpoint paths; require the original external assets and repo sources to run.
