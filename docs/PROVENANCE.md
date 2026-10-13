# Source provenance and paper metrics

## Source version

Core training, model, dataset and evaluation modules come from the archived `TMM_Traj_v1_stageSDDS_standard_sdd_benchmark_full.zip`, representing the Stage3E-based implementation plus later SDD evaluation support. Their original SHA256 values are recorded in `source_sha256.json`. Only `models/__init__.py` was reduced to exporting the paper's active model builders; model mathematics were not modified.

Table IV original runner and 60 new controlled results come from the `TMM_TableIV_ControlledRepro_v1` experiment; other unrelated architecture searches and baselines are excluded from this public candidate.

Visualization code is the source-coordinate Matplotlib renderer. The 15 ETH–UCY frames and 3 SDD crop/trajectory arrays are deliberately excluded pending dataset publication rights and Fig. 6 alignment review.

## Current paper results (archived, NOT rerun in this release candidate)

- ETH–UCY five-fold mean minADE/minFDE: **0.21/0.36 meters** across seeds 42, 43, 44 (Stage3E).
- SDD mean minADE/minFDE: **7.40/11.78 pixels** across three seeds (locked evaluation).
- Table IV means over 4 arms x 5 folds x 3 seeds: base 0.239/0.406; rotation 0.218/0.372; mask/noise 0.228/0.395; combined 0.208/0.358 (meters).
- Trajectory-social parameters: 82,529; archived native batch-one latency: median 0.660 ms, p95 0.775 ms. The full visual configuration: 3.608 ms median, p95 3.807 ms. Do not compare these native timings directly to the matched UNIV multi-agent benchmark.
- In the matched UNIV GPU-resident forward benchmark, N=57 K=20: Proposed 1.712 ms; TUTR 4.788 ms; EigenTrajectory 10.650 ms; ViTE 53.555 ms; FlowChain 75.360 ms. This is one observed window, with pretrained checkpoints; CPU preprocessing is excluded and FlowChain's online pipeline is not qualified. Recheck against original raw CSV before inclusion in a new paper revision.

Any claimed full reproduction must carry code SHA, data split hashes, checkpoint SHA, seed IDs, software versions and measured units. No metric is generated when a required asset is missing.
