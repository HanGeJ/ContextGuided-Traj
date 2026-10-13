# Table IV Controlled Reproduction

Generated: 2026-10-11T00:49:25.194606+00:00

**Status: COMPLETE. Complete runs: 60/60.**

Architecture: multihyp_social; five ETH–UCY LOSO held-out folds × seeds 42,43,44.
Train-group validation only; checkpoint selected by minADE@20; final held-out fold evaluated after selection.
All runs trained for 100 epochs with Adam, learning rate 0.01 and batch size 512.
Training-only augmentation switches differ between rows; code/dataset remain identical.

| Arm | Complete fold/seed runs | minADE@20 | minFDE@20 |
|---|---:|---:|---:|
| Base losses | 15/15 | 0.2390 | 0.4058 |
| + Rotation | 15/15 | 0.2179 | 0.3719 |
| + Mask/Noise | 15/15 | 0.2279 | 0.3948 |
| Rotation + Mask/Noise | 15/15 | 0.2078 | 0.3578 |

## Paired differences relative to base losses

Negative values indicate improved predictions.

| Arm | Δ minADE@20 | Δ minFDE@20 |
|---|---:|---:|
| + Rotation | -0.0211 | -0.0339 |
| + Mask/Noise | -0.0111 | -0.0110 |
| Rotation + Mask/Noise | -0.0312 | -0.0480 |
