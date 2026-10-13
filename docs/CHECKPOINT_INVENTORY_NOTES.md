# Frozen checkpoint inventory status (2026-10-14)

The original AutoDL environment reported all **51/51 checkpoint paths present**:
45 ETH–UCY checkpoints (three architecture groups x five scenes x three seeds),
plus three SDD generators and three SDD visual adapters. The total reported
checkpoint size is **25,033,953 bytes (23.87 MiB)**. All 51 SHA256 values are
distinct. Paths, sizes and hashes are preserved in
`CHECKPOINT_INVENTORY_20261014.json`.

**Evidence level:** `inventory_only`. This verifies only what AutoDL's original
file inventory reported. No model weights are included in this source archive.
The original checkpoint bytes have **not** been copied or independently loaded
here. Benchmark reproduction, correct parameter interpretation, third-party
pretrained weight licenses and redistribution permission remain unverified.

For a private backup on the original AutoDL filesystem (outside this Git tree),
run:

```bash
cd /root/autodl-tmp
python ContextGuided_export_tool/export_autodl_checkpoints.py \
  --source /root/autodl-tmp/TMM_Traj/source/tmm_v1 \
  --destination /root/autodl-tmp/ContextGuided_private_assets \
  --include-weights
```

After running, verify that the command reports `found: 51`, `expected: 51`
and `copied_weights: true`, and verify hashes with the script supplied below:

```bash
cd /root/autodl-tmp/ContextGuided-Traj
python tools/verify_staged_checkpoints.py \
  --staging /root/autodl-tmp/ContextGuided_private_assets \
  --inventory docs/CHECKPOINT_INVENTORY_20261014.json
```

Never copy a private staging directory into a GitHub repository before the
rightsholders have approved public redistribution and actual benchmark evaluation.
