from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path.cwd()
required_files = [
    ROOT / "data/sdd_raw_visual/sdd/train.npz",
    ROOT / "data/sdd_raw_visual/sdd/test.npz",
    ROOT / "data/sdd_raw_visual/sdd/scene_crops_m16_160px/train.npy",
    ROOT / "data/sdd_raw_visual/sdd/scene_crops_m16_160px/test.npy",
]
for seed in (42, 43, 44):
    required_files += [
        ROOT / f"runs/tmm_stage3b_raw_control/sdd_raw_visual/multihyp_social/sdd/seed_{seed}/best_model.pt",
        ROOT / (
            "runs/tmm_stage3d_decoupled/decoupled_social_visual_adapter/"
            "m16_160px/mobilenet_v3_small/frozen/pretrained/cscale_0.2/"
            f"soft_t0.05_d0.0001/seed_{seed}/geometry_best.pt"
        ),
    ]
missing = [str(p.relative_to(ROOT)) for p in required_files if not p.exists()]
if missing:
    print("STAGE_SDDS_PREREQUISITES_FAIL")
    print(json.dumps({"missing": missing}, indent=2))
    raise SystemExit(2)

for split in ("train", "test"):
    path = ROOT / f"data/sdd_raw_visual/sdd/{split}.npz"
    with np.load(path, allow_pickle=False) as d:
        req = {"obs", "pred", "pixel_obs", "pixel_pred", "scale_m_per_px", "clip_index", "dt"}
        absent = sorted(req - set(d.files))
        if absent:
            raise SystemExit(f"{path}: missing fields {absent}")
        if d["obs"].shape[1:] != (8,2) or d["pred"].shape[1:] != (12,2):
            raise SystemExit(f"{path}: unexpected shapes {d['obs'].shape}/{d['pred'].shape}")
print("STAGE_SDDS_PREREQUISITES_OK")
