from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data.visual_dataset import build_visual_dataloaders
from models import build_visual_model
from train_visual_decoupled import load_generator_checkpoint
from utils import set_seed


SCENE_NAMES = ("bookstore", "coupa", "deathCircle", "gates", "hyang", "little", "nexus", "quad")
SEEDS = (42, 43, 44)
LOCKED = {
    "generator": "multihyp_social",
    "variant": "decoupled_social_visual_adapter",
    "crop_tag": "m16_160px",
    "visual_backbone": "mobilenet_v3_small",
    "backbone_trainable": "frozen",
    "pretrained_backbone": True,
    "correction_scale": 0.20,
    "ranking_enabled": False,
    "candidate_budget": 20,
}


def _move(batch, device, key):
    v = batch.get(key)
    return v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v


def _forward(model, batch, device):
    return model(
        _move(batch, device, "obs"),
        scene_crop=_move(batch, device, "scene_crop"),
        neighbor_rel=_move(batch, device, "neighbor_rel"),
        neighbor_mask=_move(batch, device, "neighbor_mask"),
        enable_reranker=False,
    )


def _stats(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {"mean": None, "sd": None, "n": 0}
    return {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=0)),
        "n": int(x.size),
    }


def _protocol_audit(data_root: Path) -> dict:
    fold = data_root / "sdd"
    train_path = fold / "train.npz"
    test_path = fold / "test.npz"
    manifest_path = fold / "manifest.json"
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run the existing raw-SDD preparation stage first."
            )

    report = {
        "data_root": str(data_root),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "manifest_path": str(manifest_path),
        "required_keys": {},
    }
    split_payload = {}
    for split, path in (("train", train_path), ("test", test_path)):
        with np.load(path, allow_pickle=False) as d:
            required = (
                "obs", "pred", "pixel_obs", "pixel_pred", "scale_m_per_px",
                "clip_index", "scene_index", "video_index", "dt",
            )
            missing = [k for k in required if k not in d]
            if missing:
                raise RuntimeError(f"{path}: missing required fields {missing}")
            n = int(len(d["obs"]))
            if d["obs"].shape[1:] != (8, 2) or d["pred"].shape[1:] != (12, 2):
                raise RuntimeError(
                    f"{path}: expected obs/pred shapes [N,8,2]/[N,12,2], "
                    f"got {d['obs'].shape}/{d['pred'].shape}"
                )
            scale = np.asarray(d["scale_m_per_px"], dtype=np.float64).reshape(-1)
            if len(scale) != n or (not np.isfinite(scale).all()) or (scale <= 0).any():
                raise RuntimeError(f"{path}: invalid scale_m_per_px")
            dt = np.asarray(d["dt"], dtype=np.float64).reshape(-1)
            if len(dt) != n or not np.allclose(dt, 0.4, atol=1e-6):
                raise RuntimeError(f"{path}: expected dt=0.4 s for all samples")

            # Raw-SDD preparation stores absolute metric coordinates as
            # pixel_coordinate * meters_per_pixel. This verifies that the
            # pixel evaluator uses the exact coordinate system from SDD.
            obs_recon = np.asarray(d["obs"], dtype=np.float64) / scale[:, None, None]
            pred_recon = np.asarray(d["pred"], dtype=np.float64) / scale[:, None, None]
            obs_px = np.asarray(d["pixel_obs"], dtype=np.float64)
            pred_px = np.asarray(d["pixel_pred"], dtype=np.float64)
            max_recon = float(max(
                np.max(np.abs(obs_recon - obs_px)),
                np.max(np.abs(pred_recon - pred_px)),
            ))
            if max_recon > 5e-3:
                raise RuntimeError(
                    f"{path}: metric-to-pixel reconstruction mismatch {max_recon:.6g}px"
                )
            clips = set(map(int, np.asarray(d["clip_index"]).reshape(-1).tolist()))
            split_payload[split] = {
                "num_samples": n,
                "num_clips": len(clips),
                "clip_indices": sorted(clips),
                "scale_min": float(scale.min()),
                "scale_max": float(scale.max()),
                "metric_pixel_reconstruction_max_abs_px": max_recon,
            }
            report["required_keys"][split] = list(required)

    overlap = set(split_payload["train"]["clip_indices"]) & set(split_payload["test"]["clip_indices"])
    if overlap:
        raise RuntimeError(f"SDD train/test recording leakage: {sorted(overlap)}")
    report["splits"] = split_payload
    report["train_test_clip_overlap"] = []
    report["obs_len"] = 8
    report["pred_len"] = 12
    report["dt_seconds"] = 0.4
    report["sampling_hz"] = 2.5
    report["candidate_budget"] = 20
    report["coordinate_for_paper_comparison"] = "pixel"
    report["selection_policy"] = (
        "post-freeze measurement audit only; no training, hyperparameter tuning, "
        "checkpoint selection, or architecture selection is permitted"
    )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report["manifest_protocol"] = manifest.get("protocol")
        report["manifest_source"] = manifest.get("source")
    return report


def _evaluate_one_seed(model, loader, scales: np.ndarray, raw_seq_id: np.ndarray,
                       device: torch.device, amp: bool) -> dict:
    model.eval()
    use_amp = bool(amp and device.type == "cuda")
    offset = 0
    per_sample = defaultdict(list)
    per_scene = defaultdict(lambda: defaultdict(list))
    per_clip = defaultdict(lambda: defaultdict(list))
    n_modes_seen = None

    with torch.no_grad():
        for batch in loader:
            target_m = _move(batch, device, "pred")
            bs = int(target_m.size(0))
            sl = slice(offset, offset + bs)
            if offset + bs > len(scales):
                raise RuntimeError("Test loader produced more samples than test.npz")

            # Guard against any future sampler/order change.
            batch_seq = batch.get("seq_id")
            if isinstance(batch_seq, torch.Tensor):
                got_seq = batch_seq.detach().cpu().numpy().astype(np.int64)
            else:
                got_seq = np.asarray(batch_seq, dtype=np.int64)
            if not np.array_equal(got_seq, raw_seq_id[sl]):
                raise RuntimeError(
                    "Test loader order does not match test.npz; refusing scale-based pixel conversion."
                )

            scale = torch.as_tensor(scales[sl], device=device, dtype=target_m.dtype)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = _forward(model, batch, device)

            cand_m = out.get("candidates")
            if cand_m is None:
                raise RuntimeError("Locked model did not return candidate trajectories")
            if cand_m.ndim != 4:
                raise RuntimeError(f"Expected candidates [B,K,T,2], got {tuple(cand_m.shape)}")
            k = int(cand_m.size(1))
            if k < 20:
                raise RuntimeError(f"Expected at least K=20 candidates, got K={k}")
            if n_modes_seen is None:
                n_modes_seen = k
            elif n_modes_seen != k:
                raise RuntimeError("Candidate budget changed within the test loader")
            cand_m = cand_m[:, :20]

            # Translation cancels in displacement error. Dividing metric-space
            # residuals by meters/pixel is exactly equivalent to evaluating the
            # reconstructed trajectories in SDD pixel coordinates.
            top_px_err = (out["pred"] - target_m) / scale[:, None, None]
            cand_px_err = (cand_m - target_m[:, None]) / scale[:, None, None, None]
            top_dist_px = torch.linalg.norm(top_px_err.float(), dim=-1)
            cand_dist_px = torch.linalg.norm(cand_px_err.float(), dim=-1)
            top_ade_px = top_dist_px.mean(dim=-1)
            top_fde_px = top_dist_px[:, -1]
            ade_mode_px = cand_dist_px.mean(dim=-1)
            fde_mode_px = cand_dist_px[:, :, -1]
            minade_px, minade_idx = ade_mode_px.min(dim=1)
            minfde_px = fde_mode_px.min(dim=1).values
            # Diagnostic paired FDE for the candidate that minimizes ADE.
            paired_fde_px = fde_mode_px.gather(1, minade_idx[:, None]).squeeze(1)

            # Metric-space cross-check against the legacy Stage-3E evaluator.
            top_dist_m = torch.linalg.norm((out["pred"] - target_m).float(), dim=-1)
            cand_dist_m = torch.linalg.norm((cand_m - target_m[:, None]).float(), dim=-1)
            top_ade_m = top_dist_m.mean(dim=-1)
            top_fde_m = top_dist_m[:, -1]
            minade_m = cand_dist_m.mean(dim=-1).min(dim=1).values
            minfde_m = cand_dist_m[:, :, -1].min(dim=1).values

            tensors = {
                "top1_ade_px": top_ade_px,
                "top1_fde_px": top_fde_px,
                "minade20_px": minade_px,
                "minfde20_px": minfde_px,
                "fde_at_minade20_px": paired_fde_px,
                "top1_ade_m": top_ade_m,
                "top1_fde_m": top_fde_m,
                "minade20_m": minade_m,
                "minfde20_m": minfde_m,
            }
            scene_arr = batch["scene_index"].detach().cpu().numpy().astype(np.int64)
            clip_arr = batch["clip_index"].detach().cpu().numpy().astype(np.int64)
            for metric, tensor in tensors.items():
                vals = tensor.detach().cpu().numpy().astype(np.float64)
                per_sample[metric].extend(vals.tolist())
                for i, val in enumerate(vals.tolist()):
                    scene_id = int(scene_arr[i])
                    scene_name = SCENE_NAMES[scene_id] if 0 <= scene_id < len(SCENE_NAMES) else f"scene_{scene_id}"
                    per_scene[scene_name][metric].append(float(val))
                    per_clip[str(int(clip_arr[i]))][metric].append(float(val))
            offset += bs

    if offset != len(scales):
        raise RuntimeError(f"Test loader covered {offset} samples, expected {len(scales)}")

    def summarize(group):
        return {metric: _stats(vals) for metric, vals in group.items()}

    return {
        "num_samples": int(offset),
        "candidate_budget_returned": int(n_modes_seen or 0),
        "candidate_budget_evaluated": 20,
        "overall": summarize(per_sample),
        "by_scene": {name: summarize(vals) for name, vals in sorted(per_scene.items())},
        "by_clip_index": {name: summarize(vals) for name, vals in sorted(per_clip.items(), key=lambda z: int(z[0]))},
    }


def _aggregate_seed_metrics(rows: list[dict]) -> dict:
    metric_names = (
        "top1_ade_px", "top1_fde_px", "minade20_px", "minfde20_px",
        "fde_at_minade20_px", "top1_ade_m", "top1_fde_m", "minade20_m", "minfde20_m",
    )
    out = {}
    for metric in metric_names:
        vals = [r["metrics"]["overall"][metric]["mean"] for r in rows]
        out[metric] = _stats(vals)
    return out


def _write_csv(path: Path, rows: list[dict], aggregate: dict):
    fields = [
        "seed", "num_samples", "top1_ade_px", "top1_fde_px",
        "minade20_px", "minfde20_px", "fde_at_minade20_px",
        "top1_ade_m", "top1_fde_m", "minade20_m", "minfde20_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            overall = row["metrics"]["overall"]
            w.writerow({
                "seed": row["seed"],
                "num_samples": row["metrics"]["num_samples"],
                **{k: overall[k]["mean"] for k in fields if k not in ("seed", "num_samples")},
            })
        w.writerow({
            "seed": "mean_3seed",
            "num_samples": rows[0]["metrics"]["num_samples"] if rows else "",
            **{k: aggregate[k]["mean"] for k in fields if k not in ("seed", "num_samples")},
        })


def _paper_row(aggregate: dict) -> str:
    ade = aggregate["minade20_px"]["mean"]
    fde = aggregate["minfde20_px"]["mean"]
    ade_sd = aggregate["minade20_px"]["sd"]
    fde_sd = aggregate["minfde20_px"]["sd"]
    return (
        "% Standard SDD benchmark row: 3-seed mean; K=20; pixel ADE/FDE\n"
        f"Ours & -- & {ade:.2f} & {fde:.2f} \\\\\n"
        f"% 3-seed mean +/- std: ADE {ade:.4f} +/- {ade_sd:.4f}, "
        f"FDE {fde:.4f} +/- {fde_sd:.4f}\n"
    )


def _bundle(outdir: Path, files: list[Path]) -> Path:
    bundle = outdir / "TMM_Stage_SDDS_Standard_SDD_Benchmark_review_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if p.exists():
                z.write(p, arcname=p.name)
    return bundle


def main():
    ap = argparse.ArgumentParser(
        description="TMM Stage SDD-S: post-freeze literature-compatible SDD pixel benchmark evaluation."
    )
    ap.add_argument("--data_root", default="data/sdd_raw_visual")
    ap.add_argument("--raw_control_root", default="runs/tmm_stage3b_raw_control/sdd_raw_visual")
    ap.add_argument("--stage3d_root", default="runs/tmm_stage3d_decoupled")
    ap.add_argument("--outdir", default="runs/tmm_stage_sdds_standard_sdd")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--acknowledge_postfreeze_evaluation", action="store_true")
    args = ap.parse_args()

    if not args.acknowledge_postfreeze_evaluation:
        raise SystemExit(
            "This is a post-freeze re-evaluation of the already locked SDD test. "
            "Re-run with --acknowledge_postfreeze_evaluation to confirm that the result "
            "will not be used for model/hyperparameter selection."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol_audit(Path(args.data_root))
    protocol_path = outdir / "protocol_audit.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    test_path = Path(args.data_root) / "sdd" / "test.npz"
    with np.load(test_path, allow_pickle=False) as d:
        scales = np.asarray(d["scale_m_per_px"], dtype=np.float64).reshape(-1)
        raw_seq_id = np.asarray(d["seq_id"], dtype=np.int64).reshape(-1)

    device = torch.device(args.device)
    rows = []
    for seed in SEEDS:
        set_seed(seed)
        generator = (
            Path(args.raw_control_root) / "multihyp_social" / "sdd" /
            f"seed_{seed}" / "best_model.pt"
        )
        geometry = (
            Path(args.stage3d_root) / "decoupled_social_visual_adapter" /
            "m16_160px" / "mobilenet_v3_small" / "frozen" / "pretrained" /
            "cscale_0.2" / "soft_t0.05_d0.0001" / f"seed_{seed}" / "geometry_best.pt"
        )
        for p in (generator, geometry):
            if not p.exists():
                raise FileNotFoundError(
                    f"Required frozen checkpoint missing: {p}. "
                    "Stage SDD-S intentionally performs no retraining."
                )

        loaders = build_visual_dataloaders(
            args.data_root, "sdd", LOCKED["crop_tag"],
            batch_size=args.batch_size, num_workers=args.num_workers,
            val_ratio=0.1, split_seed=seed, val_source="train_group",
            pin_memory=device.type == "cuda",
        )
        model = build_visual_model(
            LOCKED["variant"],
            obs_len=8, pred_len=12, channels=2, individual=False,
            num_modes=20, hyp_hidden_dim=64,
            social_hidden_dim=32, social_dim=32,
            visual_dim=64, context_hidden_dim=128, reranker_hidden_dim=128,
            visual_backbone=LOCKED["visual_backbone"],
            # Avoid any network access. The full frozen visual backbone is
            # overwritten by the Stage-3D checkpoint below.
            pretrained_backbone=False,
            backbone_trainable=LOCKED["backbone_trainable"],
            correction_scale=LOCKED["correction_scale"],
        ).to(device)
        generator_report = load_generator_checkpoint(model, generator)
        gck = torch.load(geometry, map_location=device)
        model.load_state_dict(gck["model_state_dict"], strict=True)
        metrics = _evaluate_one_seed(
            model, loaders["test"], scales, raw_seq_id, device,
            amp=not args.no_amp,
        )
        row = {
            "seed": seed,
            "generator_checkpoint": str(generator),
            "geometry_checkpoint": str(geometry),
            "generator_load_report": generator_report,
            "metrics": metrics,
        }
        rows.append(row)
        o = metrics["overall"]
        print(
            f"seed={seed} | pixel minADE@20/minFDE@20 "
            f"{o['minade20_px']['mean']:.4f}/{o['minfde20_px']['mean']:.4f} | "
            f"Top1 {o['top1_ade_px']['mean']:.4f}/{o['top1_fde_px']['mean']:.4f} | "
            f"metric cross-check {o['minade20_m']['mean']:.4f}/{o['minfde20_m']['mean']:.4f}",
            flush=True,
        )

    aggregate = _aggregate_seed_metrics(rows)
    payload = {
        "stage": "TMM Stage SDD-S — Standard SDD Benchmark Replication",
        "status": "post_freeze_measurement_only",
        "model_selection_after_result": "forbidden",
        "protocol": {
            "split": "public TUTR/SocialVAE SDD train-test recording split",
            "source_coordinates": "raw Stanford Drone Dataset annotation centers",
            "observation_prediction": "8 -> 12",
            "sampling_interval_s": 0.4,
            "sampling_rate_hz": 2.5,
            "candidate_budget": 20,
            "reported_unit": "pixel",
            "metric": "minADE@20 / minFDE@20; minima computed independently over the fixed 20 candidates",
        },
        "locked_configuration": LOCKED,
        "protocol_audit": protocol,
        "rows": rows,
        "aggregate_3seed": aggregate,
    }
    json_path = outdir / "standard_sdd_benchmark.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    csv_path = outdir / "standard_sdd_benchmark_by_seed.csv"
    _write_csv(csv_path, rows, aggregate)
    tex_path = outdir / "paper_row.tex"
    tex_path.write_text(_paper_row(aggregate), encoding="utf-8")
    decision_path = outdir / "README_RESULT.txt"
    decision_path.write_text(
        "TMM Stage SDD-S completed.\n"
        "Use aggregate_3seed.minade20_px/minfde20_px for the literature comparison table.\n"
        "The test result is measurement-only and must not be used to change the model, checkpoints, or hyperparameters.\n",
        encoding="utf-8",
    )
    bundle = _bundle(outdir, [protocol_path, json_path, csv_path, tex_path, decision_path])

    a = aggregate
    print("\n===== TMM STAGE SDD-S FINAL =====")
    print(
        "Paper SDD row (3-seed mean, pixels): "
        f"minADE@20={a['minade20_px']['mean']:.4f}±{a['minade20_px']['sd']:.4f}, "
        f"minFDE@20={a['minfde20_px']['mean']:.4f}±{a['minfde20_px']['sd']:.4f}"
    )
    print("Saved:", json_path)
    print("CSV:", csv_path)
    print("LaTeX row:", tex_path)
    print("Review bundle:", bundle)


if __name__ == "__main__":
    main()
