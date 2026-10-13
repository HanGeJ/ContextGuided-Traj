from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from data.visual_dataset import build_visual_dataloaders
from models import build_visual_model
from train_visual_decoupled import evaluate, load_generator_checkpoint
from utils import set_seed


LOCKED = {
    "variant": "decoupled_social_visual_adapter",
    "crop_tag": "m16_160px",
    "visual_backbone": "mobilenet_v3_small",
    "backbone_trainable": "frozen",
    "pretrained_backbone": True,
    "correction_scale": 0.20,
    "ranking_enabled": False,
    "selection_basis": (
        "Stage-3D validation only: three seeds; geometry-only social generator + "
        "frozen ImageNet MobileNetV3-Small; correction scale 0.20."
    ),
}


def _stats(values):
    values = [float(x) for x in values]
    if not values:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return {"mean": mean, "sd": math.sqrt(var), "n": len(values)}


def _aggregate(rows, key):
    return _stats([r[key] for r in rows])


def _metrics(m):
    keep = (
        "top1_ade", "top1_fde", "minade_k", "minfde_k",
        "top1_miss_rate", "miss_rate_k", "nll_top1", "nll_best",
        "context_correction_rms", "context_gate_mean",
    )
    return {k: float(m[k]) for k in keep if k in m}


def main():
    ap = argparse.ArgumentParser(
        description="One-time held-out SDD evaluation for the Stage-3D locked configuration."
    )
    ap.add_argument("--data_root", default="data/sdd_raw_visual")
    ap.add_argument("--raw_control_root", default="runs/tmm_stage3b_raw_control/sdd_raw_visual")
    ap.add_argument("--stage3d_root", default="runs/tmm_stage3d_decoupled")
    ap.add_argument("--out", default="runs/tmm_stage3e_locked_sdd_test.json")
    ap.add_argument("--lock_file", default="runs/tmm_stage3e_sdd_test_lock.json")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--allow_retest", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    lock_path = Path(args.lock_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if lock_path.exists() and not args.allow_retest:
        old = json.loads(lock_path.read_text())
        if old.get("complete"):
            raise SystemExit(
                f"Held-out SDD test is already sealed at {lock_path}. "
                "Refusing to re-evaluate without --allow_retest."
            )

    device = torch.device(args.device)
    rows = []

    # The final configuration is frozen before any held-out SDD test is read.
    pending_lock = {
        "stage": "TMM-v1 Stage 3E",
        "complete": False,
        "held_out_test_unlocked": True,
        "locked_configuration": LOCKED,
        "seeds": [42, 43, 44],
    }
    lock_path.write_text(json.dumps(pending_lock, indent=2))

    for seed in (42, 43, 44):
        set_seed(seed)
        generator = (
            Path(args.raw_control_root)
            / "multihyp_social" / "sdd" / f"seed_{seed}" / "best_model.pt"
        )
        geometry = (
            Path(args.stage3d_root)
            / "decoupled_social_visual_adapter"
            / "m16_160px"
            / "mobilenet_v3_small"
            / "frozen"
            / "pretrained"
            / "cscale_0.2"
            / "soft_t0.05_d0.0001"
            / f"seed_{seed}"
            / "geometry_best.pt"
        )
        if not generator.exists():
            raise FileNotFoundError(generator)
        if not geometry.exists():
            raise FileNotFoundError(geometry)

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
            pretrained_backbone=True,
            backbone_trainable=LOCKED["backbone_trainable"],
            correction_scale=LOCKED["correction_scale"],
        ).to(device)
        report = load_generator_checkpoint(model, generator)

        # Pre-registered comparator: the exact social generator before visual
        # geometry correction. The zero-initialized adapter preserves its
        # candidates and logits.
        model.eval()
        baseline_test = evaluate(
            model, loaders["test"], device,
            best_k=20, miss_threshold=2.0, amp=args.amp,
            enable_reranker=False,
        )

        gck = torch.load(geometry, map_location=device)
        model.load_state_dict(gck["model_state_dict"])
        final_val = evaluate(
            model, loaders["val"], device,
            best_k=20, miss_threshold=2.0, amp=args.amp,
            enable_reranker=False,
        )
        final_test = evaluate(
            model, loaders["test"], device,
            best_k=20, miss_threshold=2.0, amp=args.amp,
            enable_reranker=False,
        )

        row = {
            "seed": seed,
            "generator_checkpoint": str(generator),
            "geometry_checkpoint": str(geometry),
            "generator_load_report": report,
            "baseline_social_test": _metrics(baseline_test),
            "locked_validation": _metrics(final_val),
            "locked_test": _metrics(final_test),
        }
        rows.append(row)
        print(
            f"seed {seed} | social test "
            f"{baseline_test['top1_ade']:.4f}/{baseline_test['top1_fde']:.4f} "
            f"min@20 {baseline_test['minade_k']:.4f}/{baseline_test['minfde_k']:.4f} | "
            f"LOCKED visual test "
            f"{final_test['top1_ade']:.4f}/{final_test['top1_fde']:.4f} "
            f"min@20 {final_test['minade_k']:.4f}/{final_test['minfde_k']:.4f}",
            flush=True,
        )

    metrics = ("top1_ade", "top1_fde", "minade_k", "minfde_k")
    aggregate = {"baseline_social": {}, "locked_visual": {}, "relative_change_percent": {}}
    for metric in metrics:
        b = _aggregate([r["baseline_social_test"] for r in rows], metric)
        f = _aggregate([r["locked_test"] for r in rows], metric)
        aggregate["baseline_social"][metric] = b
        aggregate["locked_visual"][metric] = f
        aggregate["relative_change_percent"][metric] = (
            (f["mean"] - b["mean"]) / b["mean"] * 100.0
            if math.isfinite(b["mean"]) and abs(b["mean"]) > 1e-12
            else float("nan")
        )

    payload = {
        "stage": "TMM-v1 Stage 3E",
        "held_out_test_used": True,
        "selection_after_test_forbidden": True,
        "locked_configuration": LOCKED,
        "rows": rows,
        "aggregate": aggregate,
    }
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lock_payload = dict(pending_lock)
    lock_payload.update({
        "complete": True,
        "result_file": str(out_path),
        "selection_after_test_forbidden": True,
    })
    lock_path.write_text(json.dumps(lock_payload, indent=2))

    print("\n===== STAGE 3E LOCKED SDD TEST =====")
    for name in ("baseline_social", "locked_visual"):
        a = aggregate[name]
        print(
            f"{name:16s} | Top1 "
            f"{a['top1_ade']['mean']:.4f}±{a['top1_ade']['sd']:.4f}/"
            f"{a['top1_fde']['mean']:.4f}±{a['top1_fde']['sd']:.4f} | "
            f"min@20 {a['minade_k']['mean']:.4f}±{a['minade_k']['sd']:.4f}/"
            f"{a['minfde_k']['mean']:.4f}±{a['minfde_k']['sd']:.4f}"
        )
    print("Relative locked-vs-social delta (%; negative is better):")
    print(json.dumps(aggregate["relative_change_percent"], indent=2))
    print("SDD held-out test is now SEALED for model selection.")
    print("Saved:", out_path)
    print("Lock:", lock_path)


if __name__ == "__main__":
    main()
