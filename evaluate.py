"""Evaluate standard fixed-budget and multi-view inference separately."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from config import DEFAULT_VARIANT
from data import build_dataloaders, list_available_folds
from models import build_model
from utils import (
    count_parameters,
    estimate_flops,
    evaluate_loader,
    model_size_bytes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", default="data/ethucy_5fold_social")
    parser.add_argument("--test_set", default="eth")
    parser.add_argument("--view_counts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--best_k", type=int, default=20)
    parser.add_argument("--miss_threshold", type=float, default=2.0)
    parser.add_argument("--gaussian_std", type=float, default=0.1)
    parser.add_argument("--conflict_radius", type=float, default=3.0)
    parser.add_argument("--prediction_latency_ms", type=float, default=None,
                        help="Optional measured p95 latency for response-margin analysis.")
    parser.add_argument("--selector", choices=("confidence_diverse", "fps"),
                        default="confidence_diverse")
    parser.add_argument("--confidence_weight", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    args = parser.parse_args()

    available = list_available_folds(args.data_root)
    if args.test_set not in available:
        raise FileNotFoundError(
            f"Fold '{args.test_set}' is unavailable. Available folds: {available}"
        )
    device = torch.device(args.device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    train_args = checkpoint.get("args", {})
    variant = train_args.get("variant", DEFAULT_VARIANT)
    model = build_model(
        variant,
        obs_len=train_args.get("obs_len", 8),
        pred_len=train_args.get("pred_len", 12),
        channels=train_args.get("channels", 2),
        individual=train_args.get("individual", False),
        num_modes=train_args.get("num_modes", 20),
        hyp_hidden_dim=train_args.get("hyp_hidden_dim", 64),
        social_hidden_dim=train_args.get("social_hidden_dim", 32),
        social_dim=train_args.get("social_dim", 32),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loaders = build_dataloaders(
        args.data_root, args.test_set,
        batch_size=args.batch_size,
        num_workers=0,
        val_ratio=0.1,
        split_seed=train_args.get("seed", 42),
        val_source="train_group",
        relative_coords=train_args.get("relative_coords", True),
        max_test_samples=args.max_test_samples,
        pin_memory=args.device.startswith("cuda"),
    )

    results = {}
    overall_start = time.perf_counter()
    for views in sorted(set(args.view_counts)):
        if views < 1:
            raise ValueError("All view counts must be positive")
        start = time.perf_counter()
        row = evaluate_loader(
            model, loaders["test"], device,
            best_k=args.best_k,
            gaussian_std=args.gaussian_std,
            miss_threshold=args.miss_threshold,
            n_views=views,
            selector=args.selector,
            confidence_weight=args.confidence_weight,
            conflict_radius=args.conflict_radius,
            prediction_latency_ms=args.prediction_latency_ms,
            seed=train_args.get("seed", 42) + 1000 + views,
        )
        row["whole_dataset_evaluation_seconds"] = time.perf_counter() - start
        results[str(views)] = row

    first_batch = next(iter(loaders["test"]))
    single = {key: (value[:1] if isinstance(value, torch.Tensor) else value)
              for key, value in first_batch.items()}
    single_flops = estimate_flops(model, single)
    report = {
        "checkpoint": str(Path(args.ckpt)),
        "variant": variant,
        "data_root": args.data_root,
        "test_set": args.test_set,
        "best_k": args.best_k,
        "selector": args.selector,
        "confidence_weight": args.confidence_weight,
        "parameters": count_parameters(model),
        "fp32_weight_bytes": model_size_bytes(model, 4),
        "fp16_weight_bytes": model_size_bytes(model, 2),
        "single_forward_flops": single_flops,
        "network_flops_by_views": {str(v): single_flops * v for v in sorted(set(args.view_counts))},
        "results_by_views": results,
        "total_script_seconds": time.perf_counter() - overall_start,
    }
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
