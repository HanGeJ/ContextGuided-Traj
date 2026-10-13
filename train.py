from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from config import DEFAULT_VARIANT, PACKAGE_NAME
from data import build_dataloaders, list_available_folds
from models import VARIANT_TO_MODEL, build_model
from models.components import augment_batch, augment_batch_social
from utils import (
    compute_eval_metrics,
    compute_training_loss,
    count_parameters,
    estimate_flops,
    evaluate_loader,
    model_size_bytes,
    set_seed,
    summarize_aux_metrics,
)


def _model_kwargs(args) -> dict:
    return {
        "obs_len": args.obs_len,
        "pred_len": args.pred_len,
        "channels": args.channels,
        "individual": args.individual,
        "num_modes": args.num_modes,
        "hyp_hidden_dim": args.hyp_hidden_dim,
        "social_hidden_dim": args.social_hidden_dim,
        "social_dim": args.social_dim,
    }


def _checkpoint_metric_key(args, model) -> str:
    key = str(args.checkpoint_metric)
    if key == "auto":
        return "minade_k" if model.supports_multimodal else "top1_ade"
    if not model.supports_multimodal and key.startswith("min"):
        raise ValueError(
            f"checkpoint_metric={key} requires a multi-hypothesis model."
        )
    return key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train latency-aware pedestrian trajectory forecasting variants."
    )
    parser.add_argument("--variant", default=DEFAULT_VARIANT,
                        choices=tuple(VARIANT_TO_MODEL))
    parser.add_argument("--data_root", default="data/ethucy_5fold_social")
    parser.add_argument("--test_set", default="univ",
                        help="Fold directory name under data_root.")
    parser.add_argument("--val_source", default="train_group",
                        choices=("train_group", "train", "test", "test_split"))
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--individual", action="store_true")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="runs/tits_revision")

    parser.add_argument("--best_k", type=int, default=20)
    parser.add_argument("--miss_threshold", type=float, default=2.0)
    parser.add_argument("--gaussian_std", type=float, default=0.1)
    parser.add_argument("--num_modes", type=int, default=20)
    parser.add_argument("--hyp_hidden_dim", type=int, default=64)
    parser.add_argument("--social_hidden_dim", type=int, default=32)
    parser.add_argument("--social_dim", type=int, default=32)
    parser.add_argument("--cls_lambda", type=float, default=0.2,
                        help="Weight of the hypothesis ranking/classification term.")
    parser.add_argument("--nll_lambda", type=float, default=0.1)
    parser.add_argument(
        "--ranking_loss", choices=("hard_ce", "soft_quality"), default="hard_ce",
        help=(
            "Hypothesis-ranking supervision. hard_ce preserves the Stage-2 "
            "winner classification; soft_quality uses detached ADE/FDE-based "
            "soft targets."
        ),
    )
    parser.add_argument(
        "--rank_temperature", type=float, default=0.1,
        help="Temperature for soft_quality targets; smaller values are sharper.",
    )
    parser.add_argument(
        "--rank_fde_weight", type=float, default=0.5,
        help="Weight of FDE in the soft ranking cost ADE + weight * FDE.",
    )
    parser.add_argument(
        "--checkpoint_metric",
        choices=("auto", "minade_k", "minfde_k", "top1_ade", "top1_fde"),
        default="auto",
        help=(
            "Validation metric used for checkpoint selection. Stage-2B search "
            "uses top1_ade so ranking quality is selected on validation only."
        ),
    )
    parser.add_argument(
        "--skip_test", action="store_true",
        help=(
            "Do not evaluate the held-out test split. Use this during "
            "hyperparameter search to prevent test-set tuning."
        ),
    )

    parser.add_argument("--use_augment", action="store_true")
    parser.add_argument("--no_augment", dest="use_augment", action="store_false")
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--mask_prob", type=float, default=0.3)
    parser.add_argument("--max_mask_steps", type=int, default=4)
    parser.add_argument("--rotation_max_deg", type=float, default=180.0,
                        help="Random training rotation range [-deg,+deg].")
    parser.add_argument("--use_inversion", action="store_true",
                        help="Ablation only; redundant for full-circle rotation.")

    parser.add_argument("--eval_views", type=int, default=1,
                        help="Final rotation-view count; primary comparison remains view=1.")
    parser.add_argument("--selector", choices=("confidence_diverse", "fps"),
                        default="confidence_diverse")
    parser.add_argument("--confidence_weight", type=float, default=0.5)
    parser.add_argument("--conflict_radius", type=float, default=3.0)

    parser.add_argument("--relative_coords", dest="relative_coords", action="store_true")
    parser.add_argument("--absolute_coords", dest="relative_coords", action="store_false")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.set_defaults(relative_coords=True, use_augment=True)
    args = parser.parse_args()

    set_seed(args.seed)
    available = list_available_folds(args.data_root)
    if args.test_set not in available:
        raise FileNotFoundError(
            f"Fold '{args.test_set}' is unavailable under {args.data_root}. "
            f"Available folds: {available}"
        )

    output_dir = (
        Path(args.out_dir) / Path(args.data_root).name / args.variant
        / args.test_set / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_dataloaders(
        args.data_root, args.test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        split_seed=args.seed,
        val_source=args.val_source,
        relative_coords=args.relative_coords,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device)
    model = build_model(args.variant, **_model_kwargs(args)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    metric_key = _checkpoint_metric_key(args, model)
    best_value = float("inf")
    best_epoch = None
    best_checkpoint = output_dir / "best_model.pt"
    history = []

    print(
        f"Config | variant={args.variant} fold={args.test_set} "
        f"validation={args.val_source} augmentation={args.use_augment} "
        f"ranking={args.ranking_loss} checkpoint_metric={metric_key} "
        f"skip_test={args.skip_test} final_views={args.eval_views}", flush=True,
    )
    if args.ranking_loss == "soft_quality":
        print(
            f"Rank   | temperature={args.rank_temperature:g} "
            f"fde_weight={args.rank_fde_weight:g} lambda={args.cls_lambda:g}",
            flush=True,
        )
    print(
        f"Data   | train={len(loaders['train'].dataset)} "
        f"val={len(loaders['val'].dataset)} test={len(loaders['test'].dataset)}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"ade": 0.0, "fde": 0.0, "best_k_ade": 0.0,
                   "best_k_fde": 0.0}
        sample_count = 0
        aux_rows = []

        for batch in loaders["train"]:
            obs = batch["obs"].to(device)
            target = batch["pred"].to(device)
            ego_obs_rel = batch["ego_obs_rel"].to(device)
            ego_valid = batch["ego_valid"].to(device)
            neighbor_rel = batch["neighbor_rel"].to(device)
            neighbor_mask = batch["neighbor_mask"].to(device)

            if args.use_augment:
                if getattr(model, "uses_social_context", False):
                    obs, target, neighbor_rel, neighbor_mask, ego_obs_rel = (
                        augment_batch_social(
                            obs, target, neighbor_rel, neighbor_mask,
                            ego_obs_rel=ego_obs_rel,
                            noise_std=args.noise_std,
                            mask_prob=args.mask_prob,
                            max_mask_steps=args.max_mask_steps,
                            rotation_max_deg=args.rotation_max_deg,
                            use_inversion=args.use_inversion,
                        )
                    )
                else:
                    obs, target, ego_obs_rel = augment_batch(
                        obs, target, ego_obs_rel=ego_obs_rel,
                        noise_std=args.noise_std,
                        mask_prob=args.mask_prob,
                        max_mask_steps=args.max_mask_steps,
                        rotation_max_deg=args.rotation_max_deg,
                        use_inversion=args.use_inversion,
                    )

            optimizer.zero_grad(set_to_none=True)
            output = model(
                obs, ego_obs_rel=ego_obs_rel, ego_valid=ego_valid,
                neighbor_rel=neighbor_rel, neighbor_mask=neighbor_mask,
            )
            loss, aux = compute_training_loss(
                output, target,
                cls_lambda=args.cls_lambda,
                nll_lambda=args.nll_lambda,
                ranking_loss=args.ranking_loss,
                rank_temperature=args.rank_temperature,
                rank_fde_weight=args.rank_fde_weight,
            )
            loss.backward()
            optimizer.step()

            metrics = compute_eval_metrics(
                output, target, best_k=args.best_k,
                gaussian_std=args.gaussian_std,
                miss_threshold=args.miss_threshold,
            )
            batch_size = obs.size(0)
            for key in running:
                if key in metrics:
                    running[key] += float(metrics[key].detach()) * batch_size
            sample_count += batch_size
            aux_rows.append(aux)

        train_metrics = {key: value / sample_count for key, value in running.items()}
        validation = evaluate_loader(
            model, loaders["val"], device,
            best_k=args.best_k,
            gaussian_std=args.gaussian_std,
            miss_threshold=args.miss_threshold,
            n_views=1,
            selector=args.selector,
            confidence_weight=args.confidence_weight,
            conflict_radius=args.conflict_radius,
            seed=args.seed + epoch,
        )
        val_standard = validation["standard"]
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "train_aux": summarize_aux_metrics(aux_rows),
            "validation": validation,
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train top1={train_metrics['ade']:.4f}/{train_metrics['fde']:.4f} "
            f"min@{args.best_k}={train_metrics['best_k_ade']:.4f}/"
            f"{train_metrics['best_k_fde']:.4f} | "
            f"val top1={val_standard['top1_ade']:.4f}/"
            f"{val_standard['top1_fde']:.4f} "
            f"min@{args.best_k}={val_standard['minade_k']:.4f}/"
            f"{val_standard['minfde_k']:.4f}", flush=True,
        )

        if val_standard[metric_key] < best_value:
            best_value = val_standard[metric_key]
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "package_name": PACKAGE_NAME,
            }, best_checkpoint)

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, allow_nan=False), encoding="utf-8"
    )

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Always re-evaluate the selected checkpoint on validation. During Stage-2B
    # hyperparameter search this is the only post-training evaluation, so the
    # held-out test fold is not used to choose ranking temperature.
    selected_validation = evaluate_loader(
        model, loaders["val"], device,
        best_k=args.best_k,
        gaussian_std=args.gaussian_std,
        miss_threshold=args.miss_threshold,
        n_views=1,
        selector=args.selector,
        confidence_weight=args.confidence_weight,
        conflict_radius=args.conflict_radius,
        seed=args.seed + 77777,
    )

    test_result = None
    total_eval_seconds = None
    if not args.skip_test:
        start = time.perf_counter()
        test_result = evaluate_loader(
            model, loaders["test"], device,
            best_k=args.best_k,
            gaussian_std=args.gaussian_std,
            miss_threshold=args.miss_threshold,
            n_views=args.eval_views,
            selector=args.selector,
            confidence_weight=args.confidence_weight,
            conflict_radius=args.conflict_radius,
            seed=args.seed + 99999,
        )
        total_eval_seconds = time.perf_counter() - start

    complexity_loader = loaders["val"] if args.skip_test else loaders["test"]
    first_batch = next(iter(complexity_loader))
    single = {
        key: (value[:1] if isinstance(value, torch.Tensor) else value)
        for key, value in first_batch.items()
    }
    single_forward_flops = estimate_flops(model, single)
    params = count_parameters(model)
    summary = {
        "package": PACKAGE_NAME,
        "variant": args.variant,
        "data_root": args.data_root,
        "test_set": args.test_set,
        "seed": args.seed,
        "validation_source": args.val_source,
        "checkpoint_metric": metric_key,
        "best_epoch": best_epoch,
        "best_metric_value": best_value,
        "arguments": vars(args),
        "selected_validation": selected_validation,
        "test": test_result,
        "model_parameters": params,
        "fp32_weight_bytes": model_size_bytes(model, 4),
        "fp16_weight_bytes": model_size_bytes(model, 2),
        "single_forward_flops": single_forward_flops,
        "network_flops_at_eval_views": single_forward_flops * args.eval_views,
        "whole_test_evaluation_seconds": total_eval_seconds,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    val_standard = selected_validation["standard"]
    print("\nSelected validation checkpoint", flush=True)
    print(
        f"  epoch={best_epoch} metric={metric_key} value={best_value:.6f} | "
        f"top1 ADE/FDE={val_standard['top1_ade']:.4f}/"
        f"{val_standard['top1_fde']:.4f} | minADE/minFDE@{args.best_k}="
        f"{val_standard['minade_k']:.4f}/{val_standard['minfde_k']:.4f}",
        flush=True,
    )

    if args.skip_test:
        print(
            "  Held-out test evaluation SKIPPED (hyperparameter-search safe mode).",
            flush=True,
        )
    else:
        standard = test_result["standard"]
        view = test_result["view_inference"]
        print("\nFinal held-out test", flush=True)
        print(
            f"  Standard single-view | top1 ADE/FDE={standard['top1_ade']:.4f}/"
            f"{standard['top1_fde']:.4f} | minADE/minFDE@{args.best_k}="
            f"{standard['minade_k']:.4f}/{standard['minfde_k']:.4f}", flush=True,
        )
        print(
            f"  View inference ({args.eval_views} views) | expanded@"
            f"{view['expanded_candidate_budget']}={view['expanded_minade']:.4f}/"
            f"{view['expanded_minfde']:.4f} | selected@{view['selected_candidate_budget']}="
            f"{view.get('selected_minade_k', float('nan')):.4f}/"
            f"{view.get('selected_minfde_k', float('nan')):.4f}", flush=True,
        )

    print(
        f"  Parameters={params} | single-forward FLOPs={single_forward_flops} | "
        f"effective network FLOPs={single_forward_flops * args.eval_views}",
        flush=True,
    )


if __name__ == "__main__":
    main()
