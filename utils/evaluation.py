from __future__ import annotations

from collections import defaultdict

import torch

from models.components import (
    select_confidence_diverse_k,
    select_coverage_optimal_k,
    symmetry_view_candidates,
)
from .metrics import (
    compute_candidate_metrics,
    compute_eval_metrics,
    compute_safety_counts,
    finalize_safety_counts,
    merge_count_dict,
)


def _to_device(batch: dict, key: str, device: torch.device):
    value = batch.get(key)
    return value.to(device) if isinstance(value, torch.Tensor) else value


def _mean_metric(pred: torch.Tensor, target: torch.Tensor,
                 miss_threshold: float) -> dict[str, torch.Tensor]:
    dist = torch.linalg.norm(pred - target, dim=-1)
    return {
        "top1_ade": dist.mean(),
        "top1_fde": dist[:, -1].mean(),
        "top1_miss_rate": (dist[:, -1] > miss_threshold).float().mean(),
    }


@torch.no_grad()
def evaluate_loader(model, loader, device: torch.device, best_k: int = 20,
                    gaussian_std: float = 0.1, miss_threshold: float = 2.0,
                    n_views: int = 1, selector: str = "confidence_diverse",
                    confidence_weight: float = 0.5,
                    conflict_radius: float = 3.0,
                    prediction_latency_ms: float | None = None,
                    seed: int | None = None) -> dict:
    model.eval()
    standard_sum: dict[str, float] = defaultdict(float)
    view_sum: dict[str, float] = defaultdict(float)
    standard_safety: dict[str, float] = {}
    view_safety: dict[str, float] = {}
    n_samples = 0

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    for batch in loader:
        obs = _to_device(batch, "obs", device)
        target = _to_device(batch, "pred", device)
        ego_obs_rel = _to_device(batch, "ego_obs_rel", device)
        ego_valid = _to_device(batch, "ego_valid", device)
        neighbor_rel = _to_device(batch, "neighbor_rel", device)
        neighbor_mask = _to_device(batch, "neighbor_mask", device)
        batch_size = obs.size(0)

        output = model(
            obs, ego_obs_rel=ego_obs_rel, ego_valid=ego_valid,
            neighbor_rel=neighbor_rel, neighbor_mask=neighbor_mask,
        )
        metrics = compute_eval_metrics(
            output, target, best_k=best_k, gaussian_std=gaussian_std,
            miss_threshold=miss_threshold, generator=generator,
        )
        name_map = {
            "mse": "mse",
            "ade": "top1_ade",
            "fde": "top1_fde",
            "miss_rate": "top1_miss_rate",
            "best_k_ade": "minade_k",
            "best_k_fde": "minfde_k",
            "best_k_miss_rate": "miss_rate_k",
            "nll_top1": "nll_top1",
            "nll_best": "nll_best",
        }
        for source, target_name in name_map.items():
            if source in metrics:
                standard_sum[target_name] += float(metrics[source].detach()) * batch_size

        merge_count_dict(standard_safety, compute_safety_counts(
            output["pred"], output.get("candidates"),
            origin=_to_device(batch, "origin", device),
            ego_pred_abs=_to_device(batch, "ego_pred_abs", device),
            ego_valid=ego_valid,
            gt_conflict=_to_device(batch, "gt_conflict", device),
            dt=_to_device(batch, "dt", device),
            conflict_radius=conflict_radius,
            target_rel=target,
            prediction_latency_ms=prediction_latency_ms,
            ego_speed_mps=_to_device(batch, "ego_speed_mps", device),
        ))

        # A view-count result is generated for every requested n_views. At
        # n_views=1, the expanded pool is the original candidate set.
        view_top1, merged, scores = symmetry_view_candidates(
            model, obs, ego_obs_rel=ego_obs_rel, ego_valid=ego_valid,
            neighbor_rel=neighbor_rel, neighbor_mask=neighbor_mask,
            n_views=n_views,
        )
        for key, value in _mean_metric(view_top1, target, miss_threshold).items():
            view_sum[key] += float(value.detach()) * batch_size
        expanded_metrics = compute_candidate_metrics(
            merged, target, miss_threshold=miss_threshold
        )
        view_sum["expanded_minade"] += float(expanded_metrics["best_k_ade"].detach()) * batch_size
        view_sum["expanded_minfde"] += float(expanded_metrics["best_k_fde"].detach()) * batch_size
        view_sum["expanded_miss_rate"] += float(
            expanded_metrics["best_k_miss_rate"].detach()
        ) * batch_size

        selected = None
        selected_scores = None
        if merged.size(1) >= best_k:
            if merged.size(1) == best_k:
                selected = merged
                selected_scores = scores
            elif selector == "fps":
                selected, indices = select_coverage_optimal_k(merged, K=best_k)
                selected_scores = torch.gather(scores, 1, indices)
            elif selector == "confidence_diverse":
                selected, indices = select_confidence_diverse_k(
                    merged, scores, K=best_k,
                    confidence_weight=confidence_weight,
                )
                selected_scores = torch.gather(scores, 1, indices)
            else:
                raise ValueError("selector must be 'confidence_diverse' or 'fps'")
            selected_metrics = compute_candidate_metrics(
                selected, target, miss_threshold=miss_threshold
            )
            view_sum["selected_minade_k"] += float(
                selected_metrics["best_k_ade"].detach()
            ) * batch_size
            view_sum["selected_minfde_k"] += float(
                selected_metrics["best_k_fde"].detach()
            ) * batch_size
            view_sum["selected_miss_rate_k"] += float(
                selected_metrics["best_k_miss_rate"].detach()
            ) * batch_size

        safety_candidates = selected if selected is not None else merged
        merge_count_dict(view_safety, compute_safety_counts(
            view_top1, safety_candidates,
            origin=_to_device(batch, "origin", device),
            ego_pred_abs=_to_device(batch, "ego_pred_abs", device),
            ego_valid=ego_valid,
            gt_conflict=_to_device(batch, "gt_conflict", device),
            dt=_to_device(batch, "dt", device),
            conflict_radius=conflict_radius,
            target_rel=target,
            prediction_latency_ms=prediction_latency_ms,
            ego_speed_mps=_to_device(batch, "ego_speed_mps", device),
        ))
        n_samples += batch_size

    if n_samples == 0:
        raise RuntimeError("Evaluation loader contains no samples")

    standard = {key: value / n_samples for key, value in standard_sum.items()}
    standard["candidate_budget"] = int(best_k if model.supports_multimodal else 1)
    view = {key: value / n_samples for key, value in view_sum.items()}
    base_modes = int(getattr(getattr(model, "head", None), "K", 1))
    view.update({
        "views": int(n_views),
        "expanded_candidate_budget": int(base_modes * n_views),
        "selected_candidate_budget": int(best_k) if base_modes * n_views >= best_k else None,
        "selector": selector,
        "confidence_weight": float(confidence_weight),
    })
    return {
        "num_samples": int(n_samples),
        "standard": standard,
        "view_inference": view,
        "safety_standard": finalize_safety_counts(standard_safety),
        "safety_view_inference": finalize_safety_counts(view_safety),
    }
