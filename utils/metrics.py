from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def _deterministic_metrics(pred: torch.Tensor, target: torch.Tensor,
                           miss_threshold: float = 2.0) -> Dict[str, torch.Tensor]:
    dists = torch.linalg.norm(pred - target, dim=-1)
    return {
        "mse": F.mse_loss(pred, target),
        "ade": dists.mean(),
        "fde": dists[:, -1].mean(),
        "miss_rate": (dists[:, -1] > miss_threshold).float().mean(),
    }


def compute_candidate_metrics(candidates: torch.Tensor, target: torch.Tensor,
                              miss_threshold: float = 2.0) -> Dict[str, torch.Tensor]:
    """Oracle fixed/expanded-budget metrics for [B,K,T,2] candidates."""
    dists = torch.linalg.norm(candidates - target.unsqueeze(1), dim=-1)
    ade_per = dists.mean(dim=-1)
    fde_per = dists[..., -1]
    min_ade = ade_per.min(dim=1).values
    min_fde = fde_per.min(dim=1).values
    return {
        "best_k_ade": min_ade.mean(),
        "best_k_fde": min_fde.mean(),
        "best_k_miss_rate": (min_fde > miss_threshold).float().mean(),
    }


def compute_nll_metric(candidates: torch.Tensor, log_vars: torch.Tensor,
                       logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    batch_size = target.size(0)
    row = torch.arange(batch_size, device=target.device)

    top1_idx = logits.argmax(dim=1)
    top1_pred = candidates[row, top1_idx]
    top1_lv = log_vars[row, top1_idx].clamp(-6, 6)
    nll_top1 = 0.5 * (top1_lv + (top1_pred - target) ** 2 / top1_lv.exp())

    per_mode = ((candidates - target.unsqueeze(1)) ** 2).mean(dim=(-1, -2))
    best_idx = per_mode.argmin(dim=1)
    best_pred = candidates[row, best_idx]
    best_lv = log_vars[row, best_idx].clamp(-6, 6)
    nll_best = 0.5 * (best_lv + (best_pred - target) ** 2 / best_lv.exp())

    return {"nll_top1": nll_top1.mean(), "nll_best": nll_best.mean()}


def _gaussian_best_k(mean_pred: torch.Tensor, target: torch.Tensor, best_k: int,
                     gaussian_std: float, miss_threshold: float,
                     generator: torch.Generator | None = None) -> Dict[str, torch.Tensor]:
    if best_k <= 1:
        det = _deterministic_metrics(mean_pred, target, miss_threshold)
        return {
            "best_k_ade": det["ade"],
            "best_k_fde": det["fde"],
            "best_k_miss_rate": det["miss_rate"],
        }
    noise = torch.randn(
        (mean_pred.size(0), best_k - 1) + mean_pred.shape[1:],
        device=mean_pred.device,
        dtype=mean_pred.dtype,
        generator=generator,
    ) * gaussian_std
    candidates = torch.cat([mean_pred.unsqueeze(1), mean_pred.unsqueeze(1) + noise], dim=1)
    return compute_candidate_metrics(candidates, target, miss_threshold)


def compute_eval_metrics(output: dict, target: torch.Tensor, best_k: int = 1,
                         gaussian_std: float = 0.0,
                         miss_threshold: float = 2.0,
                         generator: torch.Generator | None = None) -> Dict[str, torch.Tensor]:
    metrics = _deterministic_metrics(output["pred"], target, miss_threshold)
    if "candidates" in output:
        metrics.update(compute_candidate_metrics(
            output["candidates"], target, miss_threshold=miss_threshold
        ))
        if "log_vars" in output and "mode_logits" in output:
            metrics.update(compute_nll_metric(
                output["candidates"], output["log_vars"],
                output["mode_logits"], target,
            ))
    else:
        metrics.update(_gaussian_best_k(
            output["pred"], target, best_k, gaussian_std,
            miss_threshold=miss_threshold, generator=generator,
        ))
    return metrics


def _confusion_counts(predicted: torch.Tensor, ground_truth: torch.Tensor,
                      valid: torch.Tensor) -> dict[str, float]:
    p = predicted.bool() & valid
    g = ground_truth.bool() & valid
    return {
        "tp": float((p & g).sum().item()),
        "fp": float((p & ~g & valid).sum().item()),
        "fn": float((~p & g & valid).sum().item()),
        "tn": float((~p & ~g & valid).sum().item()),
    }


def compute_safety_counts(top1_rel: torch.Tensor,
                          candidates_rel: torch.Tensor | None,
                          origin: torch.Tensor,
                          ego_pred_abs: torch.Tensor,
                          ego_valid: torch.Tensor,
                          gt_conflict: torch.Tensor,
                          dt: torch.Tensor,
                          conflict_radius: float = 3.0,
                          target_rel: torch.Tensor | None = None,
                          prediction_latency_ms: float | None = None,
                          ego_speed_mps: torch.Tensor | None = None) -> dict[str, float]:
    """Open-loop pedestrian-vehicle proximity-conflict proxy counts.

    A predicted conflict occurs when the synchronized target and ego centers
    approach within ``conflict_radius`` at any future step. This is a safety
    proxy for risk-screening experiments; it does not represent collision
    geometry or closed-loop vehicle safety.
    """
    valid = ego_valid.reshape(-1) > 0.5
    gt_hit = None
    if target_rel is not None:
        target_abs = target_rel + origin[:, None, :]
        gt_distance = torch.linalg.norm(target_abs - ego_pred_abs, dim=-1)
        gt_hit = gt_distance <= float(conflict_radius)
        gt = gt_hit.any(dim=1)
    else:
        gt = gt_conflict.reshape(-1) > 0.5
    top_abs = top1_rel + origin[:, None, :]
    top_dist = torch.linalg.norm(top_abs - ego_pred_abs, dim=-1)
    top_hit = top_dist <= float(conflict_radius)
    pred_top = top_hit.any(dim=1)

    if candidates_rel is None:
        pred_any = pred_top
        any_hit = top_hit[:, None]
    else:
        cand_abs = candidates_rel + origin[:, None, None, :]
        cand_dist = torch.linalg.norm(cand_abs - ego_pred_abs[:, None], dim=-1)
        any_hit = cand_dist <= float(conflict_radius)
        pred_any = any_hit.any(dim=(1, 2))

    result: dict[str, float] = {
        "valid": float(valid.sum().item()),
        "gt_positive": float((gt & valid).sum().item()),
        "top1_min_distance_sum": float(top_dist.amin(dim=1)[valid].sum().item()) if valid.any() else 0.0,
    }
    if prediction_latency_ms is not None:
        latency_s = float(prediction_latency_ms) / 1000.0
        result["assumed_prediction_latency_ms_sum"] = float(prediction_latency_ms) * float(valid.sum().item())
        if ego_speed_mps is not None and valid.any():
            result["latency_travel_distance_sum_m"] = float(
                (ego_speed_mps.reshape(-1)[valid] * latency_s).sum().item()
            )
        else:
            result["latency_travel_distance_sum_m"] = 0.0
    for prefix, pred in (("top1", pred_top), ("anyk", pred_any)):
        for key, value in _confusion_counts(pred, gt, valid).items():
            result[f"{prefix}_{key}"] = value
        if prediction_latency_ms is not None and gt_hit is not None:
            tp_mask = pred & gt & valid
            gt_first = gt_hit.float().argmax(dim=1)
            gt_time_s = (gt_first.to(dt.dtype) + 1.0) * dt
            margins = gt_time_s[tp_mask] - float(prediction_latency_ms) / 1000.0
            result[f"{prefix}_response_margin_sum_s"] = float(margins.sum().item()) if tp_mask.any() else 0.0
            result[f"{prefix}_response_margin_count"] = float(tp_mask.sum().item())
            result[f"{prefix}_positive_response_margin_count"] = float((margins > 0.0).sum().item()) if tp_mask.any() else 0.0

    # Earliest predicted conflict time among true positives.
    top_first = top_hit.float().argmax(dim=1)
    top_tp = pred_top & gt & valid
    result["top1_detection_time_sum_s"] = float(
        (((top_first[top_tp].to(dt.dtype) + 1.0) * dt[top_tp]).sum().item())
    ) if top_tp.any() else 0.0
    result["top1_detection_count"] = float(top_tp.sum().item())

    if candidates_rel is None:
        any_time_mask = top_hit
    else:
        any_time_mask = any_hit.any(dim=1)
    any_first = any_time_mask.float().argmax(dim=1)
    any_tp = pred_any & gt & valid
    result["anyk_detection_time_sum_s"] = float(
        (((any_first[any_tp].to(dt.dtype) + 1.0) * dt[any_tp]).sum().item())
    ) if any_tp.any() else 0.0
    result["anyk_detection_count"] = float(any_tp.sum().item())
    return result


def merge_count_dict(total: dict[str, float], batch: dict[str, float]) -> None:
    for key, value in batch.items():
        total[key] = total.get(key, 0.0) + float(value)


def _safe_ratio(num: float, den: float) -> float | None:
    return float(num / den) if den > 0 else None


def finalize_safety_counts(counts: dict[str, float]) -> dict[str, float]:
    valid = counts.get("valid", 0.0)
    result = {
        "valid_samples": int(valid),
        "conflict_prevalence": _safe_ratio(counts.get("gt_positive", 0.0), valid),
        "mean_top1_min_distance_m": _safe_ratio(
            counts.get("top1_min_distance_sum", 0.0), valid
        ),
    }
    if "assumed_prediction_latency_ms_sum" in counts:
        result["assumed_prediction_latency_ms"] = _safe_ratio(
            counts.get("assumed_prediction_latency_ms_sum", 0.0), valid
        )
        result["mean_vehicle_travel_during_latency_m"] = _safe_ratio(
            counts.get("latency_travel_distance_sum_m", 0.0), valid
        )
    for prefix in ("top1", "anyk"):
        tp = counts.get(f"{prefix}_tp", 0.0)
        fp = counts.get(f"{prefix}_fp", 0.0)
        fn = counts.get(f"{prefix}_fn", 0.0)
        tn = counts.get(f"{prefix}_tn", 0.0)
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        result.update({
            f"{prefix}_conflict_precision": precision,
            f"{prefix}_conflict_recall": recall,
            f"{prefix}_conflict_f1": (
                _safe_ratio(2.0 * precision * recall, precision + recall)
                if precision is not None and recall is not None else None
            ),
            f"{prefix}_false_alarm_rate": _safe_ratio(fp, fp + tn),
            f"{prefix}_missed_hazard_rate": _safe_ratio(fn, tp + fn),
            f"{prefix}_mean_detection_time_s": _safe_ratio(
                counts.get(f"{prefix}_detection_time_sum_s", 0.0),
                counts.get(f"{prefix}_detection_count", 0.0),
            ),
        })
        if f"{prefix}_response_margin_count" in counts:
            margin_count = counts.get(f"{prefix}_response_margin_count", 0.0)
            result[f"{prefix}_mean_response_margin_s"] = _safe_ratio(
                counts.get(f"{prefix}_response_margin_sum_s", 0.0), margin_count
            )
            result[f"{prefix}_positive_response_margin_rate"] = _safe_ratio(
                counts.get(f"{prefix}_positive_response_margin_count", 0.0),
                margin_count,
            )
    return result
