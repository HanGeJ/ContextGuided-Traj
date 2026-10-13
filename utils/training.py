from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_nll_loss(cands, log_vars, target, best_idx):
    B = target.size(0)
    bp = cands[torch.arange(B, device=target.device), best_idx]
    lv = log_vars[torch.arange(B, device=target.device), best_idx].clamp(-6, 6)
    return (0.5 * (lv + (bp - target) ** 2 / lv.exp())).mean()


def candidate_quality(cands: torch.Tensor, target: torch.Tensor,
                      fde_weight: float = 0.5):
    """Return per-candidate ADE, FDE, and a scalar ranking cost.

    The cost is ``ADE + fde_weight * FDE``. Distances use Euclidean trajectory
    errors in the native coordinate system, matching the evaluation geometry.
    """
    if cands.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected candidates [B,K,T,2] and target [B,T,2].")
    if cands.size(0) != target.size(0) or cands.size(2) != target.size(1):
        raise ValueError("Candidate/target batch or horizon mismatch.")
    if cands.size(-1) != 2 or target.size(-1) != 2:
        raise ValueError("Trajectory coordinates must have final dimension 2.")
    if float(fde_weight) < 0.0:
        raise ValueError("fde_weight must be non-negative.")

    dist = torch.linalg.norm(cands - target.unsqueeze(1), dim=-1)
    ade = dist.mean(dim=-1)
    fde = dist[..., -1]
    cost = ade + float(fde_weight) * fde
    return ade, fde, cost


def quality_aware_targets(cands: torch.Tensor, target: torch.Tensor,
                          temperature: float = 0.1,
                          fde_weight: float = 0.5):
    """Build detached soft ranking targets from candidate geometric quality.

    A lower trajectory cost receives larger probability mass. The target is
    detached deliberately so the ranking objective supervises mode logits and
    does not create an indirect gradient that moves candidate coordinates via
    the target distribution itself.
    """
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and > 0.")
    ade, fde, cost = candidate_quality(cands, target, fde_weight=fde_weight)
    target_prob = torch.softmax(-cost.detach() / temperature, dim=1)
    return target_prob, ade, fde, cost


def soft_quality_ranking_loss(logits: torch.Tensor,
                              cands: torch.Tensor,
                              target: torch.Tensor,
                              temperature: float = 0.1,
                              fde_weight: float = 0.5):
    target_prob, ade, fde, cost = quality_aware_targets(
        cands, target, temperature=temperature, fde_weight=fde_weight,
    )
    log_prob = torch.log_softmax(logits, dim=1)
    loss = -(target_prob * log_prob).sum(dim=1).mean()
    return loss, target_prob, ade, fde, cost


def compute_training_loss(output, target, cls_lambda=0.2, nll_lambda=0.1,
                          ranking_loss="hard_ce", rank_temperature=0.1,
                          rank_fde_weight=0.5):
    """Training objective for deterministic and multi-hypothesis variants.

    ``ranking_loss='hard_ce'`` exactly preserves the Stage-2 winner-classification
    behavior. ``ranking_loss='soft_quality'`` replaces the one-hot winner target
    with a detached quality distribution derived from candidate ADE/FDE.
    ``cls_lambda`` is retained as the ranking-term weight for checkpoint/config
    compatibility with earlier experiments.
    """
    aux = {}
    if "candidates" in output:
        c, lg = output["candidates"], output["mode_logits"]

        # Preserve the original WTA regression winner definition so Stage 2B
        # changes only ranking supervision, not candidate-generation training.
        pm = ((c - target.unsqueeze(1)) ** 2).mean(dim=(-1, -2))
        bi = pm.argmin(dim=1)
        B = target.size(0)
        reg = F.mse_loss(c[torch.arange(B, device=target.device), bi], target)

        ranking_loss = str(ranking_loss)
        if ranking_loss == "hard_ce":
            rank = F.cross_entropy(lg, bi)
            aux["cls_loss"] = float(rank.detach())
        elif ranking_loss == "soft_quality":
            rank, target_prob, ade, fde, quality = soft_quality_ranking_loss(
                lg, c, target,
                temperature=rank_temperature,
                fde_weight=rank_fde_weight,
            )
            pred_prob = torch.softmax(lg, dim=1)
            eps = torch.finfo(pred_prob.dtype).eps
            target_entropy = -(target_prob * (target_prob + eps).log()).sum(dim=1).mean()
            pred_entropy = -(pred_prob * (pred_prob + eps).log()).sum(dim=1).mean()
            quality_best = quality.argmin(dim=1)
            confidence_best = lg.argmax(dim=1)
            top1_match = (quality_best == confidence_best).float().mean()

            aux["rank_loss"] = float(rank.detach())
            aux["rank_target_entropy"] = float(target_entropy.detach())
            aux["rank_pred_entropy"] = float(pred_entropy.detach())
            aux["rank_top1_match"] = float(top1_match.detach())
            aux["rank_mean_candidate_ade"] = float(ade.mean().detach())
            aux["rank_mean_candidate_fde"] = float(fde.mean().detach())
        else:
            raise ValueError(
                f"Unknown ranking_loss={ranking_loss!r}; expected hard_ce or soft_quality."
            )

        loss = reg + float(cls_lambda) * rank
        aux["reg_loss"] = float(reg.detach())
        aux["ranking_term"] = float(rank.detach())

        if "log_vars" in output:
            nll = gaussian_nll_loss(c, output["log_vars"], target, bi)
            loss = loss + float(nll_lambda) * nll
            aux["nll_loss"] = float(nll.detach())
    else:
        loss = F.mse_loss(output["pred"], target)
        aux["reg_loss"] = float(loss.detach())
    return loss, aux


def summarize_aux_metrics(aux_list):
    if not aux_list:
        return {}
    ks = sorted(set().union(*[a.keys() for a in aux_list]))
    return {
        k: sum(a[k] for a in aux_list if k in a)
        / sum(1 for a in aux_list if k in a)
        for k in ks
    }
