from __future__ import annotations

import math
import time
from typing import Iterable

import torch
import torch.nn as nn


def rotate_2d(x: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Rotate a tensor whose final dimension is ``(x, y)``."""
    c, s = math.cos(float(angle_rad)), math.sin(float(angle_rad))
    # Row-vector convention: x @ R^T.
    r_t = x.new_tensor([[c, s], [-s, c]])
    return x @ r_t


def _rotate_per_sample(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply one planar rotation angle to each sample in a batch."""
    c, s = angles.cos(), angles.sin()
    r_t = torch.stack(
        [torch.stack([c, s], dim=-1), torch.stack([-s, c], dim=-1)], dim=-2
    )
    original_shape = x.shape
    flat = x.reshape(x.size(0), -1, 2)
    return torch.bmm(flat, r_t).reshape(original_shape)


def kinematic_features(obs: torch.Tensor,
                       ego_obs_rel: torch.Tensor | None = None,
                       ego_valid: torch.Tensor | None = None,
                       use_ego_context: bool = False) -> torch.Tensor:
    """Return a compact five-dimensional motion descriptor.

    The first three dimensions describe target motion. When ego context is
    enabled, the last two dimensions encode target-to-vehicle range and radial
    closing motion. Keeping the descriptor dimensionality fixed preserves state
    dictionary compatibility with the original multi-hypothesis checkpoints.
    """
    vel = obs[:, 1:] - obs[:, :-1]
    speed = torch.linalg.norm(vel, dim=-1)
    mean_speed = speed.mean(dim=1, keepdim=True)
    terminal_speed = speed[:, -1:].contiguous()
    if vel.size(1) >= 2:
        mean_accel = torch.linalg.norm(vel[:, 1:] - vel[:, :-1], dim=-1).mean(
            dim=1, keepdim=True
        )
    else:
        mean_accel = torch.zeros_like(mean_speed)

    range_feature = torch.zeros_like(mean_speed)
    closing_feature = torch.zeros_like(mean_speed)
    if use_ego_context and ego_obs_rel is not None:
        if ego_valid is None:
            valid = torch.ones(obs.size(0), 1, device=obs.device, dtype=obs.dtype)
        else:
            valid = ego_valid.reshape(-1, 1).to(dtype=obs.dtype)
        # Both trajectories share the target's last observed origin. Their
        # difference is the synchronized vehicle-to-pedestrian displacement.
        relative_vehicle = ego_obs_rel - obs
        distance = torch.linalg.norm(relative_vehicle, dim=-1)
        current_range = distance[:, -1:].clamp_min(0.0)
        if distance.size(1) >= 2:
            # Positive values indicate that the vehicle and pedestrian became
            # closer during the last sampled interval.
            radial_closing = distance[:, -2:-1] - distance[:, -1:]
        else:
            radial_closing = torch.zeros_like(current_range)
        range_feature = (torch.log1p(current_range) / 4.0) * valid
        closing_feature = torch.tanh(radial_closing) * valid

    return torch.cat(
        [mean_speed, terminal_speed, mean_accel, range_feature, closing_feature],
        dim=1,
    )


class TimeLinearCore(nn.Module):
    """LTSF-Linear temporal backbone with a constant-motion-style prior."""

    def __init__(self, obs_len: int, pred_len: int, channels: int = 2,
                 individual: bool = False):
        super().__init__()
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.individual = bool(individual)
        self.linear = (
            nn.ModuleList([nn.Linear(obs_len, pred_len) for _ in range(channels)])
            if individual else nn.Linear(obs_len, pred_len)
        )
        for layer in (self.linear if individual else [self.linear]):
            nn.init.constant_(layer.weight, 1.0 / obs_len)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        if self.individual:
            out = torch.stack(
                [self.linear[i](x[:, i]) for i in range(self.channels)], dim=1
            )
        else:
            out = self.linear(x)
        return out.permute(0, 2, 1)


class SocialContextEncoder(nn.Module):
    """Lightweight masked encoder for per-frame pedestrian neighbors.

    ``neighbor_rel`` stores target-centered neighbor positions independently at
    each time step. Neighbor slots are not assumed to preserve identity across
    time, so temporal differences are computed only after permutation-invariant
    within-frame aggregation.
    """

    def __init__(self, obs_len: int, hidden_dim: int = 32, out_dim: int = 32):
        super().__init__()
        self.obs_len = int(obs_len)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.relation_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.score = nn.Linear(hidden_dim, 1)
        self.temporal = nn.Sequential(
            nn.Linear(self.obs_len * hidden_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, neighbor_rel: torch.Tensor,
                neighbor_mask: torch.Tensor) -> torch.Tensor:
        if neighbor_rel.ndim != 4 or neighbor_rel.size(-1) != 2:
            raise ValueError(
                "neighbor_rel must have shape [B, T_obs, K_neighbor, 2]"
            )
        if neighbor_mask.shape != neighbor_rel.shape[:-1]:
            raise ValueError(
                "neighbor_mask must have shape [B, T_obs, K_neighbor]"
            )
        if neighbor_rel.size(1) != self.obs_len:
            raise ValueError(
                f"Expected T_obs={self.obs_len}, got {neighbor_rel.size(1)}"
            )

        mask = neighbor_mask.to(dtype=neighbor_rel.dtype)
        distance = torch.linalg.norm(neighbor_rel, dim=-1, keepdim=True)
        unit = neighbor_rel / distance.clamp_min(1e-3)
        relation_feature = torch.cat(
            [neighbor_rel, distance, unit], dim=-1
        )
        h = self.relation_mlp(relation_feature)

        valid = mask > 0.5
        score = self.score(h).squeeze(-1)
        score = score.masked_fill(~valid, -1e4)
        weight = torch.softmax(score, dim=-1) * mask
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = (h * weight.unsqueeze(-1)).sum(dim=2)

        # For frames with no valid neighbor, force the pooled representation to
        # exactly zero rather than relying on attention numerics.
        frame_valid = (mask.sum(dim=-1, keepdim=True) > 0).to(pooled.dtype)
        pooled = pooled * frame_valid
        return self.temporal(pooled.reshape(pooled.size(0), -1))


class MultiHypothesisHead(nn.Module):
    """K-mode residual prediction head with confidence and log-variance."""

    def __init__(self, obs_len: int, pred_len: int, hidden_dim: int = 64,
                 num_modes: int = 20, use_ego_context: bool = False,
                 context_dim: int = 0):
        super().__init__()
        self.K = int(num_modes)
        self.T = int(pred_len)
        self.use_ego_context = bool(use_ego_context)
        self.context_dim = int(context_dim)
        in_dim = pred_len * 2 + obs_len * 2 + 5 + self.context_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.offset = nn.Linear(hidden_dim, num_modes * pred_len * 2)
        self.logvar = nn.Linear(hidden_dim, num_modes * pred_len * 2)
        self.logit = nn.Linear(hidden_dim, num_modes)

    def forward(self, mean_pred: torch.Tensor, obs: torch.Tensor,
                ego_obs_rel: torch.Tensor | None = None,
                ego_valid: torch.Tensor | None = None,
                context_feature: torch.Tensor | None = None):
        batch_size = obs.size(0)
        descriptor = kinematic_features(
            obs, ego_obs_rel=ego_obs_rel, ego_valid=ego_valid,
            use_ego_context=self.use_ego_context,
        )
        parts = [
            mean_pred.reshape(batch_size, -1),
            obs.reshape(batch_size, -1),
            descriptor,
        ]
        if self.context_dim > 0:
            if context_feature is None:
                context_feature = obs.new_zeros((batch_size, self.context_dim))
            if context_feature.shape != (batch_size, self.context_dim):
                raise ValueError(
                    f"context_feature must have shape "
                    f"[{batch_size}, {self.context_dim}]"
                )
            parts.append(context_feature)
        h = self.mlp(torch.cat(parts, dim=1))
        offsets = self.offset(h).view(batch_size, self.K, self.T, 2)
        log_vars = self.logvar(h).view(batch_size, self.K, self.T, 2)
        logits = self.logit(h)
        candidates = mean_pred.unsqueeze(1) + offsets
        top_idx = logits.argmax(dim=1)
        top1 = candidates[torch.arange(batch_size, device=obs.device), top_idx]
        return candidates, logits, top1, log_vars


def augment_batch(obs: torch.Tensor, target: torch.Tensor,
                  ego_obs_rel: torch.Tensor | None = None,
                  noise_std: float = 0.02, mask_prob: float = 0.3,
                  max_mask_steps: int = 4,
                  rotation_max_deg: float = 180.0,
                  use_inversion: bool = False):
    """Apply geometry-consistent training augmentation.

    ``rotation_max_deg=180`` samples angles over the full planar circle
    ``[-pi, pi]``. Central inversion is disabled by default because inversion is
    already contained in that rotation distribution. It remains available as an
    explicit ablation when a narrower rotation interval is used.
    """
    batch_size, obs_len, _ = obs.shape
    device = obs.device
    max_rad = math.radians(float(rotation_max_deg))
    angles = (torch.rand(batch_size, device=device) * 2.0 - 1.0) * max_rad
    obs_aug = _rotate_per_sample(obs, angles)
    target_aug = _rotate_per_sample(target, angles)
    ego_aug = (_rotate_per_sample(ego_obs_rel, angles)
               if ego_obs_rel is not None else None)

    if use_inversion:
        sign = torch.where(
            torch.rand(batch_size, 1, 1, device=device) < 0.5,
            obs.new_tensor(-1.0), obs.new_tensor(1.0),
        )
        obs_aug = obs_aug * sign
        target_aug = target_aug * sign
        if ego_aug is not None:
            ego_aug = ego_aug * sign

    if mask_prob > 0.0 and max_mask_steps > 0:
        max_steps = min(int(max_mask_steps), obs_len)
        do_mask = torch.rand(batch_size, device=device) < float(mask_prob)
        mask_len = torch.randint(1, max_steps + 1, (batch_size,), device=device)
        steps = torch.arange(obs_len, device=device).unsqueeze(0)
        keep = (steps >= mask_len.unsqueeze(1)) | (~do_mask.unsqueeze(1))
        obs_aug = obs_aug * keep.unsqueeze(-1).to(obs_aug.dtype)

    if noise_std > 0.0:
        obs_aug = obs_aug + torch.randn_like(obs_aug) * float(noise_std)

    return obs_aug, target_aug, ego_aug


def augment_batch_social(obs: torch.Tensor, target: torch.Tensor,
                         neighbor_rel: torch.Tensor,
                         neighbor_mask: torch.Tensor,
                         ego_obs_rel: torch.Tensor | None = None,
                         noise_std: float = 0.02, mask_prob: float = 0.3,
                         max_mask_steps: int = 4,
                         rotation_max_deg: float = 180.0,
                         use_inversion: bool = False):
    """Geometry-consistent augmentation for target and social context.

    ``neighbor_rel`` is defined as neighbor minus target. The same planar
    transform is therefore applied to it. When target observation noise is
    injected, the identical perturbation is subtracted from neighbor-relative
    positions so that the relation remains consistent with the noisy target.
    """
    batch_size, obs_len, _ = obs.shape
    device = obs.device
    max_rad = math.radians(float(rotation_max_deg))
    angles = (torch.rand(batch_size, device=device) * 2.0 - 1.0) * max_rad

    obs_aug = _rotate_per_sample(obs, angles)
    target_aug = _rotate_per_sample(target, angles)
    neighbor_aug = _rotate_per_sample(neighbor_rel, angles)
    mask_aug = neighbor_mask.to(dtype=obs.dtype).clone()
    ego_aug = (_rotate_per_sample(ego_obs_rel, angles)
               if ego_obs_rel is not None else None)

    if use_inversion:
        sign = torch.where(
            torch.rand(batch_size, 1, 1, device=device) < 0.5,
            obs.new_tensor(-1.0), obs.new_tensor(1.0),
        )
        obs_aug = obs_aug * sign
        target_aug = target_aug * sign
        neighbor_aug = neighbor_aug * sign.unsqueeze(2)
        if ego_aug is not None:
            ego_aug = ego_aug * sign

    if mask_prob > 0.0 and max_mask_steps > 0:
        max_steps = min(int(max_mask_steps), obs_len)
        do_mask = torch.rand(batch_size, device=device) < float(mask_prob)
        mask_len = torch.randint(1, max_steps + 1, (batch_size,), device=device)
        steps = torch.arange(obs_len, device=device).unsqueeze(0)
        keep = (steps >= mask_len.unsqueeze(1)) | (~do_mask.unsqueeze(1))
        obs_aug = obs_aug * keep.unsqueeze(-1).to(obs_aug.dtype)
        neighbor_aug = neighbor_aug * keep[:, :, None, None].to(neighbor_aug.dtype)
        mask_aug = mask_aug * keep[:, :, None].to(mask_aug.dtype)

    if noise_std > 0.0:
        noise = torch.randn_like(obs_aug) * float(noise_std)
        obs_aug = obs_aug + noise
        neighbor_aug = neighbor_aug - noise.unsqueeze(2) * mask_aug.unsqueeze(-1)

    neighbor_aug = neighbor_aug * mask_aug.unsqueeze(-1)
    return obs_aug, target_aug, neighbor_aug, mask_aug, ego_aug


def _view_angles(n_views: int) -> list[float]:
    if n_views < 1:
        raise ValueError("n_views must be at least 1")
    return [2.0 * math.pi * i / n_views for i in range(n_views)]


@torch.no_grad()
def symmetry_view_candidates(model: nn.Module, obs: torch.Tensor,
                             ego_obs_rel: torch.Tensor | None = None,
                             ego_valid: torch.Tensor | None = None,
                             neighbor_rel: torch.Tensor | None = None,
                             neighbor_mask: torch.Tensor | None = None,
                             n_views: int = 1):
    """Generate confidence-scored candidates from discrete rotation views.

    Returns:
        top1:              highest-confidence trajectory over all views
        merged_candidates: [B, K*n_views, T, 2]
        merged_log_scores: normalized log scores over the merged pool
    """
    all_candidates: list[torch.Tensor] = []
    all_log_scores: list[torch.Tensor] = []
    for angle in _view_angles(int(n_views)):
        obs_t = rotate_2d(obs, angle)
        ego_t = rotate_2d(ego_obs_rel, angle) if ego_obs_rel is not None else None
        neighbor_t = (
            rotate_2d(neighbor_rel, angle) if neighbor_rel is not None else None
        )
        out = model(
            obs_t, ego_obs_rel=ego_t, ego_valid=ego_valid,
            neighbor_rel=neighbor_t, neighbor_mask=neighbor_mask,
        )
        if "candidates" in out:
            candidates = rotate_2d(out["candidates"], -angle)
            # Each transformed view contributes equal prior mass. Per-view
            # log-softmax avoids arbitrary cross-view logit offsets.
            log_scores = torch.log_softmax(out["mode_logits"], dim=1) - math.log(n_views)
        else:
            candidates = rotate_2d(out["pred"], -angle).unsqueeze(1)
            log_scores = obs.new_zeros((obs.size(0), 1)) - math.log(n_views)
        all_candidates.append(candidates)
        all_log_scores.append(log_scores)

    merged = torch.cat(all_candidates, dim=1)
    scores = torch.cat(all_log_scores, dim=1)
    top_idx = scores.argmax(dim=1)
    top1 = merged[torch.arange(obs.size(0), device=obs.device), top_idx]
    return top1, merged, scores


@torch.no_grad()
def symmetry_tta_candidates(model: nn.Module, obs: torch.Tensor,
                            n_rotations: int = 8, use_negation: bool = False,
                            ego_obs_rel: torch.Tensor | None = None,
                            ego_valid: torch.Tensor | None = None,
                            neighbor_rel: torch.Tensor | None = None,
                            neighbor_mask: torch.Tensor | None = None):
    """Backward-compatible wrapper around rotation-view candidate generation.

    The main experiments should keep ``use_negation=False`` when the rotations
    cover the full circle. Enabling negation is retained only for legacy ablation.
    """
    top1, merged, scores = symmetry_view_candidates(
        model, obs, ego_obs_rel=ego_obs_rel, ego_valid=ego_valid,
        neighbor_rel=neighbor_rel, neighbor_mask=neighbor_mask,
        n_views=n_rotations,
    )
    if not use_negation:
        return top1, merged

    top_neg, merged_neg, scores_neg = symmetry_view_candidates(
        model, -obs,
        ego_obs_rel=(-ego_obs_rel if ego_obs_rel is not None else None),
        ego_valid=ego_valid,
        neighbor_rel=(-neighbor_rel if neighbor_rel is not None else None),
        neighbor_mask=neighbor_mask,
        n_views=n_rotations,
    )
    merged_neg = -merged_neg
    top_neg = -top_neg
    merged_all = torch.cat([merged, merged_neg], dim=1)
    scores_all = torch.cat([scores - math.log(2.0), scores_neg - math.log(2.0)], dim=1)
    top_idx = scores_all.argmax(dim=1)
    top_all = merged_all[torch.arange(obs.size(0), device=obs.device), top_idx]
    return top_all, merged_all


@torch.no_grad()
def select_coverage_optimal_k(candidates: torch.Tensor, K: int = 20):
    """Select K geometrically diverse trajectories via farthest-point sampling."""
    batch_size, num_candidates, pred_len, channels = candidates.shape
    if K < 1 or K > num_candidates:
        raise ValueError(f"K={K} must satisfy 1 <= K <= {num_candidates}")
    device = candidates.device
    flat = candidates.reshape(batch_size, num_candidates, pred_len * channels).cpu()
    dists = torch.cdist(flat, flat)
    indices = torch.zeros(batch_size, K, dtype=torch.long)
    row = torch.arange(batch_size)
    indices[:, 0] = dists.sum(dim=2).argmin(dim=1)
    min_dist = dists[row, indices[:, 0]]
    for k in range(1, K):
        min_dist.scatter_(1, indices[:, k - 1:k], -1.0)
        indices[:, k] = min_dist.argmax(dim=1)
        min_dist = torch.minimum(min_dist, dists[row, indices[:, k]])
    indices = indices.to(device)
    gather_idx = indices[:, :, None, None].expand(-1, -1, pred_len, channels)
    return torch.gather(candidates, 1, gather_idx), indices


@torch.no_grad()
def select_confidence_diverse_k(candidates: torch.Tensor,
                                log_scores: torch.Tensor | None,
                                K: int = 20,
                                confidence_weight: float = 0.5):
    """Fixed-budget selection balancing mode confidence and trajectory spread.

    A high-confidence trajectory seeds the set. Subsequent selections maximize
    a convex combination of normalized confidence and minimum distance to the
    current set. This avoids the central-seed bias of pure FPS while retaining
    geometrically distinct hypotheses.
    """
    batch_size, num_candidates, pred_len, channels = candidates.shape
    if K < 1 or K > num_candidates:
        raise ValueError(f"K={K} must satisfy 1 <= K <= {num_candidates}")
    weight = float(confidence_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("confidence_weight must be in [0, 1]")

    device = candidates.device
    flat = candidates.reshape(batch_size, num_candidates, pred_len * channels).cpu()
    dists = torch.cdist(flat, flat)
    max_dist = dists.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    dists = dists / max_dist
    if log_scores is None:
        confidence = torch.full(
            (batch_size, num_candidates), 1.0 / num_candidates,
            dtype=flat.dtype,
        )
    else:
        confidence = torch.softmax(log_scores.detach().cpu(), dim=1)
        c_min = confidence.amin(dim=1, keepdim=True)
        c_max = confidence.amax(dim=1, keepdim=True)
        confidence = (confidence - c_min) / (c_max - c_min).clamp_min(1e-8)

    row = torch.arange(batch_size)
    indices = torch.zeros(batch_size, K, dtype=torch.long)
    indices[:, 0] = confidence.argmax(dim=1)
    min_dist = dists[row, indices[:, 0]]
    selected_mask = torch.zeros(batch_size, num_candidates, dtype=torch.bool)
    selected_mask[row, indices[:, 0]] = True

    for k in range(1, K):
        objective = weight * confidence + (1.0 - weight) * min_dist
        objective[selected_mask] = -1.0
        indices[:, k] = objective.argmax(dim=1)
        selected_mask[row, indices[:, k]] = True
        min_dist = torch.minimum(min_dist, dists[row, indices[:, k]])

    indices = indices.to(device)
    gather_idx = indices[:, :, None, None].expand(-1, -1, pred_len, channels)
    return torch.gather(candidates, 1, gather_idx), indices


@torch.no_grad()
def anytime_symmetry_candidates(model: nn.Module, obs: torch.Tensor,
                                ego_obs_rel: torch.Tensor | None = None,
                                ego_valid: torch.Tensor | None = None,
                                neighbor_rel: torch.Tensor | None = None,
                                neighbor_mask: torch.Tensor | None = None,
                                deadline_ms: float = 50.0,
                                max_views: int = 16):
    """Sequential batch-one inference that stops at an end-to-end deadline.

    This function is intended for deployment demonstrations. Benchmark tables
    should additionally report deterministic view-count sweeps because runtime
    stopping decisions vary across hardware and background load.
    """
    if obs.size(0) != 1:
        raise ValueError("anytime_symmetry_candidates requires batch size 1")
    if deadline_ms <= 0:
        raise ValueError("deadline_ms must be positive")

    start = time.perf_counter()
    candidates: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    used = 0
    for angle in _view_angles(max_views):
        if used > 0 and (time.perf_counter() - start) * 1000.0 >= deadline_ms:
            break
        obs_t = rotate_2d(obs, angle)
        ego_t = rotate_2d(ego_obs_rel, angle) if ego_obs_rel is not None else None
        neighbor_t = (
            rotate_2d(neighbor_rel, angle) if neighbor_rel is not None else None
        )
        out = model(
            obs_t, ego_obs_rel=ego_t, ego_valid=ego_valid,
            neighbor_rel=neighbor_t, neighbor_mask=neighbor_mask,
        )
        c = out.get("candidates", out["pred"].unsqueeze(1))
        c = rotate_2d(c, -angle)
        if "mode_logits" in out:
            s = torch.log_softmax(out["mode_logits"], dim=1)
        else:
            s = c.new_zeros((1, c.size(1)))
        candidates.append(c)
        scores.append(s)
        used += 1
        if obs.device.type == "cuda":
            torch.cuda.synchronize(obs.device)

    merged = torch.cat(candidates, dim=1)
    merged_scores = torch.cat(scores, dim=1) - math.log(max(used, 1))
    top_idx = merged_scores.argmax(dim=1)
    top1 = merged[torch.arange(1, device=obs.device), top_idx]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return top1, merged, merged_scores, used, elapsed_ms
