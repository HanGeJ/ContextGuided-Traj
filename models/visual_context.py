from __future__ import annotations

import torch
import torch.nn as nn

from .components import MultiHypothesisHead, SocialContextEncoder, TimeLinearCore


class SceneVisualEncoder(nn.Module):
    """Compact ImageNet backbone with a low-dimensional projection head."""

    def __init__(self, backbone: str = "mobilenet_v3_small", out_dim: int = 64,
                 pretrained: bool = True, trainable: str = "full"):
        super().__init__()
        try:
            from torchvision.models import (
                ConvNeXt_Tiny_Weights,
                MobileNet_V3_Small_Weights,
                ResNet18_Weights,
                convnext_tiny,
                mobilenet_v3_small,
                resnet18,
            )
        except Exception as exc:
            raise RuntimeError("torchvision is required for Stage 3B visual models") from exc

        backbone = str(backbone)
        trainable = str(trainable)
        if trainable not in ("full", "last", "frozen"):
            raise ValueError("trainable must be one of: full, last, frozen")
        self.backbone_name = backbone
        self.trainable_mode = trainable

        if backbone == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = mobilenet_v3_small(weights=weights)
            self.features = net.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            feature_dim = 576
            if trainable == "last":
                for p in self.features.parameters():
                    p.requires_grad = False
                # Unfreeze the final two inverted-residual blocks.
                for block in list(self.features.children())[-2:]:
                    for p in block.parameters():
                        p.requires_grad = True
        elif backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            net = resnet18(weights=weights)
            self.features = nn.Sequential(*list(net.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d(1)
            feature_dim = 512
            if trainable == "last":
                for p in self.features.parameters():
                    p.requires_grad = False
                # layer4 is the final residual stage in the truncated sequential.
                for p in self.features[-1].parameters():
                    p.requires_grad = True
        elif backbone == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            net = convnext_tiny(weights=weights)
            self.features = net.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            feature_dim = 768
            if trainable == "last":
                for p in self.features.parameters():
                    p.requires_grad = False
                # Unfreeze the final ConvNeXt stage.
                for p in self.features[-1].parameters():
                    p.requires_grad = True
        else:
            raise ValueError(f"Unsupported visual backbone: {backbone}")

        if trainable == "frozen":
            for p in self.features.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(feature_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )
        self.out_dim = int(out_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h = self.features(image)
        h = self.pool(h).flatten(1)
        return self.proj(h)


class ContextResidualAdapter(nn.Module):
    """Context-specific residual correction for a trajectory-only candidate set.

    The adapter is zero-initialized at its final layers. At initialization the
    model exactly recovers the motion-only candidate geometry and logits, then
    learns only context-dependent deviations from that strong prior.
    """

    def __init__(self, obs_len: int, pred_len: int, num_modes: int,
                 context_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.K = int(num_modes)
        self.T = int(pred_len)
        in_dim = int(context_dim) + obs_len * 2 + pred_len * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.correction = nn.Linear(hidden_dim, num_modes * pred_len * 2)
        self.gate = nn.Linear(hidden_dim, num_modes)
        self.logit_delta = nn.Linear(hidden_dim, num_modes)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.logit_delta.weight)
        nn.init.zeros_(self.logit_delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, obs: torch.Tensor, mean_pred: torch.Tensor,
                context: torch.Tensor):
        b = obs.size(0)
        h = self.mlp(torch.cat([
            obs.reshape(b, -1), mean_pred.reshape(b, -1), context,
        ], dim=1))
        correction = self.correction(h).view(b, self.K, self.T, 2)
        # Initial sigmoid(0)=0.5 while correction starts exactly zero. The gate
        # remains interpretable and can specialize by hypothesis.
        gate = torch.sigmoid(self.gate(h)).view(b, self.K, 1, 1)
        logit_delta = self.logit_delta(h)
        return gate * correction, logit_delta, gate.squeeze(-1).squeeze(-1)


class VisualConcatModel(nn.Module):
    supports_multimodal = True
    uses_social_context = False
    uses_visual_context = True

    def __init__(self, obs_len=8, pred_len=12, channels=2, individual=False,
                 num_modes=20, hyp_hidden_dim=64, visual_dim=64,
                 visual_backbone="mobilenet_v3_small", pretrained_backbone=True,
                 backbone_trainable="full", social_hidden_dim=32, social_dim=32,
                 use_social=False, **kwargs):
        super().__init__()
        self.obs_len = int(obs_len)
        self.use_social = bool(use_social)
        self.uses_social_context = self.use_social
        self.base = TimeLinearCore(obs_len, pred_len, channels, individual)
        self.visual_encoder = SceneVisualEncoder(
            visual_backbone, visual_dim, pretrained_backbone, backbone_trainable,
        )
        context_dim = int(visual_dim)
        if self.use_social:
            self.social_encoder = SocialContextEncoder(obs_len, social_hidden_dim, social_dim)
            context_dim += int(social_dim)
        self.head = MultiHypothesisHead(
            obs_len, pred_len, hidden_dim=hyp_hidden_dim, num_modes=num_modes,
            context_dim=context_dim,
        )

    def forward(self, obs, scene_crop=None, neighbor_rel=None, neighbor_mask=None, **kwargs):
        if scene_crop is None:
            raise ValueError("scene_crop is required for visual-context models")
        mean_pred = self.base(obs)
        visual = self.visual_encoder(scene_crop)
        context_parts = [visual]
        social = None
        if self.use_social:
            b = obs.size(0)
            if neighbor_rel is None:
                neighbor_rel = obs.new_zeros((b, self.obs_len, 1, 2))
                neighbor_mask = obs.new_zeros((b, self.obs_len, 1))
            social = self.social_encoder(neighbor_rel, neighbor_mask)
            context_parts.append(social)
        context = torch.cat(context_parts, dim=1)
        candidates, logits, top1, log_vars = self.head(mean_pred, obs, context_feature=context)
        out = {
            "pred": top1, "mean_pred": mean_pred, "candidates": candidates,
            "mode_logits": logits, "log_vars": log_vars,
            "visual_feature": visual, "context_feature": context,
        }
        if social is not None:
            out["social_feature"] = social
        return out


class VisualGatedResidualModel(nn.Module):
    supports_multimodal = True
    uses_social_context = False
    uses_visual_context = True

    def __init__(self, obs_len=8, pred_len=12, channels=2, individual=False,
                 num_modes=20, hyp_hidden_dim=64, visual_dim=64,
                 visual_backbone="mobilenet_v3_small", pretrained_backbone=True,
                 backbone_trainable="full", social_hidden_dim=32, social_dim=32,
                 context_hidden_dim=128, use_social=False, **kwargs):
        super().__init__()
        self.obs_len = int(obs_len)
        self.use_social = bool(use_social)
        self.uses_social_context = self.use_social
        self.base = TimeLinearCore(obs_len, pred_len, channels, individual)
        # Motion branch has no context input and remains an explicit trajectory prior.
        self.motion_head = MultiHypothesisHead(
            obs_len, pred_len, hidden_dim=hyp_hidden_dim, num_modes=num_modes,
            context_dim=0,
        )
        # Compatibility with utility code that reads model.head.K.
        self.head = self.motion_head
        self.visual_encoder = SceneVisualEncoder(
            visual_backbone, visual_dim, pretrained_backbone, backbone_trainable,
        )
        context_dim = int(visual_dim)
        if self.use_social:
            self.social_encoder = SocialContextEncoder(obs_len, social_hidden_dim, social_dim)
            context_dim += int(social_dim)
        self.context_adapter = ContextResidualAdapter(
            obs_len, pred_len, num_modes, context_dim, hidden_dim=context_hidden_dim,
        )

    def forward(self, obs, scene_crop=None, neighbor_rel=None, neighbor_mask=None, **kwargs):
        if scene_crop is None:
            raise ValueError("scene_crop is required for visual-context models")
        b = obs.size(0)
        mean_pred = self.base(obs)
        base_candidates, base_logits, _, log_vars = self.motion_head(mean_pred, obs)
        visual = self.visual_encoder(scene_crop)
        parts = [visual]
        social = None
        if self.use_social:
            if neighbor_rel is None:
                neighbor_rel = obs.new_zeros((b, self.obs_len, 1, 2))
                neighbor_mask = obs.new_zeros((b, self.obs_len, 1))
            social = self.social_encoder(neighbor_rel, neighbor_mask)
            parts.append(social)
        context = torch.cat(parts, dim=1)
        correction, logit_delta, gate = self.context_adapter(obs, mean_pred, context)
        candidates = base_candidates + correction
        logits = base_logits + logit_delta
        top_idx = logits.argmax(dim=1)
        top1 = candidates[torch.arange(b, device=obs.device), top_idx]
        out = {
            "pred": top1, "mean_pred": mean_pred, "candidates": candidates,
            "mode_logits": logits, "log_vars": log_vars,
            "visual_feature": visual, "context_feature": context,
            "context_gate": gate,
            "base_candidates": base_candidates,
        }
        if social is not None:
            out["social_feature"] = social
        return out



class WarmStartVisualAdapterModel(nn.Module):
    """Baseline-preserving visual adapter on top of a trajectory generator.

    The module names ``base``, ``head`` and (optionally) ``social_encoder``
    intentionally match the Stage-3B raw-control checkpoints. Loading a
    ``multihyp`` or ``multihyp_social`` checkpoint therefore restores the exact
    trajectory generator before the zero-initialized visual residual adapter is
    trained. This fixes the Stage-3B initialization mismatch where the visual
    model had a zero-initialized context adapter but a randomly initialized
    trajectory generator.
    """

    supports_multimodal = True
    uses_visual_context = True

    def __init__(self, obs_len=8, pred_len=12, channels=2, individual=False,
                 num_modes=20, hyp_hidden_dim=64, visual_dim=64,
                 visual_backbone="mobilenet_v3_small", pretrained_backbone=True,
                 backbone_trainable="frozen", social_hidden_dim=32, social_dim=32,
                 context_hidden_dim=128, use_social_base=False,
                 adapter_uses_social=True, **kwargs):
        super().__init__()
        self.obs_len = int(obs_len)
        self.use_social_base = bool(use_social_base)
        self.uses_social_context = self.use_social_base
        self.adapter_uses_social = bool(adapter_uses_social and self.use_social_base)

        self.base = TimeLinearCore(obs_len, pred_len, channels, individual)
        head_context_dim = 0
        if self.use_social_base:
            self.social_encoder = SocialContextEncoder(
                obs_len, social_hidden_dim, social_dim,
            )
            head_context_dim = int(social_dim)
        self.head = MultiHypothesisHead(
            obs_len, pred_len, hidden_dim=hyp_hidden_dim, num_modes=num_modes,
            context_dim=head_context_dim,
        )

        self.visual_encoder = SceneVisualEncoder(
            visual_backbone, visual_dim, pretrained_backbone, backbone_trainable,
        )
        adapter_context_dim = int(visual_dim)
        if self.adapter_uses_social:
            adapter_context_dim += int(social_dim)
        self.context_adapter = ContextResidualAdapter(
            obs_len, pred_len, num_modes, adapter_context_dim,
            hidden_dim=context_hidden_dim,
        )

    def generator_modules(self):
        modules = [self.base, self.head]
        if self.use_social_base:
            modules.append(self.social_encoder)
        return modules

    def set_generator_trainable(self, trainable: bool, head_only: bool = False):
        """Control the warm-start generator independently from visual context."""
        if head_only:
            for p in self.base.parameters():
                p.requires_grad = False
            if self.use_social_base:
                for p in self.social_encoder.parameters():
                    p.requires_grad = False
            for p in self.head.parameters():
                p.requires_grad = bool(trainable)
            return
        for module in self.generator_modules():
            for p in module.parameters():
                p.requires_grad = bool(trainable)

    def forward(self, obs, scene_crop=None, neighbor_rel=None,
                neighbor_mask=None, **kwargs):
        if scene_crop is None:
            raise ValueError("scene_crop is required for visual-context models")
        b = obs.size(0)
        mean_pred = self.base(obs)
        social = None
        if self.use_social_base:
            if neighbor_rel is None:
                neighbor_rel = obs.new_zeros((b, self.obs_len, 1, 2))
                neighbor_mask = obs.new_zeros((b, self.obs_len, 1))
            social = self.social_encoder(neighbor_rel, neighbor_mask)
            base_candidates, base_logits, _, log_vars = self.head(
                mean_pred, obs, context_feature=social,
            )
        else:
            base_candidates, base_logits, _, log_vars = self.head(mean_pred, obs)

        visual = self.visual_encoder(scene_crop)
        context_parts = [visual]
        if self.adapter_uses_social and social is not None:
            context_parts.append(social)
        context = torch.cat(context_parts, dim=1)
        correction, logit_delta, gate = self.context_adapter(
            obs, mean_pred, context,
        )
        candidates = base_candidates + correction
        logits = base_logits + logit_delta
        top_idx = logits.argmax(dim=1)
        top1 = candidates[torch.arange(b, device=obs.device), top_idx]
        out = {
            "pred": top1,
            "mean_pred": mean_pred,
            "candidates": candidates,
            "mode_logits": logits,
            "log_vars": log_vars,
            "visual_feature": visual,
            "context_feature": context,
            "context_gate": gate,
            "base_candidates": base_candidates,
            "base_mode_logits": base_logits,
            "context_correction": correction,
            "logit_delta": logit_delta,
        }
        if social is not None:
            out["social_feature"] = social
        return out



class VisualCandidateReranker(nn.Module):
    """Independent candidate reranker conditioned on visual/social context.

    The reranker receives a fixed candidate set and only changes mode scores.
    Its final layer is zero initialized, so at initialization the original
    generator ranking is recovered exactly.
    """

    def __init__(self, visual_dim: int, social_dim: int = 0, hidden_dim: int = 128):
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.social_dim = int(social_dim)
        # 10 candidate-level geometric/statistical descriptors + global context.
        in_dim = 10 + self.visual_dim + self.social_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _geometry_features(candidates: torch.Tensor, mean_pred: torch.Tensor,
                           base_logits: torch.Tensor, log_vars: torch.Tensor) -> torch.Tensor:
        # candidates: [B,K,T,2]
        residual = candidates - mean_pred.unsqueeze(1)
        residual_rms = residual.square().mean(dim=(-1, -2)).sqrt()
        endpoint_residual = torch.linalg.norm(residual[:, :, -1], dim=-1)
        step = candidates[:, :, 1:] - candidates[:, :, :-1]
        speed = torch.linalg.norm(step, dim=-1)
        speed_mean = speed.mean(dim=-1)
        speed_std = speed.std(dim=-1, unbiased=False)
        if step.size(2) >= 2:
            accel = torch.linalg.norm(step[:, :, 1:] - step[:, :, :-1], dim=-1).mean(dim=-1)
            a = step[:, :, :-1]
            b = step[:, :, 1:]
            denom = (torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1)).clamp_min(1e-5)
            cross = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).abs() / denom
            turn = cross.clamp_max(1.0).mean(dim=-1)
        else:
            accel = torch.zeros_like(speed_mean)
            turn = torch.zeros_like(speed_mean)
        path_len = speed.sum(dim=-1)
        flat = candidates.flatten(2)
        centroid = flat.mean(dim=1, keepdim=True)
        set_distance = torch.linalg.norm(flat - centroid, dim=-1) / max(1.0, float(flat.size(-1)) ** 0.5)
        logvar_mean = log_vars.mean(dim=(-1, -2))
        base_prob = torch.softmax(base_logits, dim=1)
        return torch.stack([
            base_logits, base_prob, residual_rms, endpoint_residual, path_len,
            speed_mean, speed_std, accel, turn, logvar_mean + set_distance,
        ], dim=-1)

    def forward(self, candidates: torch.Tensor, mean_pred: torch.Tensor,
                base_logits: torch.Tensor, log_vars: torch.Tensor,
                visual_feature: torch.Tensor, social_feature: torch.Tensor | None = None):
        geom = self._geometry_features(candidates, mean_pred, base_logits, log_vars)
        b, k, _ = geom.shape
        ctx = [visual_feature]
        if self.social_dim > 0:
            if social_feature is None:
                social_feature = visual_feature.new_zeros((b, self.social_dim))
            ctx.append(social_feature)
        global_ctx = torch.cat(ctx, dim=-1).unsqueeze(1).expand(-1, k, -1)
        score_delta = self.net(torch.cat([geom, global_ctx], dim=-1)).squeeze(-1)
        return score_delta


class DecoupledWarmVisualModel(WarmStartVisualAdapterModel):
    """Warm-start visual model with decoupled geometry and ranking.

    Stage A learns a bounded visual correction to candidate geometry while the
    original generator logits are preserved. Stage B freezes all candidate
    geometry and trains an independent visual/social reranker. This prevents
    ranking gradients from perturbing the candidate set.
    """

    def __init__(self, *args, correction_scale: float = 0.20,
                 reranker_hidden_dim: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.correction_scale = float(correction_scale)
        visual_dim = int(self.visual_encoder.out_dim)
        social_dim = int(self.social_encoder.out_dim) if self.adapter_uses_social else 0
        self.visual_reranker = VisualCandidateReranker(
            visual_dim=visual_dim, social_dim=social_dim, hidden_dim=reranker_hidden_dim,
        )
        # The legacy logit-delta head is intentionally unused in this model.
        for p in self.context_adapter.logit_delta.parameters():
            p.requires_grad = False

    def _bounded_correction(self, correction: torch.Tensor) -> torch.Tensor:
        if self.correction_scale <= 0:
            return correction
        scale = correction.new_tensor(self.correction_scale)
        return scale * torch.tanh(correction / scale)

    def forward(self, obs, scene_crop=None, neighbor_rel=None, neighbor_mask=None,
                enable_reranker: bool = True, **kwargs):
        if scene_crop is None:
            raise ValueError("scene_crop is required for visual-context models")
        b = obs.size(0)
        mean_pred = self.base(obs)
        social = None
        if self.use_social_base:
            if neighbor_rel is None:
                neighbor_rel = obs.new_zeros((b, self.obs_len, 1, 2))
                neighbor_mask = obs.new_zeros((b, self.obs_len, 1))
            social = self.social_encoder(neighbor_rel, neighbor_mask)
            base_candidates, base_logits, _, log_vars = self.head(
                mean_pred, obs, context_feature=social,
            )
        else:
            base_candidates, base_logits, _, log_vars = self.head(mean_pred, obs)

        visual = self.visual_encoder(scene_crop)
        context_parts = [visual]
        if self.adapter_uses_social and social is not None:
            context_parts.append(social)
        context = torch.cat(context_parts, dim=1)
        raw_correction, _, gate = self.context_adapter(obs, mean_pred, context)
        correction = self._bounded_correction(raw_correction)
        candidates = base_candidates + correction

        rank_delta = self.visual_reranker(
            candidates, mean_pred, base_logits, log_vars, visual,
            social if self.adapter_uses_social else None,
        ) if enable_reranker else torch.zeros_like(base_logits)
        logits = base_logits + rank_delta
        top_idx = logits.argmax(dim=1)
        top1 = candidates[torch.arange(b, device=obs.device), top_idx]
        out = {
            "pred": top1, "mean_pred": mean_pred, "candidates": candidates,
            "mode_logits": logits, "log_vars": log_vars,
            "visual_feature": visual, "context_feature": context,
            "context_gate": gate, "base_candidates": base_candidates,
            "base_mode_logits": base_logits, "context_correction": correction,
            "raw_context_correction": raw_correction, "rank_delta": rank_delta,
        }
        if social is not None:
            out["social_feature"] = social
        return out

    def set_phase(self, phase: str):
        """Select trainable parameters for geometry or ranking optimization."""
        phase = str(phase)
        for p in self.parameters():
            p.requires_grad = False
        if phase == "geometry":
            # Respect SceneVisualEncoder's frozen-backbone choice: projection is
            # always trainable; feature extractor is trainable only if it was
            # configured as full/last.
            for p in self.visual_encoder.proj.parameters():
                p.requires_grad = True
            if self.visual_encoder.trainable_mode in ("full", "last"):
                for p in self.visual_encoder.features.parameters():
                    # For 'last', only parameters that were originally enabled
                    # remain identifiable by module structure; enabling all here
                    # would violate the requested mode, so handle explicitly.
                    p.requires_grad = self.visual_encoder.trainable_mode == "full"
                if self.visual_encoder.trainable_mode == "last":
                    children = list(self.visual_encoder.features.children())
                    tail = children[-2:] if self.visual_encoder.backbone_name == "mobilenet_v3_small" else children[-1:]
                    for block in tail:
                        for p in block.parameters():
                            p.requires_grad = True
            for name, p in self.context_adapter.named_parameters():
                if not name.startswith("logit_delta."):
                    p.requires_grad = True
        elif phase == "ranking":
            for p in self.visual_reranker.parameters():
                p.requires_grad = True
        elif phase == "frozen":
            pass
        else:
            raise ValueError("phase must be geometry, ranking, or frozen")

def build_visual_model(variant: str, **kwargs):
    variant = str(variant)
    if variant == "visual_concat":
        return VisualConcatModel(use_social=False, **kwargs)
    if variant == "social_visual_concat":
        return VisualConcatModel(use_social=True, **kwargs)
    if variant == "visual_gated":
        return VisualGatedResidualModel(use_social=False, **kwargs)
    if variant == "social_visual_gated":
        return VisualGatedResidualModel(use_social=True, **kwargs)
    if variant == "warm_visual_adapter":
        return WarmStartVisualAdapterModel(use_social_base=False, **kwargs)
    if variant == "warm_social_visual_adapter":
        return WarmStartVisualAdapterModel(use_social_base=True, **kwargs)
    if variant == "decoupled_visual_adapter":
        return DecoupledWarmVisualModel(use_social_base=False, **kwargs)
    if variant == "decoupled_social_visual_adapter":
        return DecoupledWarmVisualModel(use_social_base=True, **kwargs)
    raise ValueError(f"Unknown visual variant: {variant}")
