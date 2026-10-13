from __future__ import annotations

import torch.nn as nn

from .components import MultiHypothesisHead, SocialContextEncoder, TimeLinearCore
from .control_backbones import (
    MultiHypGRUModel, MultiHypGRUSocialModel,
    MultiHypTransformerModel, MultiHypTransformerSocialModel,
)


class BaselineModel(nn.Module):
    supports_multimodal = False
    uses_ego_context = False
    uses_social_context = False

    def __init__(self, obs_len, pred_len, channels, individual, **kwargs):
        super().__init__()
        self.base = TimeLinearCore(obs_len, pred_len, channels, individual)

    def forward(self, obs, **kwargs):
        return {"pred": self.base(obs)}


class MultiHypModel(nn.Module):
    supports_multimodal = True
    uses_social_context = False

    def __init__(self, obs_len, pred_len, channels, individual,
                 num_modes=20, hyp_hidden_dim=64,
                 use_ego_context=False, **kwargs):
        super().__init__()
        self.uses_ego_context = bool(use_ego_context)
        self.base = TimeLinearCore(obs_len, pred_len, channels, individual)
        self.head = MultiHypothesisHead(
            obs_len, pred_len, hidden_dim=hyp_hidden_dim,
            num_modes=num_modes, use_ego_context=self.uses_ego_context,
        )

    def forward(self, obs, ego_obs_rel=None, ego_valid=None, **kwargs):
        mean_pred = self.base(obs)
        candidates, logits, top1, log_vars = self.head(
            mean_pred, obs, ego_obs_rel=ego_obs_rel, ego_valid=ego_valid,
        )
        return {
            "pred": top1,
            "mean_pred": mean_pred,
            "candidates": candidates,
            "mode_logits": logits,
            "log_vars": log_vars,
        }


class MultiHypEgoModel(MultiHypModel):
    uses_ego_context = True

    def __init__(self, *args, **kwargs):
        kwargs["use_ego_context"] = True
        super().__init__(*args, **kwargs)


class MultiHypSocialModel(nn.Module):
    """Trajectory trend + lightweight social-context residual predictor."""

    supports_multimodal = True
    uses_ego_context = False
    uses_social_context = True

    def __init__(self, obs_len, pred_len, channels, individual,
                 num_modes=20, hyp_hidden_dim=64,
                 social_hidden_dim=32, social_dim=32, **kwargs):
        super().__init__()
        self.obs_len = int(obs_len)
        self.base = TimeLinearCore(obs_len, pred_len, channels, individual)
        self.social_encoder = SocialContextEncoder(
            obs_len=obs_len, hidden_dim=social_hidden_dim, out_dim=social_dim,
        )
        self.head = MultiHypothesisHead(
            obs_len, pred_len, hidden_dim=hyp_hidden_dim, num_modes=num_modes,
            use_ego_context=False, context_dim=social_dim,
        )

    def forward(self, obs, neighbor_rel=None, neighbor_mask=None, **kwargs):
        batch_size = obs.size(0)
        if neighbor_rel is None:
            neighbor_rel = obs.new_zeros((batch_size, self.obs_len, 1, 2))
        if neighbor_mask is None:
            neighbor_mask = obs.new_zeros((batch_size, self.obs_len, neighbor_rel.size(2)))

        mean_pred = self.base(obs)
        social_feature = self.social_encoder(neighbor_rel, neighbor_mask)
        candidates, logits, top1, log_vars = self.head(
            mean_pred, obs, context_feature=social_feature,
        )
        return {
            "pred": top1,
            "mean_pred": mean_pred,
            "candidates": candidates,
            "mode_logits": logits,
            "log_vars": log_vars,
            "social_feature": social_feature,
        }


VARIANT_TO_MODEL = {
    "baseline": BaselineModel,
    "multihyp": MultiHypModel,
    "multihyp_ego": MultiHypEgoModel,
    "multihyp_social": MultiHypSocialModel,
    "multihyp_gru": MultiHypGRUModel,
    "multihyp_transformer": MultiHypTransformerModel,
    "multihyp_gru_social": MultiHypGRUSocialModel,
    "multihyp_transformer_social": MultiHypTransformerSocialModel,
}


def build_model(variant, **kwargs):
    try:
        cls = VARIANT_TO_MODEL[str(variant)]
    except KeyError as exc:
        raise ValueError(f"Unknown variant={variant}. Available: {tuple(VARIANT_TO_MODEL)}") from exc
    return cls(**kwargs)
