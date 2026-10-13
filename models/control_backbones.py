from __future__ import annotations

import torch
import torch.nn as nn

from .components import MultiHypothesisHead, SocialContextEncoder


class GRUCore(nn.Module):
    """Compact recurrent control backbone for matched multi-hypothesis tests."""

    def __init__(self, obs_len: int, pred_len: int, channels: int = 2,
                 hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, pred_len * channels)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(obs)
        pred = self.proj(out[:, -1])
        return pred.view(obs.size(0), self.pred_len, self.channels)


class TransformerCore(nn.Module):
    """Small Transformer control backbone with matched output interface."""

    def __init__(self, obs_len: int, pred_len: int, channels: int = 2,
                 hidden_dim: int = 64, num_layers: int = 2,
                 nhead: int = 4, ff_dim: int = 128):
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError("hidden_dim must be divisible by nhead")
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.input_proj = nn.Linear(channels, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, obs_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, pred_len * channels)
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.size(1) != self.obs_len:
            raise ValueError(f"expected obs_len={self.obs_len}, got {obs.size(1)}")
        x = self.input_proj(obs) + self.pos
        x = self.encoder(x)
        x = self.norm(x[:, -1])
        pred = self.proj(x)
        return pred.view(obs.size(0), self.pred_len, self.channels)


class _ControlMultiHypBase(nn.Module):
    supports_multimodal = True
    uses_ego_context = False
    uses_social_context = False

    core_cls = None

    def __init__(self, obs_len, pred_len, channels, individual,
                 num_modes=20, hyp_hidden_dim=64,
                 temporal_hidden_dim=64, **kwargs):
        super().__init__()
        del individual
        if self.core_cls is None:
            raise RuntimeError("core_cls must be defined")
        self.base = self.core_cls(
            obs_len=obs_len,
            pred_len=pred_len,
            channels=channels,
            hidden_dim=temporal_hidden_dim,
        )
        self.head = MultiHypothesisHead(
            obs_len, pred_len,
            hidden_dim=hyp_hidden_dim,
            num_modes=num_modes,
            use_ego_context=False,
        )

    def forward(self, obs, **kwargs):
        mean_pred = self.base(obs)
        candidates, logits, top1, log_vars = self.head(mean_pred, obs)
        return {
            "pred": top1,
            "mean_pred": mean_pred,
            "candidates": candidates,
            "mode_logits": logits,
            "log_vars": log_vars,
        }


class MultiHypGRUModel(_ControlMultiHypBase):
    core_cls = GRUCore


class MultiHypTransformerModel(_ControlMultiHypBase):
    core_cls = TransformerCore


class _ControlMultiHypSocialBase(nn.Module):
    supports_multimodal = True
    uses_ego_context = False
    uses_social_context = True

    core_cls = None

    def __init__(self, obs_len, pred_len, channels, individual,
                 num_modes=20, hyp_hidden_dim=64,
                 social_hidden_dim=32, social_dim=32,
                 temporal_hidden_dim=64, **kwargs):
        super().__init__()
        del individual
        if self.core_cls is None:
            raise RuntimeError("core_cls must be defined")
        self.obs_len = int(obs_len)
        self.base = self.core_cls(
            obs_len=obs_len,
            pred_len=pred_len,
            channels=channels,
            hidden_dim=temporal_hidden_dim,
        )
        self.social_encoder = SocialContextEncoder(
            obs_len=obs_len,
            hidden_dim=social_hidden_dim,
            out_dim=social_dim,
        )
        self.head = MultiHypothesisHead(
            obs_len, pred_len,
            hidden_dim=hyp_hidden_dim,
            num_modes=num_modes,
            use_ego_context=False,
            context_dim=social_dim,
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


class MultiHypGRUSocialModel(_ControlMultiHypSocialBase):
    core_cls = GRUCore


class MultiHypTransformerSocialModel(_ControlMultiHypSocialBase):
    core_cls = TransformerCore
