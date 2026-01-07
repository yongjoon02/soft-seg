"""MedSegDiff flow backbone with 2-channel output (hard + soft)."""
import math
from typing import Iterable

import torch
import torch.nn as nn

from src.archs.components.diffusion_unet import MedSegDiffUNet
from src.registry.base import ARCHS_REGISTRY


def _attn_flags_from_resolutions(
    image_size: int,
    channel_mult: Iterable[int],
    attn_resolutions: Iterable[int],
) -> tuple[bool, ...]:
    """Convert resolution list to per-level attention flags."""
    attn_set = set(attn_resolutions)
    flags = []
    curr = image_size
    for _ in channel_mult:
        flags.append(curr in attn_set)
        curr = max(1, math.floor(curr / 2))
    return tuple(flags)


@ARCHS_REGISTRY.register(name="medsegdiff_flow_soft2hard")
class MedSegDiffFlowSoft2Hard(nn.Module):
    """MedSegDiff UNet used as a flow-matching backbone (2-channel output)."""

    def __init__(
        self,
        img_resolution: int,
        model_channels: int = 64,
        channel_mult: Iterable[int] = (1, 2, 4, 8),
        channel_mult_emb: int = 4,  # unused (kept for signature parity)
        num_blocks: int = 3,  # unused (MedSegDiff uses fixed depth per stage)
        attn_resolutions: Iterable[int] = (32, 16, 8, 8),
        dropout: float = 0.0,  # unused
        label_dim: int = 0,  # unused
        augment_dim: int = 0,  # unused
        time_scale: float = 1000.0,  # scale flow t (0-1) to diffusion-style range
        **_,
    ):
        super().__init__()
        self.time_scale = float(time_scale)

        full_self_attn = _attn_flags_from_resolutions(
            image_size=img_resolution,
            channel_mult=channel_mult,
            attn_resolutions=attn_resolutions,
        )

        self.base_unet = MedSegDiffUNet(
            dim=model_channels,
            image_size=img_resolution,
            mask_channels=2,
            input_img_channels=1,
            dim_mult=tuple(channel_mult),
            full_self_attn=full_self_attn,
            mid_transformer_depth=1,
        )
        self.soft_head = nn.Conv2d(model_channels, 1, 1)
        self.soft_feat = nn.Sequential(
            nn.Conv2d(1, model_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gate_conv = nn.Conv2d(model_channels, model_channels, 1)
        self.hard_head = nn.Conv2d(model_channels, 1, 1)

    def forward(self, x, time, cond):
        if isinstance(time, torch.Tensor):
            time_in = time * self.time_scale
        else:
            time_in = time * self.time_scale
        _, common_feat = self.base_unet(x, time_in, cond, return_features=True)

        v_soft = self.soft_head(common_feat)
        f_soft = self.soft_feat(v_soft)

        gate = torch.sigmoid(self.gate_conv(f_soft.detach()))
        refined_feat = common_feat * (1.0 + gate)

        v_hard = self.hard_head(refined_feat)
        return torch.cat([v_hard, v_soft], dim=1)
