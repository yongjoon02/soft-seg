"""Flow-matching loss with SAUNA-weighted geometry regularization."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry.losses import register_loss


def _squeeze_channel(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 4 and tensor.shape[1] == 1:
        return tensor.squeeze(1)
    return tensor


def weighted_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Weighted Dice loss for binary segmentation."""
    if weight is None:
        weight = torch.ones_like(pred)

    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    weight_flat = weight.view(weight.size(0), -1)

    intersection = (pred_flat * target_flat * weight_flat).sum(dim=1)
    pred_sum = (pred_flat * weight_flat).sum(dim=1)
    target_sum = (target_flat * weight_flat).sum(dim=1)

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return 1.0 - dice.mean()


@register_loss(
    name="flow_sauna_fm",
    description="Flow matching + SAUNA-weighted BCE/Dice geometry loss",
    supports_multiclass=False,
    supports_soft_labels=True,
)
class FlowSaunaFMLoss(nn.Module):
    """Flow matching loss with SAUNA soft-label weighting and time gating."""

    def __init__(
        self,
        alpha: float = 2.0,
        lambda_geo: float = 0.1,
        eps: float = 1e-7,
        use_hard_ut: bool = True,
        dice_scale: float = 0.1,  # Scale factor for Dice loss gradient
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.lambda_geo = float(lambda_geo)
        self.eps = float(eps)
        self.use_hard_ut = bool(use_hard_ut)
        self.dice_scale = float(dice_scale)

    def forward(
        self,
        v: torch.Tensor,
        ut: torch.Tensor | None,
        xt: torch.Tensor,
        geometry: torch.Tensor,
        t: torch.Tensor | None = None,
        geometry_pred: torch.Tensor | None = None,
        hard_labels: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hard_labels is None:
            hard_labels = geometry

        if self.use_hard_ut:
            if x0 is None:
                raise ValueError("flow_sauna_fm requires x0 when use_hard_ut=True.")
            ut = hard_labels - x0
        elif ut is None:
            raise ValueError("flow_sauna_fm requires ut when use_hard_ut=False.")

        loss_fm = F.mse_loss(v, ut)

        if t is None:
            raise ValueError("flow_sauna_fm requires t to compute x1_pred.")
        t_expanded = t.view(-1, 1, 1, 1)
        x1_pred = xt + (1 - t_expanded) * v

        x1_pred = torch.clamp(x1_pred, 0.0, 1.0)

        target_soft_2d = _squeeze_channel(geometry)
        target_hard_2d = _squeeze_channel(hard_labels)
        x1_pred_2d = _squeeze_channel(x1_pred)

        target_soft_2d = torch.clamp(target_soft_2d, 0.0, 1.0)
        target_hard_2d = torch.clamp(target_hard_2d, 0.0, 1.0)

        soft_weight = 1.0 + self.alpha * target_soft_2d
        if t is None:
            geo_weight = soft_weight
        else:
            # Use t**2 to reduce early-timestep geometry loss impact (conflict mitigation)
            t_weight = (t ** 2).view(-1, 1, 1).expand_as(target_soft_2d)
            geo_weight = soft_weight * t_weight

        x1_pred_clamped = torch.clamp(x1_pred_2d, self.eps, 1 - self.eps)
        bce_per_pixel = -(
            target_hard_2d * torch.log(x1_pred_clamped)
            + (1 - target_hard_2d) * torch.log(1 - x1_pred_clamped)
        )
        bce_loss = (bce_per_pixel * geo_weight).mean()

        dice_loss_raw = weighted_dice_loss(x1_pred_2d, target_hard_2d, weight=geo_weight)
        # Scale Dice loss to reduce gradient magnitude (typically 10x larger than Flow)
        dice_loss = dice_loss_raw * self.dice_scale

        total_loss = loss_fm + self.lambda_geo * (bce_loss + dice_loss)
        return total_loss, {
            "flow": loss_fm,
            "bce": bce_loss,
            "dice": dice_loss_raw,  # Log unscaled for monitoring
        }
