"""Loss functions for segmentation tasks.

Usage:
    from src.losses import SoftCrossEntropyLoss, SoftBCELoss, DiceLoss, FocalDiceLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    """Cross entropy loss that supports soft labels.
    
    For soft labels, converts them to 2-class distribution and computes
    cross entropy with log_softmax.
    
    Args:
        soft_label: Whether to use soft label mode (default: False)
    """
    def __init__(self, soft_label: bool = False):
        super().__init__()
        self.soft_label = soft_label
        self.hard_ce = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) model output logits
            labels: (B, H, W) soft labels in [0, 1] or hard labels
        """
        if not self.soft_label:
            # Hard label mode
            return self.hard_ce(logits, (labels > 0.5).long())
        
        # Soft label mode: convert soft label to 2-class distribution
        soft_target = torch.stack([1 - labels, labels], dim=1)  # (B, 2, H, W)
        
        # Compute soft cross entropy
        log_probs = F.log_softmax(logits, dim=1)  # (B, C, H, W)
        loss = -torch.sum(soft_target * log_probs, dim=1)  # (B, H, W)
        return loss.mean()


class SoftBCELoss(nn.Module):
    """Binary Cross Entropy loss for soft labels.
    
    Converts 2-class logits to probability via softmax, then applies BCE.
    
    Args:
        soft_label: Whether to use soft label mode (default: False)
    """
    def __init__(self, soft_label: bool = False):
        super().__init__()
        self.soft_label = soft_label
        self.hard_ce = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) model output logits (C=2 for binary segmentation)
            labels: (B, H, W) soft labels in [0, 1]
        """
        if not self.soft_label:
            return self.hard_ce(logits, (labels > 0.5).long())
        
        # Get probability for foreground class (class 1)
        probs = F.softmax(logits, dim=1)[:, 1]  # (B, H, W)
        
        # BCE loss with soft labels
        eps = 1e-7
        probs = torch.clamp(probs, eps, 1 - eps)
        loss = -(labels * torch.log(probs) + (1 - labels) * torch.log(1 - probs))
        return loss.mean()


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) or (B, H, W) predictions (probabilities)
            target: (B, H, W) targets
        """
        if pred.dim() == 4:
            pred = F.softmax(pred, dim=1)[:, 1]  # Take foreground probability
        
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalDiceLoss(nn.Module):
    """Combined Focal + Dice loss for segmentation."""
    def __init__(self, wf: float = 0.1, wd: float = 0.9, gamma: float = 2.0):
        super().__init__()
        self.dice = DiceLoss()
        self.wf = wf  # Focal weight
        self.wd = wd  # Dice weight
        self.gamma = gamma  # Focal gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) predictions (probabilities)
            target: (B, H, W) targets
        """
        # Focal loss
        if pred.dim() == 4:
            pred_prob = F.softmax(pred, dim=1)[:, 1]
        else:
            pred_prob = pred
        
        eps = 1e-7
        pred_prob = torch.clamp(pred_prob, eps, 1 - eps)
        
        focal = -target * ((1 - pred_prob) ** self.gamma) * torch.log(pred_prob)
        focal = focal.mean()
        
        # Dice loss
        dice = self.dice(pred, target)
        
        return self.wf * focal + self.wd * dice


__all__ = [
    'SoftCrossEntropyLoss',
    'SoftBCELoss',
    'DiceLoss',
    'FocalDiceLoss',
]


# Register losses after class definitions (to avoid circular import)
def _register_losses():
    """Register built-in losses to registry. Called lazily."""
    try:
        from src.registry import LOSS_REGISTRY
        
        LOSS_REGISTRY.register(name='soft_ce', obj=SoftCrossEntropyLoss, metadata={
            'description': 'Cross entropy loss that supports soft labels',
            'supports_multiclass': True,
            'supports_soft_labels': True,
        })
        LOSS_REGISTRY.register(name='soft_bce', obj=SoftBCELoss, metadata={
            'description': 'Binary Cross Entropy loss for soft labels',
            'supports_multiclass': False,
            'supports_soft_labels': True,
        })
        LOSS_REGISTRY.register(name='dice', obj=DiceLoss, metadata={
            'description': 'Dice loss for segmentation',
            'supports_multiclass': True,
            'supports_soft_labels': True,
        })
        LOSS_REGISTRY.register(name='focal_dice', obj=FocalDiceLoss, metadata={
            'description': 'Combined Focal + Dice loss',
            'supports_multiclass': True,
            'supports_soft_labels': True,
        })
    except ImportError:
        pass  # Registry not available yet
