"""Architecture modules for diffusion-based segmentation."""

from .diffusion_model import DiffusionModel
from .supervised_model import SupervisedModel

__all__ = [
    'SupervisedModel',
    'DiffusionModel'
]
