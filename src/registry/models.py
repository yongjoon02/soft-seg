"""Model registry with metadata."""

from dataclasses import dataclass
from typing import Callable, Literal

from src.archs.components import CSNet, DSCNet
from src.archs.components.binomial_diffusion import create_berdiff
from src.archs.components.gaussian_diffusion import create_medsegdiff


@dataclass
class ModelInfo:
    """Model metadata for automatic configuration and documentation."""
    name: str
    class_ref: Callable  # Model class or factory function
    task: Literal['supervised', 'diffusion']
    params: int  # Number of parameters
    speed: str   # 'fast', 'medium', 'slow'
    description: str
    paper_url: str = None
    default_lr: float = 2e-4
    default_epochs: int = 300


# Model Registry with full metadata
MODEL_REGISTRY = {
    'csnet': ModelInfo(
        name='csnet',
        class_ref=CSNet,
        task='supervised',
        params=8_400_196,
        speed='fast',
        description='Channel & Spatial Attention Network (27.19 it/s)',
        default_lr=2e-3,
        default_epochs=300,
    ),
    'dscnet': ModelInfo(
        name='dscnet',
        class_ref=DSCNet,
        task='supervised',
        params=5_843_106,
        speed='medium',
        description='Dynamic Snake Convolution Network (3.80 it/s)',
        default_lr=2e-3,
        default_epochs=300,
    ),
    'medsegdiff': ModelInfo(
        name='medsegdiff',
        class_ref=create_medsegdiff,
        task='diffusion',
        params=16_224_737,
        speed='slow',
        description='MedSegDiff with Gaussian DDPM (stable)',
        default_lr=2e-4,
        default_epochs=500,
    ),
    'berdiff': ModelInfo(
        name='berdiff',
        class_ref=create_berdiff,
        task='diffusion',
        params=9_327_521,
        speed='slow',
        description='Bernoulli Diffusion (binary-optimized)',
        default_lr=2e-4,
        default_epochs=500,
    ),
}


def get_model_info(model_name: str) -> ModelInfo:
    """Get model metadata."""
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    return MODEL_REGISTRY[model_name]


def list_models(task: str = None) -> list[str]:
    """List available models, optionally filtered by task."""
    if task is None:
        return list(MODEL_REGISTRY.keys())
    return [name for name, info in MODEL_REGISTRY.items() if info.task == task]
