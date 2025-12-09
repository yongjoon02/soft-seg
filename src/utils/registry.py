"""Registry utilities - redirects to src.registry for backward compatibility.

Note: This file is kept for backward compatibility.
      New code should import from src.registry directly.

Usage (deprecated):
    from src.utils.registry import ARCHS_REGISTRY, DATASET_REGISTRY

Usage (recommended):
    from src.registry import ARCHS_REGISTRY, DATASET_REGISTRY
"""

# Re-export everything from src.registry.base for backward compatibility
from src.registry.base import (
    Registry,
    RegistryEntry,
    DATASET_REGISTRY,
    ARCHS_REGISTRY,
    MODEL_REGISTRY,
    LOSS_REGISTRY,
    METRIC_REGISTRY,
)

__all__ = [
    'Registry',
    'RegistryEntry',
    'DATASET_REGISTRY',
    'ARCHS_REGISTRY',
    'MODEL_REGISTRY',
    'LOSS_REGISTRY',
    'METRIC_REGISTRY',
]
