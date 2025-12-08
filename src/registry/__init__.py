"""Registry system for models and datasets."""

from .datasets import DATASET_REGISTRY, get_dataset_info, list_datasets
from .models import MODEL_REGISTRY, get_model_info, list_models

__all__ = [
    'MODEL_REGISTRY',
    'DATASET_REGISTRY',
    'get_model_info',
    'get_dataset_info',
    'list_models',
    'list_datasets',
]
