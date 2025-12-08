"""Dataset registry with metadata."""

from dataclasses import dataclass
from typing import Type

from src.data.octa500 import OCTA500_3M_DataModule, OCTA500_6M_DataModule
from src.data.rossa import ROSSA_DataModule
from src.data.xca import XCA_DataModule


@dataclass
class DatasetInfo:
    """Dataset metadata for automatic configuration."""
    name: str
    class_ref: Type  # DataModule class
    modality: str  # 'OCTA', 'OCT', 'Fundus'
    resolution: tuple  # (height, width)
    num_train: int
    num_val: int
    num_test: int
    description: str
    default_crop_size: int = 224
    default_batch_size: int = 8
    default_train_dir: str = None
    default_val_dir: str = None
    default_test_dir: str = None


# Dataset Registry with full metadata
DATASET_REGISTRY = {
    'octa500_3m': DatasetInfo(
        name='octa500_3m',
        class_ref=OCTA500_3M_DataModule,
        modality='OCTA',
        resolution=(304, 304),
        num_train=200,
        num_val=50,
        num_test=50,
        description='OCTA-500 3x3mm vessel segmentation',
        default_crop_size=224,
        default_batch_size=8,
        default_train_dir='data/OCTA500_3M/train',
        default_val_dir='data/OCTA500_3M/val',
        default_test_dir='data/OCTA500_3M/test',
    ),
    'octa500_6m': DatasetInfo(
        name='octa500_6m',
        class_ref=OCTA500_6M_DataModule,
        modality='OCTA',
        resolution=(400, 400),
        num_train=200,
        num_val=50,
        num_test=50,
        description='OCTA-500 6x6mm vessel segmentation',
        default_crop_size=224,
        default_batch_size=8,
        default_train_dir='data/OCTA500_6M/train',
        default_val_dir='data/OCTA500_6M/val',
        default_test_dir='data/OCTA500_6M/test',
    ),
    'rossa': DatasetInfo(
        name='rossa',
        class_ref=ROSSA_DataModule,
        modality='OCTA',
        resolution=(304, 304),
        num_train=200,
        num_val=50,
        num_test=50,
        description='ROSSA FAZ segmentation',
        default_crop_size=224,
        default_batch_size=8,
        default_train_dir='data/ROSSA/train_manual',
        default_val_dir='data/ROSSA/val',
        default_test_dir='data/ROSSA/test',
    ),
    'xca': DatasetInfo(
        name='xca',
        class_ref=XCA_DataModule,
        modality='XCA',
        resolution=(512, 512),
        num_train=155,
        num_val=20,
        num_test=46,
        description='XCA full vessel segmentation dataset',
        default_crop_size=320,
        default_batch_size=4,
        default_train_dir='data/xca_full/train',
        default_val_dir='data/xca_full/val',
        default_test_dir='data/xca_full/test',
    ),
}


def get_dataset_info(dataset_name: str) -> DatasetInfo:
    """Get dataset metadata."""
    if dataset_name not in DATASET_REGISTRY:
        available = ', '.join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    return DATASET_REGISTRY[dataset_name]


def list_datasets() -> list[str]:
    """List all available datasets."""
    return list(DATASET_REGISTRY.keys())
