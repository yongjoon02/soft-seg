"""Supervised training script."""

import os
os.environ['NCCL_P2P_DISABLE'] = '1'
import torch
torch.set_float32_matmul_precision('medium')

import autorootcwd
from lightning.pytorch.cli import LightningCLI
from src.archs.supervised_model import SupervisedModel
from script.train_base import parse_config_and_setup_args


if __name__ == "__main__":
    # Parse config and setup arguments
    data_name, DataModuleClass = parse_config_and_setup_args(
        default_config='configs/octa500_3m_supervised_models.yaml'
    )
    
    # Create LightningCLI
    cli = LightningCLI(
        SupervisedModel,
        DataModuleClass,
        save_config_kwargs={'overwrite': True},
    )
