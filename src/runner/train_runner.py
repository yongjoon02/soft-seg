"""Training runner - config-based training logic."""

import torch
from pathlib import Path
from typing import Dict, Any
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    StochasticWeightAveraging,
)

from src.registry import get_model_info, get_dataset_info
from src.experiment import ExperimentTracker, EnhancedTensorBoardLogger
from src.archs.supervised_model import SupervisedModel
from src.archs.diffusion_model import DiffusionModel
from src.utils.config import save_config


class TrainRunner:
    """Config-based training runner."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize from config dictionary.
        
        Args:
            config: Full configuration (model, data, trainer sections)
        """
        # Extract config sections
        model_cfg = config['model']
        data_cfg = config['data']
        trainer_cfg = config['trainer']
        
        self.model_name = model_cfg['arch_name']
        self.dataset_name = data_cfg['name']
        
        # Get registry info
        self.model_info = get_model_info(self.model_name)
        self.dataset_info = get_dataset_info(self.dataset_name)
        
        # Store config
        self.config = config
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.trainer_cfg = trainer_cfg
        
        # Extract special flags
        self.tag = config.get('tag', None)
        self.resume = config.get('resume', None)
        self.debug = config.get('debug', False)
        
        # Create experiment tracker
        self.tracker = ExperimentTracker()
    
    def run(self):
        """Execute training."""
        self._print_info()
        
        # Create experiment
        experiment = self.tracker.create_experiment(
            model=self.model_name,
            dataset=self.dataset_name,
            config=self.config,
            tag=self.tag,
        )
        
        print(f"\n{'='*60}")
        print(f"Experiment ID: {experiment.id}")
        print(f"Experiment dir: {experiment.dir}")
        print(f"{'='*60}\n")
        
        # Save config to experiment dir
        save_config(self.config, experiment.dir / "config.yaml")
        
        try:
            # Create components
            datamodule = self._create_datamodule()
            model = self._create_model()
            callbacks = self._create_callbacks(experiment.dir)
            logger = self._create_logger(experiment.dir)
            trainer = self._create_trainer(callbacks, logger)
            
            # Train
            print("Starting training...")
            trainer.fit(model, datamodule, ckpt_path=self.resume)
            
            # Finish experiment
            final_metrics = {
                k: v.item() if torch.is_tensor(v) else v
                for k, v in trainer.callback_metrics.items()
                if 'val/' in k
            }
            best_ckpt = experiment.dir / "checkpoints" / "best.ckpt"
            
            self.tracker.finish_experiment(
                experiment.id,
                final_metrics=final_metrics,
                best_checkpoint=str(best_ckpt) if best_ckpt.exists() else None,
            )
            
            print(f"\n{'='*60}")
            print(f"✅ Training completed!")
            print(f"   Best checkpoint: {best_ckpt}")
            print(f"   Final metrics:")
            for k, v in final_metrics.items():
                print(f"      {k}: {v:.4f}")
            print(f"{'='*60}\n")
            
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            self.tracker.mark_failed(experiment.id, "Interrupted by user")
            raise
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            self.tracker.mark_failed(experiment.id, str(e))
            raise
    
    def _print_info(self):
        """Print training info."""
        print(f"\n{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"  - Task: {self.model_info.task}")
        print(f"  - Parameters: {self.model_info.params:,}")
        print(f"\nDataset: {self.dataset_name}")
        print(f"  - Samples: {self.dataset_info.num_train} train / "
              f"{self.dataset_info.num_val} val / {self.dataset_info.num_test} test")
        print(f"\nHyperparameters:")
        print(f"  - Learning rate: {self.model_cfg.get('learning_rate')}")
        print(f"  - Batch size: {self.data_cfg.get('train_bs')}")
        print(f"  - Crop size: {self.data_cfg.get('crop_size')}")
        print(f"  - Max epochs: {self.trainer_cfg.get('max_epochs')}")
        if self.model_info.task == 'diffusion':
            print(f"  - Timesteps: {self.model_cfg.get('timesteps')}")
            print(f"  - Soft label: {self.model_cfg.get('soft_label_type')}")
            print(f"  - Use EMA: {self.model_cfg.get('use_ema')}")
        print(f"{'='*60}\n")
    
    def _create_datamodule(self):
        """Create datamodule."""
        DataModuleClass = self.dataset_info.class_ref
        
        return DataModuleClass(
            train_dir=self.data_cfg.get('train_dir', self.dataset_info.default_train_dir),
            val_dir=self.data_cfg.get('val_dir', self.dataset_info.default_val_dir),
            test_dir=self.data_cfg.get('test_dir', self.dataset_info.default_test_dir),
            crop_size=self.data_cfg.get('crop_size', self.dataset_info.default_crop_size),
            train_bs=self.data_cfg.get('train_bs', 8),
        )
    
    def _create_model(self):
        """Create model."""
        common_args = {
            'arch_name': self.model_name,
            'learning_rate': self.model_cfg.get('learning_rate', self.model_info.default_lr),
            'weight_decay': self.model_cfg.get('weight_decay', 1e-5),
            'experiment_name': f"{self.dataset_name}/{self.model_name}",
            'data_name': self.dataset_name,
            'image_size': self.model_cfg.get('image_size', 224),
            'num_classes': self.model_cfg.get('num_classes', 2),
        }
        
        if self.model_info.task == 'supervised':
            # SupervisedModel uses img_size instead of image_size
            supervised_args = common_args.copy()
            supervised_args['img_size'] = supervised_args.pop('image_size')
            supervised_args['in_channels'] = 1
            return SupervisedModel(**supervised_args)
        else:  # diffusion
            return DiffusionModel(
                **common_args,
                dim=self.model_cfg.get('dim', 64),
                timesteps=self.model_cfg.get('timesteps', 1000),
                soft_label_type=self.model_cfg.get('soft_label_type', 'none'),
                soft_label_cache=self.model_cfg.get('soft_label_cache', True),
                soft_label_fg_max=self.model_cfg.get('soft_label_fg_max', 11),
                soft_label_thickness_max=self.model_cfg.get('soft_label_thickness_max', 13),
                soft_label_kernel_ratio=self.model_cfg.get('soft_label_kernel_ratio', 0.1),
                use_ema=self.model_cfg.get('use_ema', True),
                ema_decay=self.model_cfg.get('ema_decay', 0.9999),
                num_ensemble=self.model_cfg.get('num_ensemble', 1),
            )
    
    def _create_callbacks(self, exp_dir: Path):
        """Create callbacks."""
        callbacks = [
            ModelCheckpoint(
                dirpath=exp_dir / "checkpoints",
                filename='best',
                monitor='val/dice',
                mode='max',
                save_last=True,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval='epoch'),
        ]
        
        if not self.debug:
            callbacks.append(
                StochasticWeightAveraging(
                    swa_lrs=self.model_cfg.get('learning_rate', 1e-4) / 10
                )
            )
        
        return callbacks
    
    def _create_logger(self, exp_dir: Path):
        """Create logger."""
        return EnhancedTensorBoardLogger(
            save_dir=exp_dir,
            name="tensorboard",
            version="",
        )
    
    def _create_trainer(self, callbacks, logger):
        """Create trainer."""
        # Extract precision (handle both "32" and "32-true" formats)
        precision_str = str(self.trainer_cfg.get('precision', '32-true'))
        
        # Get device list
        devices = self.trainer_cfg.get('devices', 1)
        if isinstance(devices, int):
            devices = [devices]
        
        return L.Trainer(
            max_epochs=self.trainer_cfg.get('max_epochs', 300),
            accelerator=self.trainer_cfg.get('accelerator', 'gpu'),
            devices=devices,
            precision=precision_str,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=self.trainer_cfg.get('log_every_n_steps', 50),
            check_val_every_n_epoch=self.trainer_cfg.get('check_val_every_n_epoch', 5),
            gradient_clip_val=self.trainer_cfg.get('gradient_clip_val', None),
            accumulate_grad_batches=self.trainer_cfg.get('accumulate_grad_batches', 1),
            limit_train_batches=self.trainer_cfg.get('limit_train_batches', 1.0),
            limit_val_batches=self.trainer_cfg.get('limit_val_batches', 1.0),
        )
