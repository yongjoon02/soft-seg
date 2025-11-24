"""Supervised learning model for OCT segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as L
from monai.inferers import SlidingWindowInferer
from torchmetrics import MetricCollection

from src.archs.components import CSNet, DSCNet
from src.metrics import (
    Dice, Precision, Recall, Specificity, JaccardIndex,
    clDice, Betti0Error, Betti1Error
)


MODEL_REGISTRY = {
    'csnet': CSNet,
    'dscnet': DSCNet,
}


class ModelWrapper(nn.Module):
    """Wrapper to handle models that return dict outputs (e.g., UNet3Plus)."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output = self.model(x)
        # If model returns dict, extract main output
        if isinstance(output, dict):
            return output['main_out']
        return output


class SupervisedModel(L.LightningModule):
    """Supervised segmentation model with sliding window inference."""
    
    def __init__(
        self,
        arch_name: str = 'cenet',
        in_channels: int = 1,
        num_classes: int = 2,
        img_size: int = 224,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-5,
        experiment_name: str = None,
        data_name: str = 'octa500_3m',
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Set experiment name
        if experiment_name is None:
            experiment_name = f"{data_name}/{arch_name}"
        self.experiment_name = experiment_name
        self.data_name = data_name
        self.arch_name = arch_name
        
        # 모델 생성
        if arch_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown architecture: {arch_name}. Choose from {list(MODEL_REGISTRY.keys())}")
        
        model_cls = MODEL_REGISTRY[arch_name]
        
        # Create model instance
        base_model = model_cls(in_channels=in_channels, num_classes=num_classes)
        
        # Wrap model to handle dict outputs
        self.model = ModelWrapper(base_model)
        
        # Sliding window inferer for validation (128x128 patches)
        self.inferer = SlidingWindowInferer(
            roi_size=(224, 224),
            sw_batch_size=4,
            overlap=0.25,
            mode='gaussian',
        )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Metrics
        self.val_metrics = MetricCollection({
            'dice': Dice(num_classes=num_classes, average='macro'),
            'precision': Precision(num_classes=num_classes, average='macro'),
            'recall': Recall(num_classes=num_classes, average='macro'),
            'specificity': Specificity(num_classes=num_classes, average='macro'),
            'iou': JaccardIndex(num_classes=num_classes, average='macro'),
        })
        
        # Test metrics (separate instance to avoid contamination)
        self.test_metrics = MetricCollection({
            'dice': Dice(num_classes=num_classes, average='macro'),
            'precision': Precision(num_classes=num_classes, average='macro'),
            'recall': Recall(num_classes=num_classes, average='macro'),
            'specificity': Specificity(num_classes=num_classes, average='macro'),
            'iou': JaccardIndex(num_classes=num_classes, average='macro'),
        })
        
        self.vessel_metrics = MetricCollection({
            'cldice': clDice(),
            'betti_0_error': Betti0Error(),
            'betti_1_error': Betti1Error(),
        })
        
        # Test vessel metrics (separate instance)
        self.test_vessel_metrics = MetricCollection({
            'cldice': clDice(),
            'betti_0_error': Betti0Error(),
            'betti_1_error': Betti1Error(),
        })
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        
        # Convert binary labels to class indices (squeeze channel if present)
        if labels.dim() == 4:
            labels = labels.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
        labels = labels.long()
        
        # Forward (ModelWrapper handles dict outputs)
        logits = self(images)
        
        # Compute loss
        loss = self.loss_fn(logits, labels)
        
        # Log
        self.log('train/loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        
        # Convert binary labels to class indices
        if labels.dim() == 4:
            labels = labels.squeeze(1)
        labels = labels.long()
        
        # Sliding window inference (ModelWrapper handles dict outputs)
        logits = self.inferer(images, self.model)
        
        # Compute loss
        loss = self.loss_fn(logits, labels)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        
        # General metrics
        general_metrics = self.val_metrics(preds, labels)
        
        # Vessel-specific metrics
        vessel_metrics = self.vessel_metrics(preds, labels)
        
        # Log
        self.log('val/loss', loss, prog_bar=True)
        self.log_dict({'val/' + k: v for k, v in general_metrics.items()}, prog_bar=True)
        self.log_dict({'val/' + k: v for k, v in vessel_metrics.items()}, prog_bar=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        
        # Get sample names if available
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Convert binary labels to class indices
        if labels.dim() == 4:
            labels = labels.squeeze(1)
        labels = labels.long()
        
        # Sliding window inference (ModelWrapper handles dict outputs)
        logits = self.inferer(images, self.model)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        
        # General metrics (use test_metrics, not val_metrics!)
        general_metrics = self.test_metrics(preds, labels)
        
        # Vessel-specific metrics (use test_vessel_metrics!)
        vessel_metrics = self.test_vessel_metrics(preds, labels)

        # Log
        self.log_dict({'test/' + k: v for k, v in general_metrics.items()})
        self.log_dict({'test/' + k: v for k, v in vessel_metrics.items()})
        
        
        # Store predictions for logging
        if hasattr(self.trainer, 'logger') and hasattr(self.trainer.logger, 'save_predictions'):
            # Convert to binary masks for visualization
            pred_masks = (preds > 0).float()
            label_masks = (labels > 0).float()
            
            # Prepare metrics for each sample
            sample_metrics = []
            for i in range(images.shape[0]):
                sample_metric = {}
                sample_metric.update({k: v.item() if torch.is_tensor(v) else v for k, v in general_metrics.items()})
                sample_metric.update({k: v.item() if torch.is_tensor(v) else v for k, v in vessel_metrics.items()})
                sample_metrics.append(sample_metric)
            
            # Save predictions
            self.trainer.logger.save_predictions(
                sample_names, images, pred_masks, label_masks, sample_metrics
            )
        
        return general_metrics['dice']
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=20,
            factor=0.5,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train/loss',
                'interval': 'epoch',
            }
        }
