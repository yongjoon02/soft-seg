"""Supervised learning model for OCT segmentation."""

import lightning.pytorch as L
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from torchmetrics import MetricCollection

from src.archs.components import CSNet, DSCNet
from src.losses import SoftBCELoss, SoftCrossEntropyLoss
from src.metrics import (
    Betti0Error,
    Betti1Error,
    Dice,
    JaccardIndex,
    Precision,
    Recall,
    Specificity,
    clDice,
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
        log_image_enabled: bool = False,
        log_image_names: list = None,
        soft_label: bool = False,  # soft label 학습 모드
        loss_type: str = 'ce',  # 'ce' for CrossEntropy, 'bce' for BCE
    ):
        super().__init__()
        self.save_hyperparameters()

        # Set experiment name
        if experiment_name is None:
            experiment_name = f"{data_name}/{arch_name}"
        self.experiment_name = experiment_name
        self.data_name = data_name
        self.arch_name = arch_name
        self.soft_label = soft_label  # Store soft label mode

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

        # Loss function (soft label aware)
        if loss_type == 'bce':
            self.loss_fn = SoftBCELoss(soft_label=soft_label)
        else:  # default: 'ce'
            self.loss_fn = SoftCrossEntropyLoss(soft_label=soft_label)

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

        # Squeeze channel dimension if present
        if labels.dim() == 4:
            labels = labels.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
        
        # For hard label mode, convert to long; soft label mode keeps float
        if not self.soft_label:
            labels = labels.long()

        # Forward (ModelWrapper handles dict outputs)
        logits = self(images)

        # Compute loss (SoftCrossEntropyLoss handles both modes)
        loss = self.loss_fn(logits, labels)

        # Log
        self.log('train/loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']

        # Convert soft/binary labels to class indices (threshold for soft labels)
        if labels.dim() == 4:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()

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

        # Log images to TensorBoard (specified samples only)
        if hasattr(self.hparams, 'log_image_enabled') and self.hparams.log_image_enabled:
            log_names = getattr(self.hparams, 'log_image_names', None)
            pred_binary = (preds > 0).float()
            label_binary = (labels > 0).float()

            for i in range(images.shape[0]):
                sample_name = batch['name'][i] if 'name' in batch else f'sample_{i}'
                filename = sample_name.split('/')[-1] if '/' in sample_name else sample_name

                # Only log if filename matches log_image_names (or log all if not specified)
                if log_names is None or filename in log_names:
                    print(f"[DEBUG] Logging image: {filename} at epoch {self.current_epoch}")
                    # Log to TensorBoard
                    self.logger.experiment.add_image(
                        f'val/{filename}/prediction',
                        pred_binary[i:i+1],
                        self.global_step
                    )
                    self.logger.experiment.add_image(
                        f'val/{filename}/ground_truth',
                        label_binary[i:i+1],
                        self.global_step
                    )

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']

        # Get sample names if available
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])

        # Convert soft/binary labels to class indices (threshold for soft labels)
        if labels.dim() == 4:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()

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
