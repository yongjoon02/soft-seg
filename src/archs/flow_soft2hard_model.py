"""Flow matching model with soft-to-hard coupling (dual-channel state)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import lightning.pytorch as L
from monai.inferers import SlidingWindowInferer
from torchmetrics import MetricCollection
from torchdiffeq import odeint

from src.registry.base import ARCHS_REGISTRY
from src.archs.components import unet  # noqa: F401 - register architectures
from src.archs.components.flow import SchrodingerBridgeConditionalFlowMatcher
from src.metrics.general_metrics import Dice, Precision, Recall, Specificity, JaccardIndex
from src.metrics.vessel_metrics import clDice, Betti0Error, Betti1Error
from src.archs.components.utils import random_patch_batch, select_patch_params


class FlowSoft2HardModel(L.LightningModule):
    """Flow matching with 2-channel outputs: hard + soft, soft guides hard."""

    def __init__(
        self,
        arch_name: str = 'medsegdiff_flow_soft2hard',
        image_size: int = 512,
        patch_plan: list = [(320, 6), (384, 4), (416, 3), (512, 1)],
        dim: int = 32,
        timesteps: int = 15,
        sigma: float = 0.25,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-5,
        num_classes: int = 2,
        experiment_name: str | None = None,
        num_ensemble: int = 1,
        data_name: str = 'xca',
        log_image_enabled: bool = False,
        log_image_names: list | None = None,
        # UNet architecture parameters
        model_channels: int = 32,
        channel_mult: list = [1, 2, 4, 8],
        channel_mult_emb: int = 4,
        num_blocks: int = 3,
        attn_resolutions: list = [16, 16, 8, 8],
        dropout: float = 0.0,
        label_dim: int = 0,
        augment_dim: int = 0,
        # Loss configuration
        flow_weight: float = 1.0,  # unused (kept for config compatibility)
        hard_weight: float = 1.0,
        soft_weight: float = 0.2,
        soft2hard_weight: float = 0.1,  # unused (kept for config compatibility)
        soft_sharpness: float = 10.0,  # unused (kept for config compatibility)
    ):
        super().__init__()
        self.save_hyperparameters()

        if experiment_name is None:
            experiment_name = f"{data_name}/{arch_name}"
        self.experiment_name = experiment_name
        self.data_name = data_name

        if arch_name not in ARCHS_REGISTRY:
            raise ValueError(f"Unknown architecture: {arch_name}. Available: {list(ARCHS_REGISTRY.keys())}")
        arch_class = ARCHS_REGISTRY.get(arch_name)

        self.unet = arch_class(
            img_resolution=image_size,
            model_channels=model_channels,
            channel_mult=channel_mult,
            channel_mult_emb=channel_mult_emb,
            num_blocks=num_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            label_dim=label_dim,
            augment_dim=augment_dim,
        )

        self.flow_matcher = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)

        # Metrics (foreground-only: DiceScore uses include_background=False, others use ignore_index=0)
        self.val_metrics = MetricCollection({
            'dice': Dice(num_classes=num_classes, include_background=False),
            'precision': Precision(num_classes=num_classes, average='macro', ignore_index=0),
            'recall': Recall(num_classes=num_classes, average='macro', ignore_index=0),
            'iou': JaccardIndex(num_classes=num_classes, average='macro', ignore_index=0),
        })

        # Test metrics (separate instance to avoid contamination)
        self.test_metrics = MetricCollection({
            'dice': Dice(num_classes=num_classes, include_background=False),
            'precision': Precision(num_classes=num_classes, average='macro', ignore_index=0),
            'recall': Recall(num_classes=num_classes, average='macro', ignore_index=0),
            'iou': JaccardIndex(num_classes=num_classes, average='macro', ignore_index=0),
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

        self.log_image_enabled = log_image_enabled
        self.log_image_names = log_image_names if log_image_names is not None else ['00036.png']

        self._sample_images = None

        # Sliding window inferer for validation/test (fixed settings).
        self.inferer = SlidingWindowInferer(
            roi_size=(image_size, image_size),
            sw_batch_size=4,
            overlap=0.25,
            mode='gaussian',
        )

    def _ensure_4d(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            return tensor.unsqueeze(1)
        return tensor

    def _to_unit_range(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor

    def _build_x1(self, labels: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        labels = self._to_unit_range(self._ensure_4d(labels))
        geometry = self._to_unit_range(self._ensure_4d(geometry))
        return torch.cat([labels, geometry], dim=1)

    def _soft_to_hard(self, soft_pred: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.hparams.soft_sharpness * (soft_pred - 0.5))

    def training_step(self, batch, batch_idx):
        images = batch['image']
        geometry = batch.get('geometry', batch.get('label'))
        labels = batch.get('label', geometry)

        x1 = self._build_x1(labels, geometry)

        patch_size, num_patches = select_patch_params(self.hparams.patch_plan)

        noise = torch.randn_like(x1)
        noise, x1, images = random_patch_batch([noise, x1, images], patch_size, num_patches)

        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(noise, x1)

        v = self.unet(xt, t, images)
        v_hard = v[:, 0:1]
        v_soft = v[:, 1:2]
        ut_hard = ut[:, 0:1]
        ut_soft = ut[:, 1:2]

        hard_loss = F.l1_loss(v_hard, ut_hard)
        soft_loss = F.l1_loss(v_soft, ut_soft)

        total = (
            self.hparams.hard_weight * hard_loss
            + self.hparams.soft_weight * soft_loss
        )

        self.log('train/loss', total, prog_bar=True, sync_dist=True)
        self.log('train/hard_flow_loss', hard_loss, prog_bar=False, sync_dist=True)
        self.log('train/soft_flow_loss', soft_loss, prog_bar=False, sync_dist=True)

        return total

    def sample(self, noise, images, return_intermediate: bool = False, save_steps: list | None = None):
        if save_steps is None:
            save_steps = [2, 4, 6, 8, 10, 12, 14]

        self._sample_images = images

        traj = odeint(
            self.ode_func,
            noise,
            torch.linspace(0, 1, self.hparams.timesteps, device=noise.device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )

        output_geometry = traj[-1]

        if return_intermediate:
            if save_steps is not None:
                saved_steps = {t: traj[t] for t in save_steps}
                return saved_steps, output_geometry
            return traj, output_geometry
        return output_geometry

    def _sliding_window_sample(self, images: torch.Tensor) -> torch.Tensor:
        """Sliding-window predictor for val/test."""
        noise = torch.randn(
            images.shape[0],
            2,
            images.shape[2],
            images.shape[3],
            device=images.device,
            dtype=images.dtype,
        )
        return self.sample(noise, images)

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        geometry = batch.get('geometry', batch.get('label'))
        labels = batch.get('label', geometry)
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])

        if self.hparams.num_ensemble > 1:
            output_list = []
            for _ in range(self.hparams.num_ensemble):
                output_geometry = self.inferer(images, self._sliding_window_sample)
                output_list.append(output_geometry)
            output_geometry = torch.stack(output_list).mean(dim=0)
        else:
            output_geometry = self.inferer(images, self._sliding_window_sample)

        output_hard = output_geometry[:, 0:1]
        output_soft = output_geometry[:, 1:2]
        preds = (output_hard.squeeze(1) > 0.5).long()

        hard_labels = self._to_unit_range(self._ensure_4d(labels))
        hard_labels_2d = hard_labels.squeeze(1)

        general_metrics = self.val_metrics(preds, hard_labels_2d.long())
        vessel_metrics = self.vessel_metrics(preds, hard_labels_2d.long())

        self.log_dict({'val/' + k: v for k, v in general_metrics.items()}, sync_dist=True)
        self.log_dict({'val/' + k: v for k, v in vessel_metrics.items()}, sync_dist=True)
        
        # Explicitly log val/dice for monitoring (ensure it appears in log files)
        self.log('val/dice', general_metrics['dice'], prog_bar=True, sync_dist=True, logger=True)

        recon_loss = torch.abs(output_soft - self._to_unit_range(self._ensure_4d(geometry))).mean()
        self.log('val/reconstruction_loss', recon_loss, prog_bar=True, sync_dist=True)

        self._log_images(sample_names, images, hard_labels_2d, preds, output_soft)

        return general_metrics['dice']

    def test_step(self, batch, batch_idx):
        images = batch['image']
        geometry = batch.get('geometry', batch.get('label'))
        labels = batch.get('label', geometry)
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])

        if self.hparams.num_ensemble > 1:
            output_list = []
            for _ in range(self.hparams.num_ensemble):
                output_geometry = self.inferer(images, self._sliding_window_sample)
                output_list.append(output_geometry)
            output_geometry = torch.stack(output_list).mean(dim=0)
        else:
            output_geometry = self.inferer(images, self._sliding_window_sample)

        output_hard = output_geometry[:, 0:1]
        output_soft = output_geometry[:, 1:2]
        preds = (output_hard.squeeze(1) > 0.5).long()

        hard_labels = self._to_unit_range(self._ensure_4d(labels))
        hard_labels_2d = hard_labels.squeeze(1)

        # Compute metrics (use test_metrics for test step)
        general_metrics = self.test_metrics(preds, hard_labels_2d.long())
        vessel_metrics = self.test_vessel_metrics(preds, hard_labels_2d.long())

        self.log_dict({'test/' + k: v for k, v in general_metrics.items()}, sync_dist=True)
        self.log_dict({'test/' + k: v for k, v in vessel_metrics.items()}, sync_dist=True)

        self._log_images(sample_names, images, hard_labels_2d, preds, output_soft, tag_prefix='test')

        return general_metrics['dice']

    @torch.no_grad()
    def ode_func(self, t, x):
        if isinstance(t, torch.Tensor):
            t = t.expand(x.shape[0])
        else:
            t = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)

        images = self._sample_images
        return self.unet(x, t, images)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=3,
            factor=0.5,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/dice',
                'interval': 'epoch',
                'frequency': 25,
            }
        }

    def _log_images(
        self,
        sample_names,
        images,
        labels,
        preds,
        soft_preds,
        tag_prefix: str = 'val',
    ):
        if not self.log_image_enabled:
            return

        if self.log_image_names is not None:
            indices = [i for i, name in enumerate(sample_names) if name in self.log_image_names]
            if len(indices) == 0:
                return
        else:
            indices = list(range(min(3, images.shape[0])))

        for idx in indices:
            img = images[idx].float()
            label = labels[idx].float().unsqueeze(0)
            pred = preds[idx].float().unsqueeze(0)
            soft = soft_preds[idx].float()

            vis_row = torch.cat([img, label, pred], dim=-1)
            self.logger.experiment.add_image(
                tag=f'{tag_prefix}/{sample_names[idx]}',
                img_tensor=vis_row,
                global_step=self.global_step,
            )

            soft_row = torch.cat([img, soft], dim=-1)
            self.logger.experiment.add_image(
                tag=f'{tag_prefix}_soft/{sample_names[idx]}',
                img_tensor=soft_row,
                global_step=self.global_step,
            )
