"""Visualize flow x1_pred at fixed t values for a validation sample."""
from __future__ import annotations
import autorootcwd
import argparse
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

import src.data.xca  # noqa: F401 - register datasets
from src.archs.flow_model import FlowModel
from src.registry.datasets import get_dataset_info


def _build_datamodule(data_cfg: dict):
    data_name = data_cfg.get("name", "xca")
    info = get_dataset_info(data_name)
    dm_cls = info.class_ref
    sig = inspect.signature(dm_cls.__init__)
    params = sig.parameters
    kwargs = {}

    if "train_dir" in params and data_cfg.get("train_dir") is not None:
        kwargs["train_dir"] = data_cfg["train_dir"]
    if "val_dir" in params and data_cfg.get("val_dir") is not None:
        kwargs["val_dir"] = data_cfg["val_dir"]
    if "test_dir" in params and data_cfg.get("test_dir") is not None:
        kwargs["test_dir"] = data_cfg["test_dir"]
    if "crop_size" in params and data_cfg.get("image_size") is not None:
        kwargs["crop_size"] = int(data_cfg["image_size"])
    if "train_bs" in params and data_cfg.get("train_bs") is not None:
        kwargs["train_bs"] = int(data_cfg["train_bs"])
    if "num_samples_per_image" in params and data_cfg.get("num_samples_per_image") is not None:
        kwargs["num_samples_per_image"] = int(data_cfg["num_samples_per_image"])
    if "label_subdir" in params and data_cfg.get("label_subdir") is not None:
        kwargs["label_subdir"] = data_cfg["label_subdir"]
    if "use_sauna_transform" in params and data_cfg.get("use_sauna_transform") is not None:
        kwargs["use_sauna_transform"] = bool(data_cfg["use_sauna_transform"])

    return dm_cls(**kwargs)


def _parse_t_values(text: str) -> list[float]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize x1_pred at fixed t values for a validation sample."
    )
    parser.add_argument(
        "--checkpoint",
        default="experiments/medsegdiff_flow/xca/medsegdiff_flow_xca_20260102_111643/checkpoints/best.ckpt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--config",
        default="experiments/medsegdiff_flow/xca/medsegdiff_flow_xca_20260102_111643/config.yaml",
        help="Path to experiment config.yaml.",
    )
    parser.add_argument(
        "--output",
        default="results/visualizations/medsegdiff_flow_xca_20260102_111643_t_grid.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (e.g., cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Validation sample index to visualize.",
    )
    parser.add_argument(
        "--t-values",
        default="0.1,0.3,0.5,0.7,0.9",
        help="Comma-separated t values to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for x0/eps.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers for visualization (default: 0).",
    )
    parser.add_argument(
        "--quiver-output",
        default=None,
        help="Optional output path for quiver plot of model_out_v (gradient proxy).",
    )
    parser.add_argument(
        "--quiver-t",
        type=float,
        default=0.5,
        help="t value to use for quiver visualization.",
    )
    parser.add_argument(
        "--quiver-step",
        type=int,
        default=8,
        help="Stride for quiver downsampling.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    dm = _build_datamodule(data_cfg)
    dm.setup("fit")

    if dm.val_dataset is None:
        raise ValueError("Validation dataset is not available.")
    val_loader = DataLoader(
        dm.val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=bool(args.num_workers > 0),
    )
    batch = None
    for idx, val_batch in enumerate(val_loader):
        if idx == args.sample_index:
            batch = val_batch
            break
    if batch is None:
        raise ValueError(f"Validation sample index {args.sample_index} not found.")

    model = FlowModel.load_from_checkpoint(str(checkpoint_path))
    model.eval()
    model.to(device)

    images = batch["image"].to(device)
    geometry = batch.get("geometry", batch.get("label")).to(device)
    labels = batch.get("label", geometry).to(device)
    if geometry.dim() == 3:
        geometry = geometry.unsqueeze(1)
    if labels.dim() == 3:
        labels = labels.unsqueeze(1)

    torch.manual_seed(args.seed)
    x0 = torch.randn_like(geometry)
    eps = torch.randn_like(geometry)

    t_values = _parse_t_values(args.t_values)
    preds = []
    with torch.no_grad():
        for t in t_values:
            t_tensor = torch.full((geometry.shape[0],), t, device=device, dtype=geometry.dtype)
            xt = model.flow_matcher.sample_xt(x0, geometry, t_tensor, eps)
            v = model.unet(xt, t_tensor, images)
            if hasattr(model, "_constrain_v"):
                v = model._constrain_v(v)
            x1_pred = xt + (1 - t) * v
            x1_pred = torch.clamp(x1_pred, 0.0, 1.0)
            preds.append(x1_pred[0, 0].detach().cpu())

    fig, axes = plt.subplots(1, len(t_values), figsize=(3 * len(t_values), 3))
    if len(t_values) == 1:
        axes = [axes]
    for ax, t, pred in zip(axes, t_values, preds):
        ax.imshow(pred, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"t={t:.1f}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved visualization to: {output_path}")
    print("Formula: xt + (1 - t) * v")

    if args.quiver_output:
        quiver_path = Path(args.quiver_output)
        quiver_path.parent.mkdir(parents=True, exist_ok=True)
        t_val = float(args.quiver_t)
        t_tensor = torch.full((geometry.shape[0],), t_val, device=device, dtype=geometry.dtype)
        xt = model.flow_matcher.sample_xt(x0, geometry, t_tensor, eps)
        v = model.unet(xt, t_tensor, images)
        if hasattr(model, "_constrain_v"):
            v = model._constrain_v(v)

        v_scalar = v[0, 0]
        grad_y, grad_x = torch.gradient(v_scalar)
        mag = torch.sqrt(grad_x ** 2 + grad_y ** 2).detach().cpu()
        mean_mag = mag.mean().item()

        label_mask = (labels[0, 0] > 0.5).detach().cpu()
        vessel_mag = mag[label_mask].mean().item() if label_mask.any() else 0.0
        background_mag = mag[~label_mask].mean().item() if (~label_mask).any() else 0.0
        print(
            "Quiver t={:.2f} | mean |v| all={:.6f}, vessel={:.6f}, background={:.6f}".format(
                t_val, mean_mag, vessel_mag, background_mag
            )
        )

        step = max(1, int(args.quiver_step))
        y_idx = torch.arange(0, mag.shape[0], step)
        x_idx = torch.arange(0, mag.shape[1], step)
        yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")
        u = grad_x[yy, xx].detach().cpu()
        w = grad_y[yy, xx].detach().cpu()

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(label_mask, cmap="gray", alpha=0.6)
        ax.quiver(
            xx.numpy(),
            yy.numpy(),
            u.numpy(),
            w.numpy(),
            color="red",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.002,
        )
        ax.set_title(f"model_out_v gradient quiver @ t={t_val:.2f}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(quiver_path, dpi=200)
        plt.close(fig)
        print(f"Saved quiver plot to: {quiver_path}")


if __name__ == "__main__":
    main()
