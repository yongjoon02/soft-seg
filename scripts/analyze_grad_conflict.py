#!/usr/bin/env python
"""Analyze gradient conflicts between loss components for a saved experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import autorootcwd  # noqa: F401
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

import src.losses  # noqa: F401 - register losses
from src.archs.flow_model import FlowModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze gradient conflicts for flow models.")
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Experiment directory containing config.yaml and checkpoints/",
    )
    parser.add_argument(
        "--checkpoint",
        default="last.ckpt",
        help="Checkpoint filename or absolute path (default: last.ckpt)",
    )
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to analyze.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for analysis.")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda or cpu).")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (default: 0 for compatibility).",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pin_memory for DataLoader (default: off).",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val"],
        help="Dataset split to use for analysis (default: train).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for TensorBoard logs (default: <experiment>/grad_analysis)",
    )
    parser.add_argument(
        "--components",
        default=None,
        help="Comma-separated loss components to analyze (default: auto-detect).",
    )
    return parser.parse_args()


def _load_config(exp_dir: Path) -> dict:
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def _make_datamodule(config: dict, batch_size: int):
    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("name")
    if dataset_name == "xca":
        from src.data.xca import XCADataModule
        return XCADataModule(
            train_dir=data_cfg.get("train_dir"),
            val_dir=data_cfg.get("val_dir"),
            test_dir=data_cfg.get("test_dir"),
            crop_size=data_cfg.get("image_size", 320),
            train_bs=batch_size,
            num_samples_per_image=data_cfg.get("num_samples_per_image", 1),
            label_subdir=data_cfg.get("label_subdir", "label"),
            use_sauna_transform=data_cfg.get("use_sauna_transform", False),
        )
    from src.registry.datasets import get_dataset_info
    info = get_dataset_info(dataset_name)
    return info.class_ref(
        train_dir=data_cfg.get("train_dir"),
        val_dir=data_cfg.get("val_dir"),
        test_dir=data_cfg.get("test_dir"),
        crop_size=data_cfg.get("image_size"),
        train_bs=batch_size,
    )


def _grad_vector(params):
    grads = []
    for param in params:
        if param.grad is None:
            continue
        grads.append(param.grad.detach().flatten())
    if not grads:
        return None
    return torch.cat(grads)


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.experiment_dir)
    config = _load_config(exp_dir)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = exp_dir / "checkpoints" / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = FlowModel.load_from_checkpoint(str(ckpt))
    model.to(device)
    model.train()

    datamodule = _make_datamodule(config, args.batch_size)
    datamodule.setup("fit")
    if args.split == "train":
        dataset = datamodule.train_dataset
        shuffle = True
    else:
        dataset = datamodule.val_dataset
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    output_dir = Path(args.output) if args.output else exp_dir / "grad_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir))

    unet_params = list(model.unet.parameters())

    step = 0
    for batch in loader:
        images = batch["image"].to(device)
        geometry = batch.get("geometry", batch.get("label")).to(device)
        labels = batch.get("label", geometry).to(device)

        noise = torch.randn_like(geometry)
        t, xt, ut = model.flow_matcher.sample_location_and_conditional_flow(noise, geometry)
        unet_out = model.unet(xt, t, images)
        if unet_out.shape[1] >= 2:
            v = unet_out[:, 0:1, :, :]
            geometry_pred = unet_out[:, 1:2, :, :]
        else:
            v = unet_out
            geometry_pred = None

        loss, loss_dict = model.loss_fn(
            v,
            ut,
            xt,
            geometry,
            t=t,
            geometry_pred=geometry_pred,
            hard_labels=labels,
            x0=noise,
        )

        if args.components:
            components = [c.strip() for c in args.components.split(",") if c.strip()]
        else:
            components = list(loss_dict.keys())

        grad_map = {}
        for name in components:
            if name not in loss_dict:
                continue
            model.zero_grad(set_to_none=True)
            loss_dict[name].backward(retain_graph=True)
            vec = _grad_vector(unet_params)
            if vec is None:
                continue
            grad_map[name] = vec
            writer.add_scalar(f"grad_norm/{name}", vec.norm().item(), step)

        if "flow" in grad_map and "bce" in grad_map:
            cos = torch.nn.functional.cosine_similarity(
                grad_map["flow"], grad_map["bce"], dim=0
            ).item()
            writer.add_scalar("grad_cos/flow_vs_bce", cos, step)
        if "flow" in grad_map and "dice" in grad_map:
            cos = torch.nn.functional.cosine_similarity(
                grad_map["flow"], grad_map["dice"], dim=0
            ).item()
            writer.add_scalar("grad_cos/flow_vs_dice", cos, step)
        if "flow" in grad_map and "l1geo_head" in grad_map:
            cos = torch.nn.functional.cosine_similarity(
                grad_map["flow"], grad_map["l1geo_head"], dim=0
            ).item()
            writer.add_scalar("grad_cos/flow_vs_l1geo_head", cos, step)
        if "flow" in grad_map and "bce_hard" in grad_map:
            cos = torch.nn.functional.cosine_similarity(
                grad_map["flow"], grad_map["bce_hard"], dim=0
            ).item()
            writer.add_scalar("grad_cos/flow_vs_bce_hard", cos, step)

        writer.add_scalar("loss/total", loss.item(), step)
        step += 1
        if step >= args.num_batches:
            break

    writer.flush()
    writer.close()
    print(f"✅ Gradient analysis logs saved to: {output_dir}")


if __name__ == "__main__":
    main()
