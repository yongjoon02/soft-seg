"""Check whether hard flow loss dominates soft flow loss for soft2hard model."""
from __future__ import annotations
import autorootcwd
import argparse
import inspect
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

import src.data.xca  # noqa: F401 - register datasets
from src.archs.flow_soft2hard_model import FlowSoft2HardModel
from src.registry.datasets import get_dataset_info


def _build_datamodule(data_cfg: dict):
    info = get_dataset_info(data_cfg.get("name", "xca"))
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
    if "train_bs" in params:
        kwargs["train_bs"] = 1
    if "num_samples_per_image" in params:
        kwargs["num_samples_per_image"] = 1
    if "label_subdir" in params and data_cfg.get("label_subdir") is not None:
        kwargs["label_subdir"] = data_cfg["label_subdir"]
    if "use_sauna_transform" in params:
        kwargs["use_sauna_transform"] = bool(data_cfg.get("use_sauna_transform", False))

    return dm_cls(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check hard vs soft flow loss dominance.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to soft2hard checkpoint.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment config.yaml.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda/cpu).",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of validation batches to average.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    dm = _build_datamodule(cfg.get("data", {}))
    dm.setup("fit")

    if dm.val_dataset is None:
        raise ValueError("Validation dataset is not available.")
    loader = DataLoader(dm.val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = FlowSoft2HardModel.load_from_checkpoint(str(checkpoint_path))
    model.eval()
    model.to(device)

    hard_losses = []
    soft_losses = []

    for idx, batch in enumerate(loader):
        if idx >= args.num_batches:
            break
        images = batch["image"].to(device)
        geometry = batch.get("geometry", batch.get("label")).to(device)
        labels = batch.get("label", geometry).to(device)

        x1 = model._build_x1(labels, geometry)
        x0 = torch.randn_like(x1)

        t, xt, ut = model.flow_matcher.sample_location_and_conditional_flow(x0, x1)
        v = model.unet(xt, t, images)

        v_hard = v[:, 0:1]
        v_soft = v[:, 1:2]
        ut_hard = ut[:, 0:1]
        ut_soft = ut[:, 1:2]

        hard_loss = torch.nn.functional.mse_loss(v_hard, ut_hard).item()
        soft_loss = torch.nn.functional.mse_loss(v_soft, ut_soft).item()
        hard_losses.append(hard_loss)
        soft_losses.append(soft_loss)

    hard_mean = sum(hard_losses) / max(1, len(hard_losses))
    soft_mean = sum(soft_losses) / max(1, len(soft_losses))
    ratio = soft_mean / (hard_mean + 1e-8)

    print(f"hard_flow_loss={hard_mean:.6f}")
    print(f"soft_flow_loss={soft_mean:.6f}")
    print(f"soft/hard ratio={ratio:.2f}")


if __name__ == "__main__":
    main()
