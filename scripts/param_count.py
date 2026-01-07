#!/usr/bin/env python3
"""Parameter counting utility for diffusion models."""
import argparse
import inspect
import os
import sys
from typing import Dict, Tuple


def _add_repo_to_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count parameters for diffusion models from configs or CLI."
    )
    parser.add_argument(
        "--config",
        nargs="*",
        help="YAML config path(s). If multiple, counts each config.",
    )
    parser.add_argument("--arch_name", help="Override model arch_name")
    parser.add_argument("--image_size", type=int, help="Override image_size")
    parser.add_argument("--dim", type=int, help="Override base dim")
    parser.add_argument("--timesteps", type=int, help="Override timesteps")
    parser.add_argument("--loss_type", help="Override loss_type (if supported)")
    return parser.parse_args()


def _load_config(path: str) -> Dict:
    from src.utils.config import load_config
    return load_config(path)


def _build_model(arch_name: str, model_cfg: Dict):
    from src.registry import MODEL_REGISTRY

    create_fn = MODEL_REGISTRY.get(arch_name)
    sig = inspect.signature(create_fn)
    kwargs = {}
    for name in sig.parameters.keys():
        if name in model_cfg:
            kwargs[name] = model_cfg[name]
    return create_fn(**kwargs)


def _count_params(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _format_params(value: int) -> str:
    return f"{value:,}"


def _format_millions(value: int) -> str:
    return f"{value / 1e6:.2f}M"


def _apply_overrides(model_cfg: Dict, args: argparse.Namespace) -> Dict:
    cfg = dict(model_cfg)
    if args.arch_name is not None:
        cfg["arch_name"] = args.arch_name
    if args.image_size is not None:
        cfg["image_size"] = args.image_size
    if args.dim is not None:
        cfg["dim"] = args.dim
    if args.timesteps is not None:
        cfg["timesteps"] = args.timesteps
    if args.loss_type is not None:
        cfg["loss_type"] = args.loss_type
    return cfg


def _report(title: str, total: int, trainable: int) -> None:
    print(f"{title}")
    print(f"  total:     {_format_params(total)} ({_format_millions(total)})")
    print(f"  trainable: {_format_params(trainable)} ({_format_millions(trainable)})")


def main() -> int:
    args = _parse_args()
    _add_repo_to_path()

    if not args.config and not args.arch_name:
        print("Provide --config or --arch_name.")
        return 2

    if args.config:
        for config_path in args.config:
            config = _load_config(config_path)
            model_cfg = config.get("model", {})
            model_cfg = _apply_overrides(model_cfg, args)
            arch_name = model_cfg.get("arch_name") or args.arch_name
            if arch_name is None:
                raise ValueError(f"arch_name not found in {config_path}")
            model = _build_model(arch_name, model_cfg)
            total, trainable = _count_params(model)
            _report(f"{config_path} ({arch_name})", total, trainable)
        return 0

    # CLI-only path
    model_cfg = _apply_overrides({}, args)
    arch_name = model_cfg.get("arch_name")
    model = _build_model(arch_name, model_cfg)
    total, trainable = _count_params(model)
    _report(f"{arch_name}", total, trainable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
