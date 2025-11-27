"""Configuration utilities for loading and merging YAML configs with CLI arguments."""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_config_with_args(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Merge YAML config with CLI arguments.
    
    CLI arguments override config values when provided.
    
    Args:
        config: Dictionary from YAML config
        args: argparse Namespace with CLI arguments
        
    Returns:
        Merged configuration dictionary
    """
    merged = config.copy()
    
    # Get sections
    model_cfg = merged.get('model', {})
    data_cfg = merged.get('data', {})
    trainer_cfg = merged.get('trainer', {})
    
    # Override with CLI args (if provided, i.e., not None)
    if args.model is not None:
        model_cfg['arch_name'] = args.model
    if args.data is not None:
        data_cfg['name'] = args.data
    
    # Training hyperparameters
    if args.epochs is not None:
        trainer_cfg['max_epochs'] = args.epochs
    if args.batch_size is not None:
        data_cfg['train_bs'] = args.batch_size
    if args.lr is not None:
        model_cfg['learning_rate'] = args.lr
    if args.crop_size is not None:
        data_cfg['crop_size'] = args.crop_size
        model_cfg['image_size'] = args.crop_size
    if args.gpu is not None:
        trainer_cfg['devices'] = [args.gpu]
    
    # Diffusion-specific
    if args.soft_label is not None:
        model_cfg['soft_label_type'] = args.soft_label
    if args.soft_label_fg_max is not None:
        model_cfg['soft_label_fg_max'] = args.soft_label_fg_max
    if args.soft_label_thickness_max is not None:
        model_cfg['soft_label_thickness_max'] = args.soft_label_thickness_max
    if args.timesteps is not None:
        model_cfg['timesteps'] = args.timesteps
    if args.no_ema:
        model_cfg['use_ema'] = False
    if args.ema_decay is not None:
        model_cfg['ema_decay'] = args.ema_decay
    if args.ensemble is not None:
        model_cfg['num_ensemble'] = args.ensemble
    
    # Experiment
    if args.tag is not None:
        merged['tag'] = args.tag
    if args.debug:
        merged['debug'] = True
    if args.resume is not None:
        merged['resume'] = args.resume
    
    # Update merged config
    merged['model'] = model_cfg
    merged['data'] = data_cfg
    merged['trainer'] = trainer_cfg
    
    return merged


def get_default_config_path(model_name: str, dataset_name: str) -> Optional[Path]:
    """Get default config file path based on model and dataset.
    
    New structure: configs/{model_type}/{dataset_name}/{model_name}.yaml
    
    Args:
        model_name: Model name (e.g., 'csnet', 'medsegdiff')
        dataset_name: Dataset name (e.g., 'octa500_3m')
        
    Returns:
        Path to default config file, or None if not found
    """
    from src.registry import MODEL_REGISTRY
    
    # Determine if supervised or diffusion
    model_info = MODEL_REGISTRY.get(model_name)
    if model_info is None:
        return None
    
    model_type = 'diffusion' if model_info.task == 'diffusion' else 'supervised'
    
    # New structure: configs/{model_type}/{dataset_name}/{model_name}.yaml
    config_dir = Path('configs')
    config_file = config_dir / model_type / dataset_name / f'{model_name}.yaml'
    
    if config_file.exists():
        return config_file
    
    # Fallback to old structure for backward compatibility
    old_config_file = config_dir / f'{dataset_name}_{model_type}_models.yaml'
    return old_config_file if old_config_file.exists() else None


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
