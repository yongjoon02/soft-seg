"""Base training script with shared logic for supervised and diffusion models."""

import os
import sys
import yaml
import tempfile
from src.utils.registry import DATASET_REGISTRY


def parse_config_and_setup_args(default_config: str):
    """
    Parse config file and setup command line arguments.
    
    This function handles:
    1. Loading config file (with default fallback)
    2. Extracting dataset name from config
    3. Removing 'data.name' field (not needed by LightningCLI)
    4. Converting --arch_name to LightningCLI format
    5. Setting up TensorBoard logger paths
    
    Args:
        default_config: Default config file path if not provided in args
        
    Returns:
        tuple: (data_name, DataModuleClass)
            - data_name: Dataset name (e.g., 'octa500_3m')
            - DataModuleClass: DataModule class from registry
    """
    # Add default config if not provided
    if '--config' not in sys.argv:
        sys.argv.extend(['--config', default_config])
    
    # Extract config path from arguments
    config_path = None
    if '--config' in sys.argv:
        config_idx = sys.argv.index('--config')
        if config_idx + 1 < len(sys.argv):
            config_path = sys.argv[config_idx + 1]
    
    # Parse config file to get dataset name
    data_name = None
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                data_name = config.get('data', {}).get('name')
                
                # Remove 'name' from data config before passing to LightningCLI
                # LightningCLI doesn't expect this field, so we handle it separately
                if 'data' in config and 'name' in config['data']:
                    del config['data']['name']
                    # Write modified config to a temp file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
                        yaml.dump(config, tmp)
                        temp_config_path = tmp.name
                    # Replace config path in sys.argv
                    sys.argv[sys.argv.index(config_path)] = temp_config_path
        except Exception as e:
            print(f"Warning: Could not parse config file {config_path}: {e}")
    
    if data_name is None:
        print("Error: data.name not found in config file")
        sys.exit(1)
    
    # Get appropriate DataModule from registry
    DataModuleClass = DATASET_REGISTRY.get(data_name)
    
    # Convert --arch_name to LightningCLI overrides
    # This allows using --arch_name csnet instead of --model.arch_name csnet
    if '--arch_name' in sys.argv:
        arch_idx = sys.argv.index('--arch_name')
        if arch_idx + 1 < len(sys.argv):
            arch_name = sys.argv[arch_idx + 1]
            # Remove --arch_name and its value
            sys.argv.pop(arch_idx)
            sys.argv.pop(arch_idx)
            # Add LightningCLI overrides
            sys.argv.extend(['--model.arch_name', arch_name])
            # Set TensorBoard logger name and version
            # This creates directory structure: lightning_logs/{data_name}/{arch_name}/
            sys.argv.extend(['--trainer.logger.init_args.name', data_name])
            sys.argv.extend(['--trainer.logger.init_args.version', arch_name])
    
    return data_name, DataModuleClass
