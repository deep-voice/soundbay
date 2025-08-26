"""
Configuration migration utility
------------------------------
This script helps migrate from Hydra YAML configurations to the new dataclass-based system.
It can convert existing YAML files to the new format and validate configurations.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from soundbay.config import Config, ConfigManager


def convert_hydra_yaml_to_dataclass(yaml_path: Path) -> Dict[str, Any]:
    """
    Convert Hydra YAML configuration to dataclass-compatible format
    
    Args:
        yaml_path: Path to Hydra YAML file
        
    Returns:
        Dictionary compatible with dataclass configuration
    """
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Remove Hydra-specific keys
    config_dict.pop('defaults', None)
    config_dict.pop('# @package _global_', None)
    
    # Convert nested structures to match dataclass expectations
    converted = {}
    
    # Handle data section
    if 'data' in config_dict:
        data = config_dict['data']
        converted['data'] = {
            'label_names': data.get('label_names', ['Noise', 'Call']),
            'batch_size': data.get('batch_size', 64),
            'num_workers': data.get('num_workers', 10),
            'sample_rate': data.get('sample_rate', 16000),
            'data_sample_rate': data.get('data_sample_rate', 44100),
            'min_freq': data.get('min_freq', 0),
            'n_fft': data.get('n_fft', 1024),
            'hop_length': data.get('hop_length', 256),
            'label_type': data.get('label_type', 'single_label'),
            'proba_threshold': data.get('proba_threshold', 0.5),
            'train_dataset': data.get('train_dataset', {}),
            'val_dataset': data.get('val_dataset', {})
        }
    
    # Handle experiment section
    if 'experiment' in config_dict:
        exp = config_dict['experiment']
        converted['experiment'] = {
            'debug': exp.get('debug', True),
            'manual_seed': exp.get('manual_seed', 1234),
            'name': exp.get('name'),
            'project': exp.get('project', 'finding_willy'),
            'run_id': exp.get('run_id'),
            'group_name': exp.get('group_name'),
            'bucket_name': exp.get('bucket_name', 'deepvoice-experiments'),
            'artifacts_upload_limit': exp.get('artifacts_upload_limit', 64),
            'equalize_data': exp.get('equalize_data', True),
            'checkpoint': exp.get('checkpoint', {})
        }
    
    # Handle model section
    if 'model' in config_dict:
        model = config_dict['model']
        converted['model'] = {
            'criterion': model.get('criterion', {'_target_': 'torch.nn.CrossEntropyLoss'}),
            'model': model.get('model', {
                '_target_': 'models.ResNet1Channel',
                'layers': [3, 4, 6, 3],
                'block': 'torchvision.models.resnet.Bottleneck',
                'num_classes': 2
            })
        }
    
    # Handle optim section
    if 'optim' in config_dict:
        optim = config_dict['optim']
        converted['optim'] = {
            'epochs': optim.get('epochs', 100),
            'optimizer': optim.get('optimizer', {
                '_target_': 'torch.optim.Adam',
                'lr': 5e-4
            }),
            'scheduler': optim.get('scheduler', {
                '_target_': 'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts',
                'T_0': 5
            }),
            'freeze_layers_for_finetune': optim.get('freeze_layers_for_finetune', True)
        }
    
    return converted


def validate_config(config: Config) -> bool:
    """
    Validate a configuration object
    
    Args:
        config: Configuration object to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Basic validation checks
        assert config.data.batch_size > 0, "Batch size must be positive"
        assert config.data.num_workers >= 0, "Number of workers must be non-negative"
        assert config.data.sample_rate > 0, "Sample rate must be positive"
        assert config.optim.epochs > 0, "Number of epochs must be positive"
        assert config.optim.optimizer.lr > 0, "Learning rate must be positive"
        
        # Check that required paths exist if not in debug mode
        if not config.experiment.debug:
            if config.data.train_dataset.data_path:
                assert Path(config.data.train_dataset.data_path).exists(), \
                    f"Train data path does not exist: {config.data.train_dataset.data_path}"
            if config.data.val_dataset.data_path:
                assert Path(config.data.val_dataset.data_path).exists(), \
                    f"Validation data path does not exist: {config.data.val_dataset.data_path}"
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


def migrate_config_file(input_path: Path, output_path: Optional[Path] = None) -> bool:
    """
    Migrate a Hydra configuration file to the new dataclass format
    
    Args:
        input_path: Path to input Hydra YAML file
        output_path: Path to output YAML file (optional)
        
    Returns:
        True if migration successful, False otherwise
    """
    try:
        # Convert the configuration
        converted_dict = convert_hydra_yaml_to_dataclass(input_path)
        
        # Create config object to validate
        config = Config(**converted_dict)
        
        # Validate the configuration
        if not validate_config(config):
            return False
        
        # Save the converted configuration
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_converted.yaml"
        
        config_manager = ConfigManager(config)
        config_manager.save(output_path)
        
        print(f"Successfully migrated {input_path} to {output_path}")
        return True
        
    except Exception as e:
        print(f"Migration failed for {input_path}: {e}")
        return False


def batch_migrate_configs(config_dir: Path, output_dir: Optional[Path] = None) -> None:
    """
    Migrate all YAML configuration files in a directory
    
    Args:
        config_dir: Directory containing configuration files
        output_dir: Output directory (optional)
    """
    if output_dir is None:
        output_dir = config_dir / "converted"
    
    output_dir.mkdir(exist_ok=True)
    
    yaml_files = list(config_dir.glob("*.yaml"))
    print(f"Found {len(yaml_files)} YAML files to migrate")
    
    successful = 0
    for yaml_file in yaml_files:
        output_path = output_dir / yaml_file.name
        if migrate_config_file(yaml_file, output_path):
            successful += 1
    
    print(f"Successfully migrated {successful}/{len(yaml_files)} configuration files")


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate Hydra configurations to dataclass format')
    parser.add_argument('input', type=str, help='Input YAML file or directory')
    parser.add_argument('--output', type=str, help='Output file or directory')
    parser.add_argument('--batch', action='store_true', help='Process all YAML files in directory')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}")
        return
    
    if args.batch or input_path.is_dir():
        batch_migrate_configs(input_path, output_path)
    else:
        migrate_config_file(input_path, output_path)


if __name__ == "__main__":
    main()
