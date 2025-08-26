"""
Configuration system using dataclasses to replace Hydra
-----------------------------------------------------
This module provides a dataclass-based configuration system that supports:
- Hierarchical configuration with nested dataclasses
- Override priority: cmdline > config > checkpoint > defaults
- Type validation and conversion
- Nested value overrides using dot notation
"""

import os
import json
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field, asdict
import argparse
from copy import deepcopy


@dataclass
class DatasetConfig:
    """Configuration for dataset settings"""
    _target_: str = "soundbay.data.ClassifierDataset"
    data_path: str = "./tests/assets/data/"
    path_hierarchy: int = 0
    mode: str = "train"
    metadata_path: str = "./tests/assets/annotations/sample_annotations.csv"
    augmentations_p: float = 0.8
    augmentations: Optional[List[str]] = None
    preprocessors: Optional[List[str]] = None
    seq_length: int = 1
    margin_ratio: float = 0.5
    data_sample_rate: int = 44100
    sample_rate: int = 16000
    slice_flag: bool = False


@dataclass
class DataConfig:
    """Configuration for data processing"""
    label_names: List[str] = field(default_factory=lambda: ['Noise', 'Call'])
    batch_size: int = 64
    num_workers: int = 10
    sample_rate: int = 16000
    data_sample_rate: int = 44100
    min_freq: int = 0
    n_fft: int = 1024
    hop_length: int = 256
    label_type: str = 'single_label'
    proba_threshold: float = 0.5
    train_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    val_dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(
        mode="val",
        augmentations_p=0.0,
        augmentations=None,
        margin_ratio=0.0,
        slice_flag=True
    ))


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint handling"""
    path: Optional[str] = None
    resume: str = 'allow'
    load_optimizer_state: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for experiment settings"""
    debug: bool = True
    manual_seed: Optional[int] = 1234
    name: Optional[str] = None
    project: str = 'finding_willy'
    run_id: Optional[str] = None
    group_name: Optional[str] = None
    bucket_name: str = 'deepvoice-experiments'
    artifacts_upload_limit: int = 64
    equalize_data: bool = True
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass
class ModelConfig:
    """Configuration for model settings"""
    criterion: Dict[str, str] = field(default_factory=lambda: {"_target_": "torch.nn.CrossEntropyLoss"})
    model: Dict[str, Any] = field(default_factory=lambda: {
        "_target_": "models.ResNet1Channel",
        "layers": [3, 4, 6, 3],
        "block": "torchvision.models.resnet.Bottleneck",
        "num_classes": 2
    })


@dataclass
class OptimizerConfig:
    """Configuration for optimizer settings"""
    _target_: str = "torch.optim.Adam"
    lr: float = 5e-4


@dataclass
class SchedulerConfig:
    """Configuration for scheduler settings"""
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"
    T_0: int = 5


@dataclass
class OptimConfig:
    """Configuration for optimization settings"""
    epochs: int = 100
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    freeze_layers_for_finetune: bool = True


@dataclass
class Config:
    """Main configuration class that combines all sub-configurations"""
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    
    def __post_init__(self):
        """Ensure nested dataclasses are properly initialized"""
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.experiment, dict):
            self.experiment = ExperimentConfig(**self.experiment)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.optim, dict):
            self.optim = OptimConfig(**self.optim)


class ConfigManager:
    """Manages configuration loading, merging, and overrides"""
    
    def __init__(self, base_config: Optional[Config] = None):
        self.config = base_config or Config()
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'ConfigManager':
        """Load configuration from YAML file"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Remove Hydra-specific keys
        config_dict.pop('defaults', None)
        config_dict.pop('# @package _global_', None)
        
        return cls(Config(**config_dict))
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: Union[str, Path]) -> 'ConfigManager':
        """Load configuration from checkpoint directory"""
        checkpoint_path = Path(checkpoint_path)
        args_path = checkpoint_path / 'args.yaml'
        
        if not args_path.exists():
            raise FileNotFoundError(f"Checkpoint args file not found: {args_path}")
        
        return cls.from_yaml(args_path)
    
    def merge_config(self, other_config: Union[Config, Dict[str, Any]]) -> 'ConfigManager':
        """Merge another configuration with current one"""
        if isinstance(other_config, dict):
            other_config = Config(**other_config)
        
        # Deep merge the configurations
        current_dict = asdict(self.config)
        other_dict = asdict(other_config)
        
        merged_dict = self._deep_merge(current_dict, other_dict)
        self.config = Config(**merged_dict)
        
        return self
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def override_from_dict(self, overrides: Dict[str, Any]) -> 'ConfigManager':
        """Override configuration values using nested dictionary keys"""
        config_dict = asdict(self.config)
        
        for key_path, value in overrides.items():
            keys = key_path.split('.')
            current = config_dict
            
            # Navigate to the nested location
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the value
            current[keys[-1]] = value
        
        self.config = Config(**config_dict)
        return self
    
    def override_from_args(self, args: List[str]) -> 'ConfigManager':
        """Override configuration from command line arguments"""
        parser = argparse.ArgumentParser()
        
        # Add arguments for all configuration paths
        config_dict = asdict(self.config)
        self._add_config_args(parser, config_dict)
        
        # Parse known args to avoid conflicts with other arguments
        parsed_args, _ = parser.parse_known_args(args)
        
        # Convert parsed args to override dictionary
        overrides = {}
        for key, value in parsed_args.__dict__.items():
            if value is not None:
                overrides[key] = value
        
        return self.override_from_dict(overrides)
    
    def _add_config_args(self, parser: argparse.ArgumentParser, config_dict: Dict[str, Any], prefix: str = ""):
        """Recursively add configuration arguments to parser"""
        for key, value in config_dict.items():
            arg_name = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self._add_config_args(parser, value, arg_name)
            else:
                # Determine argument type
                if isinstance(value, bool):
                    parser.add_argument(f"--{arg_name}", type=str2bool, default=value)
                elif isinstance(value, int):
                    parser.add_argument(f"--{arg_name}", type=int, default=value)
                elif isinstance(value, float):
                    parser.add_argument(f"--{arg_name}", type=float, default=value)
                elif isinstance(value, list):
                    parser.add_argument(f"--{arg_name}", type=str, default=str(value))
                else:
                    parser.add_argument(f"--{arg_name}", type=str, default=str(value))
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self.config)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = getattr(current, key)
            return current
        except AttributeError:
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        current = self.config
        
        # Navigate to the parent of the target
        for key in keys[:-1]:
            current = getattr(current, key)
        
        # Set the value
        setattr(current, keys[-1], value)


def str2bool(v: str) -> bool:
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    cmdline_args: Optional[List[str]] = None
) -> Config:
    """
    Load and merge configuration with the specified priority:
    1. Script defaults (lowest priority)
    2. Checkpoint configuration
    3. Config file overrides
    4. Command line overrides (highest priority)
    
    Args:
        config_path: Path to YAML configuration file
        checkpoint_path: Path to checkpoint directory containing args.yaml
        overrides: Dictionary of configuration overrides
        cmdline_args: Command line arguments for overrides
    
    Returns:
        Merged configuration object
    """
    manager = ConfigManager()
    
    # Load checkpoint configuration if provided
    if checkpoint_path:
        try:
            checkpoint_manager = ConfigManager.from_checkpoint(checkpoint_path)
            manager.merge_config(checkpoint_manager.config)
        except FileNotFoundError:
            print(f"Warning: Checkpoint configuration not found at {checkpoint_path}")
    
    # Load config file if provided
    if config_path:
        try:
            config_manager = ConfigManager.from_yaml(config_path)
            manager.merge_config(config_manager.config)
        except FileNotFoundError:
            print(f"Warning: Configuration file not found at {config_path}")
    
    # Apply dictionary overrides
    if overrides:
        manager.override_from_dict(overrides)
    
    # Apply command line overrides
    if cmdline_args:
        manager.override_from_args(cmdline_args)
    
    return manager.config
