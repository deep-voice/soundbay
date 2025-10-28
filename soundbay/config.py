"""
Configuration system using dataclasses
--------------------------------------
This module provides a dataclass-based configuration system that supports:
- Hierarchical configuration with nested dataclasses
- Override priority: cmdline > config > checkpoint > defaults
- Type validation and conversion
- Nested value overrides using dot notation
"""

import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import field, asdict, fields
import argparse
from copy import deepcopy
from pydantic.dataclasses import dataclass
from pydantic import field_validator

from soundbay.models import models_cfg_dict


@dataclass
class AugmentationsConfig:
    """Configuration for augmentations settings"""
    pitch_shift_p: float = 0.0
    time_stretch_p: float = 0.5
    time_masking_p: float = 0.5
    frequency_masking_p: float = 0.5
    min_semitones: int = -4
    max_semitones: int = 4
    min_rate: float = 0.9
    max_rate: float = 1.1
    min_band_part: float = 0.05
    max_band_part: float = 0.2
    min_bandwidth_fraction: float = 0.05
    max_bandwidth_fraction: float = 0.2
    add_multichannel_background_noise_p: float = 0
    min_snr_in_db: int = 3
    max_snr_in_db: int = 30
    lru_cache_size: int = 100
    sounds_path: Optional[str] = None

@dataclass
class DatasetConfig:
    """Configuration for dataset settings"""
    module_name: str = "classifier_dataset"
    data_path: str = "./tests/assets/data/"
    path_hierarchy: int = 0
    mode: str = "train"
    metadata_path: str = "./tests/assets/annotations/sample_annotations.csv"
    augmentations_p: float = 0.8
    augmentations_config: AugmentationsConfig = field(default_factory=AugmentationsConfig)
    margin_ratio: float = 0.5
    slice_flag: bool = False

    @field_validator("augmentations_p")
    def validate_augmentations_p(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("augmentations_p must be between 0 and 1")
        return v
    
    @field_validator("margin_ratio")
    def validate_margin_ratio(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("margin_ratio must be between 0 and 1")
        return v
    
    @field_validator("module_name")
    def validate_module_name(cls, v: str) -> str:
        allowed_values = ["classifier_dataset", "no_background_dataset", "inference_dataset"]
        if v not in allowed_values:
            raise ValueError(f"module_name must be one of {allowed_values}, got {v}")
        return v
    
    @field_validator("mode")
    def validate_mode(cls, v: str) -> str:
        allowed_values = ["train", "val"]
        if v not in allowed_values:
            raise ValueError(f"mode must be one of {allowed_values}, got {v}")
        return v


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
    audio_representation: Optional[str] = "spectrogram"
    normalization: Optional[str] = "peak"
    resize: bool = False
    size: tuple[int, int] = (224, 224)
    n_mels: int = 64
    seq_length: int = 1
    train_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    val_dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(
        mode="val",
        augmentations_p=0.0,
        augmentations_config=AugmentationsConfig(),
        margin_ratio=0.0,
        slice_flag=True
    ))
    
    @field_validator("label_type")
    def validate_label_type(cls, v: str) -> str:
        allowed_values = ["single_label", "multi_label"]
        if v not in allowed_values:
            raise ValueError(f"label_type must be one of {allowed_values}, got {v}")
        return v
    
    @field_validator("audio_representation")
    def validate_audio_representation(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed_values = ["spectrogram", "mel_spectrogram", "sliding_window_spectrogram"]
        if v not in allowed_values:
            raise ValueError(f"audio_representation must be one of {allowed_values}, got {v}")
        return v
    
    @field_validator("normalization")
    def validate_normalization(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed_values = ["peak", "unit"]
        if v not in allowed_values:
            raise ValueError(f"normalization must be one of {allowed_values}, got {v}")
        return v


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint handling"""
    path: Optional[str] = None
    resume: str = 'allow'
    load_optimizer_state: bool = False

    @field_validator("path")
    def validate_path(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not Path(v).exists():
            raise ValueError(f"Checkpoint path does not exist: {v}")
        return v


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
    num_classes: int = 2  # can we allow required here?
    criterion: str = "cross_entropy"
    module_name: str = "ResNet1Channel"
    model_params: Optional[Dict[str, Any]] = None

    @field_validator("criterion")
    def validate_criterion(cls, v: str) -> str:
        allowed_values = ["cross_entropy", "bce_with_logits"]
        if v not in allowed_values:
            raise ValueError(f"criterion must be one of {allowed_values}, got {v}")
        return v
    
    @field_validator("module_name")
    def validate_module_name(cls, v: str) -> str:
        allowed_values = list(models_cfg_dict.keys())
        if v not in allowed_values:
            raise ValueError(f"module_name must be one of {allowed_values}, got {v}")
        return v

    def __post_init__(self):
        # Get dataclass type
        module_cls = models_cfg_dict[self.module_name]
        valid_fields = {f.name for f in fields(module_cls)}

        # Validate provided model_params
        invalid_keys = set(self.model_params or {}) - valid_fields
        if invalid_keys:
            raise ValueError(
                f"Invalid parameters for {self.module_name}: {invalid_keys}. "
                f"Valid fields are: {sorted(valid_fields)}"
            )


@dataclass
class OptimizerConfig:
    """Configuration for optimizer settings"""
    module_name: str = "torch.optim.Adamw"
    lr: float = 5e-4


@dataclass
class SchedulerConfig:
    """Configuration for scheduler settings"""
    module_name: str = "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {"T_0": 5})


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
