"""
Configuration system using dataclasses
--------------------------------------
This module provides a dataclass-based configuration system that supports:
- Hierarchical configuration with nested dataclasses
- Override priority: cmdline > config > checkpoint > defaults
- Type validation and conversion
- Nested value overrides using dot notation
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import field, asdict, fields
from copy import deepcopy
from pydantic.dataclasses import dataclass
from pydantic import field_validator
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
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


def merge_configs(base_config: Config, override_config: Config) -> Config:
    """Merge two configs into a single config"""
    return OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(base_config), OmegaConf.structured(override_config)))

def merge_config_with_overrides(base_config: Config, overrides: DictConfig) -> Config:
    """Merge a config with overrides"""
    return OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(base_config), overrides))