# Dataclass-Based Configuration System

This document describes the new dataclass-based configuration system that replaces Hydra in the SoundBay framework.

## Overview

The new configuration system addresses the following issues with Hydra:

1. **Splits logic between code and configurations** - Now all configuration logic is centralized in dataclasses
2. **Reduces configuration overhead** - Single configuration file instead of multiple nested files
3. **Eliminates Hydra complexity** - No more decorators or path overrides to manage
4. **Better type safety** - Full type hints and validation
5. **Simpler override system** - Clear priority hierarchy for configuration overrides

## Features

### Hierarchical Configuration
The configuration is organized into logical groups using nested dataclasses:

- `DataConfig` - Dataset and data processing settings
- `ExperimentConfig` - Experiment tracking and logging settings  
- `ModelConfig` - Model architecture and criterion settings
- `OptimConfig` - Optimizer and training settings

### Override Priority System
Configuration values are resolved with the following priority (highest to lowest):

1. **Command line overrides** - Direct parameter overrides
2. **Config file overrides** - Values from YAML configuration files
3. **Checkpoint overrides** - Values from saved experiment checkpoints
4. **Script defaults** - Default values defined in dataclasses

### Type Safety and Validation
- Full type hints for all configuration parameters
- Automatic type conversion and validation
- Runtime validation of configuration values
- Clear error messages for invalid configurations

## Usage

### Basic Usage

```python
from soundbay.config import load_config

# Load with defaults only
config = load_config()

# Load with configuration file
config = load_config(config_path="path/to/config.yaml")

# Load with checkpoint
config = load_config(checkpoint_path="path/to/checkpoint")

# Load with overrides
config = load_config(
    config_path="path/to/config.yaml",
    overrides={"data.batch_size": 32, "experiment.debug": False}
)
```

### Command Line Usage

```bash
# Basic training with defaults
python train_new.py

# Training with configuration file
python train_new.py --config_path config.yaml

# Training with checkpoint
python train_new.py --checkpoint_path checkpoints/experiment_id

# Training with overrides
python train_new.py --data.batch_size 32 --experiment.debug False

# Training with JSON overrides
python train_new.py --overrides '{"data.batch_size": 32, "experiment.debug": false}'
```

### Programmatic Usage

```python
from soundbay.config import Config, ConfigManager

# Create configuration programmatically
config = Config()
config.data.batch_size = 64
config.experiment.debug = True

# Use ConfigManager for advanced operations
manager = ConfigManager(config)
manager.set("data.sample_rate", 22050)
value = manager.get("experiment.project", default="default_project")

# Save configuration
manager.save("my_config.yaml")
```

## Configuration Structure

### Data Configuration

```python
@dataclass
class DataConfig:
    label_names: List[str] = ['Noise', 'Call']
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
    val_dataset: DatasetConfig = field(default_factory=DatasetConfig)
```

### Experiment Configuration

```python
@dataclass
class ExperimentConfig:
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
```

### Model Configuration

```python
@dataclass
class ModelConfig:
    criterion: Dict[str, str] = field(default_factory=lambda: {"_target_": "torch.nn.CrossEntropyLoss"})
    model: Dict[str, Any] = field(default_factory=lambda: {
        "_target_": "models.ResNet1Channel",
        "layers": [3, 4, 6, 3],
        "block": "torchvision.models.resnet.Bottleneck",
        "num_classes": 2
    })
```

### Optimization Configuration

```python
@dataclass
class OptimConfig:
    epochs: int = 100
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    freeze_layers_for_finetune: bool = True
```

## Migration from Hydra

### Automatic Migration

Use the migration utility to convert existing Hydra configurations:

```bash
# Migrate single file
python -m soundbay.migrate_config path/to/hydra_config.yaml

# Migrate all files in directory
python -m soundbay.migrate_config path/to/config/directory --batch

# Migrate with custom output
python -m soundbay.migrate_config input.yaml --output converted.yaml
```

### Manual Migration

1. **Remove Hydra-specific keys** from YAML files:
   - Remove `defaults:` sections
   - Remove `# @package _global_` comments
   - Remove variable interpolation (`${...}`)

2. **Flatten configuration structure**:
   - Move nested configurations to top level
   - Ensure all required fields are present

3. **Update training script**:
   - Replace `@hydra.main` decorator with `load_config()` function
   - Update configuration access patterns
   - Remove Hydra-specific path handling

### Example Migration

**Before (Hydra):**
```yaml
# @package _global_
defaults:
  - ../data: defaults
  - ../model: defaults
  - ../optim: defaults
  - ../experiment: defaults

data:
  batch_size: 32
```

**After (Dataclass):**
```yaml
data:
  label_names: ['Noise', 'Call']
  batch_size: 32
  num_workers: 10
  # ... other data fields

experiment:
  debug: true
  manual_seed: 1234
  # ... other experiment fields

model:
  criterion:
    _target_: torch.nn.CrossEntropyLoss
  model:
    _target_: models.ResNet1Channel
    # ... other model fields

optim:
  epochs: 100
  optimizer:
    _target_: torch.optim.Adam
    lr: 5e-4
  # ... other optim fields
```

## Advanced Features

### Custom Configuration Classes

You can extend the configuration system with custom dataclasses:

```python
@dataclass
class CustomConfig:
    custom_param: str = "default_value"
    custom_list: List[int] = field(default_factory=lambda: [1, 2, 3])

@dataclass
class ExtendedConfig(Config):
    custom: CustomConfig = field(default_factory=CustomConfig)
```

### Validation and Constraints

Add validation to configuration values:

```python
@dataclass
class ValidatedConfig:
    batch_size: int = 64
    
    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.batch_size > 1024:
            raise ValueError("Batch size too large")
```

### Environment Variable Support

Load configuration from environment variables:

```python
import os
from soundbay.config import Config

config = Config()
config.experiment.project = os.getenv("WANDB_PROJECT", "default_project")
config.data.batch_size = int(os.getenv("BATCH_SIZE", "64"))
```

## Best Practices

1. **Use meaningful defaults** - Set sensible default values for all parameters
2. **Document configuration** - Add docstrings to dataclass fields
3. **Validate configurations** - Use `__post_init__` for validation logic
4. **Version configurations** - Include version information in configuration files
5. **Use type hints** - Always specify types for better IDE support and validation

## Troubleshooting

### Common Issues

1. **Type conversion errors** - Ensure YAML values match expected types
2. **Missing required fields** - Check that all required fields are provided
3. **Path resolution issues** - Use absolute paths or relative to working directory
4. **Override conflicts** - Verify override priority is working as expected

### Debug Configuration

Enable debug mode to see configuration loading details:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

config = load_config(config_path="debug_config.yaml")
```

## Comparison with Hydra

| Feature | Hydra | Dataclass System |
|---------|-------|------------------|
| Configuration files | Multiple nested files | Single flat file |
| Type safety | Limited | Full type hints |
| Override complexity | High | Simple dot notation |
| Learning curve | Steep | Gentle |
| IDE support | Limited | Excellent |
| Validation | External | Built-in |
| Path handling | Automatic (problematic) | Manual (explicit) |
| Dependencies | Heavy | Lightweight |

The new dataclass-based system provides a simpler, more maintainable, and type-safe alternative to Hydra while maintaining all the essential functionality for configuration management.
