"""
Example usage of the new dataclass-based configuration system
------------------------------------------------------------
This script demonstrates practical usage patterns for the new configuration system.
"""

import json
from pathlib import Path
from soundbay.config import Config, ConfigManager, load_config


def example_basic_usage():
    """Example 1: Basic configuration usage"""
    print("=== Example 1: Basic Configuration Usage ===")
    
    # Create configuration with defaults
    config = Config()
    
    print(f"Default batch size: {config.data.batch_size}")
    print(f"Default sample rate: {config.data.sample_rate}")
    print(f"Default learning rate: {config.optim.optimizer.lr}")
    
    # Modify configuration
    config.data.batch_size = 128
    config.data.sample_rate = 22050
    config.optim.optimizer.lr = 1e-3
    
    print(f"Modified batch size: {config.data.batch_size}")
    print(f"Modified sample rate: {config.data.sample_rate}")
    print(f"Modified learning rate: {config.optim.optimizer.lr}")
    print()


def example_yaml_loading():
    """Example 2: Loading from YAML file"""
    print("=== Example 2: Loading from YAML File ===")
    
    # Create a sample YAML configuration
    yaml_content = """
data:
  batch_size: 64
  num_workers: 8
  sample_rate: 16000
  label_names: ['Background', 'Whale', 'Ship']
  
experiment:
  debug: false
  project: 'whale_detection'
  equalize_data: true
  
optim:
  epochs: 200
  optimizer:
    lr: 5e-4
  scheduler:
    T_0: 10
"""
    
    # Save to temporary file
    config_file = Path("temp_config.yaml")
    with open(config_file, 'w') as f:
        f.write(yaml_content)
    
    try:
        # Load configuration from YAML
        config = load_config(config_path=config_file)
        
        print(f"Loaded batch size: {config.data.batch_size}")
        print(f"Loaded label names: {config.data.label_names}")
        print(f"Loaded project: {config.experiment.project}")
        print(f"Loaded epochs: {config.optim.epochs}")
        
        # Verify defaults are preserved for unspecified values
        print(f"Default margin ratio: {config.data.train_dataset.margin_ratio}")
        print(f"Default criterion: {config.model.criterion['_target_']}")
        
    finally:
        config_file.unlink()
    
    print()


def example_override_priority():
    """Example 3: Demonstrating override priority"""
    print("=== Example 3: Override Priority ===")
    
    # Create a base configuration file
    base_yaml = """
data:
  batch_size: 32
  sample_rate: 16000
experiment:
  debug: true
  project: 'base_project'
"""
    
    base_file = Path("base_config.yaml")
    with open(base_file, 'w') as f:
        f.write(base_yaml)
    
    try:
        # Test different override scenarios
        
        # 1. Load with base config only
        config1 = load_config(config_path=base_file)
        print(f"Base config - batch_size: {config1.data.batch_size}, project: {config1.experiment.project}")
        
        # 2. Load with base config + overrides
        overrides = {
            "data.batch_size": 128,
            "experiment.project": "override_project"
        }
        config2 = load_config(config_path=base_file, overrides=overrides)
        print(f"With overrides - batch_size: {config2.data.batch_size}, project: {config2.experiment.project}")
        
        # 3. Load with base config + overrides + command line style args
        cmdline_args = ["--data.sample_rate", "22050", "--experiment.debug", "false"]
        config3 = load_config(config_path=base_file, overrides=overrides, cmdline_args=cmdline_args)
        print(f"With cmdline - sample_rate: {config3.data.sample_rate}, debug: {config3.experiment.debug}")
        
    finally:
        base_file.unlink()
    
    print()


def example_config_manager():
    """Example 4: Using ConfigManager for advanced operations"""
    print("=== Example 4: ConfigManager Advanced Operations ===")
    
    config = Config()
    manager = ConfigManager(config)
    
    # Set values using dot notation
    manager.set("data.batch_size", 256)
    manager.set("data.train_dataset.data_path", "/custom/data/path")
    manager.set("optim.optimizer.lr", 1e-4)
    
    # Get values using dot notation
    batch_size = manager.get("data.batch_size")
    data_path = manager.get("data.train_dataset.data_path")
    lr = manager.get("optim.optimizer.lr")
    
    print(f"Batch size: {batch_size}")
    print(f"Data path: {data_path}")
    print(f"Learning rate: {lr}")
    
    # Get with default value
    nonexistent = manager.get("nonexistent.path", default="default_value")
    print(f"Nonexistent path: {nonexistent}")
    
    # Save configuration
    output_file = Path("example_config.yaml")
    manager.save(output_file)
    print(f"Configuration saved to: {output_file}")
    
    # Clean up
    output_file.unlink()
    print()


def example_nested_configuration():
    """Example 5: Working with nested configurations"""
    print("=== Example 5: Nested Configuration ===")
    
    config = Config()
    
    # Access nested dataset configuration
    print(f"Train dataset mode: {config.data.train_dataset.mode}")
    print(f"Train dataset augmentations_p: {config.data.train_dataset.augmentations_p}")
    print(f"Val dataset mode: {config.data.val_dataset.mode}")
    print(f"Val dataset augmentations_p: {config.data.val_dataset.augmentations_p}")
    
    # Modify nested configurations
    config.data.train_dataset.augmentations_p = 0.9
    config.data.val_dataset.augmentations_p = 0.0
    config.data.train_dataset.seq_length = 5
    config.data.val_dataset.seq_length = 5
    
    print(f"Modified train augmentations_p: {config.data.train_dataset.augmentations_p}")
    print(f"Modified val augmentations_p: {config.data.val_dataset.augmentations_p}")
    print(f"Modified seq_length: {config.data.train_dataset.seq_length}")
    print()


def example_programmatic_config():
    """Example 6: Creating configuration programmatically"""
    print("=== Example 6: Programmatic Configuration ===")
    
    # Create configuration from scratch
    config = Config()
    
    # Set up for a specific experiment
    config.data.label_names = ['Noise', 'Whale', 'Ship', 'Rain']
    config.data.batch_size = 32
    config.data.sample_rate = 22050
    config.data.train_dataset.data_path = "/data/whale_detection/train"
    config.data.val_dataset.data_path = "/data/whale_detection/val"
    config.data.train_dataset.metadata_path = "/data/whale_detection/train_annotations.csv"
    config.data.val_dataset.metadata_path = "/data/whale_detection/val_annotations.csv"
    
    config.experiment.project = "whale_detection_v2"
    config.experiment.debug = False
    config.experiment.equalize_data = True
    
    config.model.model["num_classes"] = 4
    config.model.model["_target_"] = "models.ResNet1Channel"
    
    config.optim.epochs = 150
    config.optim.optimizer.lr = 3e-4
    config.optim.scheduler.T_0 = 15
    
    # Save the configuration
    manager = ConfigManager(config)
    output_file = Path("whale_detection_config.yaml")
    manager.save(output_file)
    
    print(f"Created configuration for {len(config.data.label_names)}-class whale detection")
    print(f"Configuration saved to: {output_file}")
    
    # Clean up
    output_file.unlink()
    print()


def example_validation():
    """Example 7: Configuration validation"""
    print("=== Example 7: Configuration Validation ===")
    
    config = Config()
    
    # Test valid configuration
    try:
        config.data.batch_size = 64
        config.optim.optimizer.lr = 1e-3
        print("✓ Valid configuration created")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
    
    # Test invalid configuration (negative batch size)
    try:
        config.data.batch_size = -1
        print("✗ Should have failed for negative batch size")
    except Exception as e:
        print(f"✓ Correctly caught invalid configuration: {e}")
    
    # Reset to valid value
    config.data.batch_size = 64
    print()


def main():
    """Run all examples"""
    print("Dataclass Configuration System Examples")
    print("=" * 50)
    print()
    
    examples = [
        example_basic_usage,
        example_yaml_loading,
        example_override_priority,
        example_config_manager,
        example_nested_configuration,
        example_programmatic_config,
        example_validation
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Example {example.__name__} failed: {e}")
            print()
    
    print("All examples completed!")


if __name__ == "__main__":
    main()
