"""
Test script for the new dataclass-based configuration system
-----------------------------------------------------------
This script demonstrates the key features of the new configuration system
and validates that it works correctly.
"""

import json
import tempfile
from pathlib import Path
from soundbay.config import Config, ConfigManager, load_config


def test_basic_config():
    """Test basic configuration creation and access"""
    print("Testing basic configuration...")
    
    config = Config()
    
    # Test default values
    assert config.data.batch_size == 64
    assert config.experiment.debug == True
    assert config.optim.epochs == 100
    assert config.model.criterion['_target_'] == 'torch.nn.CrossEntropyLoss'
    
    print("✓ Basic configuration test passed")


def test_config_modification():
    """Test configuration modification"""
    print("Testing configuration modification...")
    
    config = Config()
    
    # Modify values
    config.data.batch_size = 128
    config.experiment.debug = False
    config.optim.optimizer.lr = 1e-3
    
    # Verify modifications
    assert config.data.batch_size == 128
    assert config.experiment.debug == False
    assert config.optim.optimizer.lr == 1e-3
    
    print("✓ Configuration modification test passed")


def test_config_manager():
    """Test ConfigManager functionality"""
    print("Testing ConfigManager...")
    
    config = Config()
    manager = ConfigManager(config)
    
    # Test get/set with dot notation
    manager.set("data.sample_rate", 22050)
    assert manager.get("data.sample_rate") == 22050
    
    # Test default value
    assert manager.get("nonexistent.path", default="default_value") == "default_value"
    
    print("✓ ConfigManager test passed")


def test_yaml_loading():
    """Test YAML configuration loading"""
    print("Testing YAML loading...")
    
    # Create a temporary YAML file
    yaml_content = """
data:
  batch_size: 32
  sample_rate: 22050
experiment:
  debug: false
  project: test_project
optim:
  epochs: 50
  optimizer:
    lr: 1e-4
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        # Load configuration from YAML
        config = load_config(config_path=yaml_path)
        
        # Verify loaded values
        assert config.data.batch_size == 32
        assert config.data.sample_rate == 22050
        assert config.experiment.debug == False
        assert config.experiment.project == "test_project"
        assert config.optim.epochs == 50
        assert config.optim.optimizer.lr == 1e-4
        
        # Verify defaults are preserved
        assert config.data.num_workers == 10  # default value
        assert config.optim.scheduler.T_0 == 5  # default value
        
        print("✓ YAML loading test passed")
        
    finally:
        Path(yaml_path).unlink()


def test_override_priority():
    """Test configuration override priority"""
    print("Testing override priority...")
    
    # Create base configuration
    base_config = Config()
    base_config.data.batch_size = 64
    
    # Test dictionary overrides
    overrides = {
        "data.batch_size": 32,
        "experiment.debug": False,
        "optim.optimizer.lr": 1e-3
    }
    
    config = load_config(overrides=overrides)
    
    # Verify overrides take precedence
    assert config.data.batch_size == 32
    assert config.experiment.debug == False
    assert config.optim.optimizer.lr == 1e-3
    
    # Verify other defaults are preserved
    assert config.data.num_workers == 10
    assert config.optim.epochs == 100
    
    print("✓ Override priority test passed")


def test_nested_overrides():
    """Test nested configuration overrides"""
    print("Testing nested overrides...")
    
    overrides = {
        "data.train_dataset.data_path": "/custom/path",
        "data.train_dataset.augmentations_p": 0.9,
        "optim.scheduler.T_0": 10
    }
    
    config = load_config(overrides=overrides)
    
    # Verify nested overrides
    assert config.data.train_dataset.data_path == "/custom/path"
    assert config.data.train_dataset.augmentations_p == 0.9
    assert config.optim.scheduler.T_0 == 10
    
    # Verify other nested defaults are preserved
    assert config.data.train_dataset.mode == "train"
    assert config.data.val_dataset.mode == "val"
    
    print("✓ Nested overrides test passed")


def test_config_saving():
    """Test configuration saving"""
    print("Testing configuration saving...")
    
    config = Config()
    config.data.batch_size = 128
    config.experiment.project = "test_save"
    
    manager = ConfigManager(config)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        manager.save(temp_path)
        
        # Load back and verify
        loaded_config = load_config(config_path=temp_path)
        
        assert loaded_config.data.batch_size == 128
        assert loaded_config.experiment.project == "test_save"
        
        print("✓ Configuration saving test passed")
        
    finally:
        Path(temp_path).unlink()


def test_complex_config():
    """Test complex configuration scenarios"""
    print("Testing complex configuration...")
    
    # Test with multiple overrides and nested structures
    overrides = {
        "data.label_names": ["Class1", "Class2", "Class3"],
        "data.train_dataset._target_": "soundbay.data.CustomDataset",
        "model.model.num_classes": 3,
        "experiment.checkpoint.resume": "must"
    }
    
    config = load_config(overrides=overrides)
    
    # Verify complex overrides
    assert config.data.label_names == ["Class1", "Class2", "Class3"]
    assert config.data.train_dataset._target_ == "soundbay.data.CustomDataset"
    assert config.model.model["num_classes"] == 3
    assert config.experiment.checkpoint.resume == "must"
    
    print("✓ Complex configuration test passed")


def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")
    
    # Test invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        yaml_path = f.name
    
    try:
        # This should handle the error gracefully
        config = load_config(config_path=yaml_path)
        # Should fall back to defaults
        assert config.data.batch_size == 64
        
        print("✓ Error handling test passed")
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
    finally:
        Path(yaml_path).unlink()


def main():
    """Run all configuration tests"""
    print("Running dataclass configuration system tests...\n")
    
    tests = [
        test_basic_config,
        test_config_modification,
        test_config_manager,
        test_yaml_loading,
        test_override_priority,
        test_nested_overrides,
        test_config_saving,
        test_complex_config,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The configuration system is working correctly.")
    else:
        print("❌ Some tests failed. Please check the configuration system.")


if __name__ == "__main__":
    main()
