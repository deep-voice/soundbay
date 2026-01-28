"""
Backward Compatibility Tests for Inference
-------------------------------------------
These tests ensure that the refactored inference code produces the same
outputs as the original Hydra-based implementation.

Reference files were generated using the original inference command:
    python inference.py --config-name runs/inference_single_audio \
        experiment.checkpoint.path=/path/to/models/zmr2i3ff/best.pth \
        data.test_dataset.file_path=../tests/assets/backward_compatibility_assets/NOPP6_EST_20090331_220000_CH10.wav \
        data.data_sample_rate=2000 \
        experiment.save_raven=yes
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import torch

from soundbay.inference import run_inference
from soundbay.config import InferenceConfig, create_inference_config
from soundbay.utils.checkpoint_utils import merge_inference_config_with_checkpoint


# Paths to test assets
TESTS_DIR = Path(__file__).parent
ASSETS_DIR = TESTS_DIR / "assets"
BACKWARD_COMPAT_DIR = ASSETS_DIR / "backward_compatibility_assets"
REFERENCE_CSV = BACKWARD_COMPAT_DIR / "Inference_results-2025-05-03_20-27-38-zmr2i3ff-NOPP6_EST_20090331_220000_CH10.csv"
REFERENCE_RAVEN = BACKWARD_COMPAT_DIR / "NOPP6_EST_20090331_220000_CH10-Raven-inference_results-2025-05-03_20-27-38-zmr2i3ff.txt"
TEST_AUDIO = BACKWARD_COMPAT_DIR / "NOPP6_EST_20090331_220000_CH10.wav"

# Model checkpoint path - using relative path from workspace root
CHECKPOINT_PATH = Path(__file__).parent.parent / "models" / "zmr2i3ff" / "best.pth"


def _probability_columns(df: pd.DataFrame) -> list:
    """Get the probability column names (last N columns that are numeric)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out known non-probability columns
    non_prob_cols = {'channel', 'begin_time', 'end_time'}
    return [c for c in numeric_cols if c not in non_prob_cols]


@pytest.fixture
def reference_csv():
    """Load the reference CSV output."""
    if not REFERENCE_CSV.exists():
        pytest.skip(f"Reference CSV not found: {REFERENCE_CSV}")
    return pd.read_csv(REFERENCE_CSV)


@pytest.fixture
def reference_raven():
    """Load the reference Raven output."""
    if not REFERENCE_RAVEN.exists():
        pytest.skip(f"Reference Raven file not found: {REFERENCE_RAVEN}")
    return pd.read_csv(REFERENCE_RAVEN, sep='\t')


@pytest.fixture
def checkpoint():
    """Load the model checkpoint."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")
    return torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'), weights_only=False)


@pytest.fixture
def inference_config(checkpoint):
    """Create inference config merged with checkpoint args."""
    # Create base config with test parameters
    config = create_inference_config(overrides=[
        f"data.test_dataset.file_path={TEST_AUDIO}",
        "data.data_sample_rate=2000",
        "experiment.save_raven=true",
    ])
    
    # Merge with checkpoint args
    config = merge_inference_config_with_checkpoint(config, checkpoint['args'])
    return config


class TestInferenceBackwardCompatibility:
    """Test that inference outputs match the reference outputs."""
    
    def test_reference_files_exist(self):
        """Verify that reference files exist for testing."""
        assert REFERENCE_CSV.exists(), f"Reference CSV not found: {REFERENCE_CSV}"
        assert REFERENCE_RAVEN.exists(), f"Reference Raven file not found: {REFERENCE_RAVEN}"
        assert TEST_AUDIO.exists(), f"Test audio file not found: {TEST_AUDIO}"
    
    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_inference_output_structure(self, checkpoint, inference_config, reference_csv):
        """Test that inference output has the same structure as reference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            
            # Run inference
            output_file = run_inference(
                args=inference_config,
                checkpoint_state_dict=checkpoint['model'],
                output_path=output_path,
                model_name="zmr2i3ff"
            )
            
            # Load output
            output_df = pd.read_csv(output_file)
            
            # Check column structure
            assert set(output_df.columns) == set(reference_csv.columns), \
                f"Column mismatch. Got: {output_df.columns.tolist()}, Expected: {reference_csv.columns.tolist()}"
            
            # Check row count
            assert len(output_df) == len(reference_csv), \
                f"Row count mismatch. Got: {len(output_df)}, Expected: {len(reference_csv)}"
    
    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_inference_probabilities_match(self, checkpoint, inference_config, reference_csv):
        """Test that inference probabilities match the reference within tolerance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            
            # Run inference
            output_file = run_inference(
                args=inference_config,
                checkpoint_state_dict=checkpoint['model'],
                output_path=output_path,
                model_name="zmr2i3ff"
            )
            
            # Load output
            output_df = pd.read_csv(output_file)
            
            # Get probability columns
            prob_cols = _probability_columns(reference_csv)
            
            # Sort both dataframes by filename and begin_time for comparison
            ref_sorted = reference_csv.sort_values(['filename', 'begin_time']).reset_index(drop=True)
            out_sorted = output_df.sort_values(['filename', 'begin_time']).reset_index(drop=True)
            
            # Compare probabilities with tolerance
            for col in prob_cols:
                ref_values = ref_sorted[col].values
                out_values = out_sorted[col].values
                
                np.testing.assert_allclose(
                    out_values, 
                    ref_values, 
                    rtol=1e-5, 
                    atol=1e-5,
                    err_msg=f"Probability mismatch in column '{col}'"
                )
    
    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_inference_timestamps_match(self, checkpoint, inference_config, reference_csv):
        """Test that inference timestamps match the reference exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            
            # Run inference
            output_file = run_inference(
                args=inference_config,
                checkpoint_state_dict=checkpoint['model'],
                output_path=output_path,
                model_name="zmr2i3ff"
            )
            
            # Load output
            output_df = pd.read_csv(output_file)
            
            # Sort both dataframes
            ref_sorted = reference_csv.sort_values(['filename', 'begin_time']).reset_index(drop=True)
            out_sorted = output_df.sort_values(['filename', 'begin_time']).reset_index(drop=True)
            
            # Compare timestamps
            np.testing.assert_array_equal(
                out_sorted['begin_time'].values,
                ref_sorted['begin_time'].values,
                err_msg="Begin time mismatch"
            )
            
            np.testing.assert_array_equal(
                out_sorted['end_time'].values,
                ref_sorted['end_time'].values,
                err_msg="End time mismatch"
            )
    
    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_raven_output_structure(self, checkpoint, inference_config, reference_raven):
        """Test that Raven output has the same structure as reference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            
            # Run inference with save_raven=True
            inference_config.experiment.save_raven = True
            
            run_inference(
                args=inference_config,
                checkpoint_state_dict=checkpoint['model'],
                output_path=output_path,
                model_name="zmr2i3ff"
            )
            
            # Find the generated Raven file
            raven_files = list(output_path.glob("*-Raven-*.txt"))
            assert len(raven_files) == 1, f"Expected 1 Raven file, found {len(raven_files)}"
            
            # Load output
            output_raven = pd.read_csv(raven_files[0], sep='\t')
            
            # Check column structure
            assert set(output_raven.columns) == set(reference_raven.columns), \
                f"Raven column mismatch. Got: {output_raven.columns.tolist()}"
    
    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_raven_detections_match(self, checkpoint, inference_config, reference_raven):
        """Test that Raven detections match the reference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            
            # Run inference with save_raven=True
            inference_config.experiment.save_raven = True
            
            run_inference(
                args=inference_config,
                checkpoint_state_dict=checkpoint['model'],
                output_path=output_path,
                model_name="zmr2i3ff"
            )
            
            # Find the generated Raven file
            raven_files = list(output_path.glob("*-Raven-*.txt"))
            output_raven = pd.read_csv(raven_files[0], sep='\t')
            
            # Check detection count
            assert len(output_raven) == len(reference_raven), \
                f"Detection count mismatch. Got: {len(output_raven)}, Expected: {len(reference_raven)}"
            
            # Sort both by begin time and compare
            ref_sorted = reference_raven.sort_values('Begin Time (s)').reset_index(drop=True)
            out_sorted = output_raven.sort_values('Begin Time (s)').reset_index(drop=True)
            
            # Compare begin and end times
            np.testing.assert_array_almost_equal(
                out_sorted['Begin Time (s)'].values,
                ref_sorted['Begin Time (s)'].values,
                decimal=3,
                err_msg="Raven begin time mismatch"
            )
            
            np.testing.assert_array_almost_equal(
                out_sorted['End Time (s)'].values,
                ref_sorted['End Time (s)'].values,
                decimal=3,
                err_msg="Raven end time mismatch"
            )
            
            # Compare class names
            assert list(out_sorted['Class Name'].values) == list(ref_sorted['Class Name'].values), \
                "Raven class name mismatch"


class TestInferenceConfig:
    """Test inference configuration handling."""
    
    def test_create_inference_config_defaults(self):
        """Test that default inference config is created correctly."""
        config = create_inference_config()
        
        assert isinstance(config, InferenceConfig)
        assert config.data.batch_size == 64
        assert config.data.test_dataset.module_name == "InferenceDataset"
        assert config.experiment.threshold == 0.5
    
    def test_create_inference_config_with_overrides(self):
        """Test that config overrides work correctly."""
        config = create_inference_config(overrides=[
            "data.batch_size=32",
            "data.data_sample_rate=2000",
            "experiment.threshold=0.7"
        ])
        
        assert config.data.batch_size == 32
        assert config.data.data_sample_rate == 2000
        assert config.experiment.threshold == 0.7
    
    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_merge_with_checkpoint_extracts_model_config(self, checkpoint):
        """Test that checkpoint merging extracts model configuration."""
        config = create_inference_config()
        merged = merge_inference_config_with_checkpoint(config, checkpoint['args'])
        
        # Should have extracted model info from checkpoint
        assert merged.model.module_name != ""
        assert merged.model.num_classes > 0
    
    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_merge_with_checkpoint_extracts_data_config(self, checkpoint):
        """Test that checkpoint merging extracts data configuration."""
        config = create_inference_config()
        merged = merge_inference_config_with_checkpoint(config, checkpoint['args'])
        
        # Should have extracted data params from checkpoint
        assert merged.data.label_names is not None
        assert len(merged.data.label_names) > 0
        assert merged.data.seq_length > 0
