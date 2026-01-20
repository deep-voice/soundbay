"""
Raven Intelligence Export Module

This module provides utilities to export soundbay models to Raven Intelligence format.
It wraps trained models with their preprocessing pipeline to create standalone models
that accept raw audio input, compatible with Raven Intelligence's expected interface.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torchaudio
from hydra.utils import instantiate
from omegaconf import OmegaConf


def _get_model_class(model_target: str):
    """
    Lazy import of model classes to avoid transformers dependency when not needed.
    """
    if model_target == 'models.ResNet182D':
        # Import inline to avoid loading transformers-dependent modules
        import torch.nn as nn
        from torchvision import models as tv_models
        from torchvision.models import ResNet18_Weights

        class ResNet182D(nn.Module):
            def __init__(self, num_classes=2, pretrained=True):
                super(ResNet182D, self).__init__()
                resnet = tv_models.resnet18(weights=ResNet18_Weights.DEFAULT) if pretrained else tv_models.resnet18(weights=None)
                num_features = resnet.fc.in_features
                resnet.fc = nn.Sequential(
                    nn.Linear(num_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
                self.resnet = resnet

            def forward(self, x):
                x = x.repeat(1, 3, 1, 1)
                return self.resnet(x)

        return ResNet182D

    elif model_target == 'models.Squeezenet2D':
        import torch.nn as nn

        class Squeezenet2D(nn.Module):
            def __init__(self, num_classes=2, pretrained=True):
                super(Squeezenet2D, self).__init__()
                self.squeezenet = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=pretrained)
                num_features = self.squeezenet.classifier[1].out_channels
                self.custom_classifier = nn.Sequential(
                    nn.Linear(num_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )

            def forward(self, x):
                x = x.repeat(1, 3, 1, 1)
                rep = self.squeezenet(x)
                return self.custom_classifier(rep)

        return Squeezenet2D

    else:
        raise ValueError(f"Unknown or unsupported model type for export: {model_target}. "
                        f"Supported: models.ResNet182D, models.Squeezenet2D")


class PeakNormalizeModule(nn.Module):
    """nn.Module version of PeakNormalize for export compatibility."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-8)


class UnitNormalizeModule(nn.Module):
    """nn.Module version of UnitNormalize for export compatibility."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        return (x - mean) / (std + 1e-8)


class PreprocessingPipeline(nn.Module):
    """
    A torch.nn.Module that encapsulates the preprocessing pipeline from args.yaml.

    This module applies:
    1. Resampling (if data_sample_rate != sample_rate)
    2. Preprocessors (MelSpectrogram, AmplitudeToDB, normalization, etc.)

    All operations are performed using torch operations to ensure export compatibility.
    """

    def __init__(
        self,
        data_sample_rate: int,
        sample_rate: int,
        preprocessors_config: Dict,
    ):
        """
        Initialize the preprocessing pipeline.

        Args:
            data_sample_rate: Original sample rate of input audio
            sample_rate: Target sample rate for the model
            preprocessors_config: Dictionary of preprocessor configurations from args.yaml
        """
        super().__init__()

        self.data_sample_rate = data_sample_rate
        self.sample_rate = sample_rate

        # Resampler
        if data_sample_rate != sample_rate:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=data_sample_rate,
                new_freq=sample_rate
            )
        else:
            self.resampler = nn.Identity()

        # Build preprocessor pipeline
        self.preprocessors = self._build_preprocessors(preprocessors_config)

    def _build_preprocessors(self, preprocessors_config: Dict) -> nn.Sequential:
        """
        Build the preprocessing pipeline from config.

        Handles special cases like PeakNormalize which need to be converted
        to proper nn.Module instances for export.
        """
        processors = []

        if preprocessors_config is None or len(preprocessors_config) == 0:
            return nn.Identity()

        for name, config in preprocessors_config.items():
            target = config.get('_target_', '')

            if 'PeakNormalize' in target:
                processors.append(PeakNormalizeModule())
            elif 'UnitNormalize' in target:
                processors.append(UnitNormalizeModule())
            else:
                # Use hydra instantiate for standard transforms
                processor = instantiate(config)
                processors.append(processor)

        return nn.Sequential(*processors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply preprocessing to raw audio.

        Args:
            x: Raw audio tensor of shape (batch, samples) or (batch, 1, samples)

        Returns:
            Preprocessed tensor ready for the model
        """
        # Ensure correct shape: (batch, 1, samples)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Apply resampling
        x = self.resampler(x)

        # Apply preprocessors
        x = self.preprocessors(x)

        return x


class RavenExportModel(nn.Module):
    """
    A wrapper model that combines preprocessing and the trained model.

    This model accepts raw audio as input and outputs class scores,
    matching Raven Intelligence's expected interface.

    Input: Raw audio tensor (batch, samples) at data_sample_rate
    Output: Class probabilities (batch, num_classes)
    """

    def __init__(
        self,
        model: nn.Module,
        preprocessing: PreprocessingPipeline,
        apply_softmax: bool = True,
    ):
        """
        Initialize the export model.

        Args:
            model: The trained soundbay model
            preprocessing: The preprocessing pipeline
            apply_softmax: Whether to apply softmax to outputs (recommended for Raven)
        """
        super().__init__()

        self.preprocessing = preprocessing
        self.model = model
        self.apply_softmax = apply_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from raw audio to class scores.

        Args:
            x: Raw audio tensor of shape (batch, samples)

        Returns:
            Class probabilities of shape (batch, num_classes)
        """
        # Apply preprocessing
        x = self.preprocessing(x)

        # Run through model
        logits = self.model(x)

        # Apply softmax for probabilities
        if self.apply_softmax:
            return torch.softmax(logits, dim=-1)
        return logits

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        args_path: Optional[Union[str, Path]] = None,
        apply_softmax: bool = True,
    ) -> 'RavenExportModel':
        """
        Create a RavenExportModel from a soundbay checkpoint.

        Args:
            checkpoint_path: Path to the .pth checkpoint file
            args_path: Path to args.yaml. If None, looks for it in the same directory
            apply_softmax: Whether to apply softmax to outputs

        Returns:
            RavenExportModel ready for export
        """
        checkpoint_path = Path(checkpoint_path)

        # Find args.yaml
        if args_path is None:
            args_path = checkpoint_path.parent / 'args.yaml'
        else:
            args_path = Path(args_path)

        if not args_path.exists():
            raise FileNotFoundError(f"args.yaml not found at {args_path}")

        # Load configuration
        args = OmegaConf.load(args_path)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Create model
        model_config = args.model.model
        model_target = model_config._target_
        model_class = _get_model_class(model_target)

        # Get model kwargs (excluding _target_)
        # Force pretrained=False since we're loading weights from checkpoint
        model_kwargs = {k: v for k, v in model_config.items() if k != '_target_'}
        model_kwargs['pretrained'] = False
        model = model_class(**model_kwargs)

        # Load weights
        model.load_state_dict(checkpoint['model'])
        model.eval()

        # Create preprocessing pipeline
        preprocessing = PreprocessingPipeline(
            data_sample_rate=args.data.data_sample_rate,
            sample_rate=args.data.sample_rate,
            preprocessors_config=dict(args.get('_preprocessors', {})),
        )

        return cls(model=model, preprocessing=preprocessing, apply_softmax=apply_softmax)

    def get_config(self, checkpoint_path: Union[str, Path]) -> dict:
        """
        Get configuration info needed for .ravenmodel file.

        Args:
            checkpoint_path: Path to the original checkpoint (for loading args.yaml)

        Returns:
            Dictionary with model configuration
        """
        checkpoint_path = Path(checkpoint_path)
        args_path = checkpoint_path.parent / 'args.yaml'
        args = OmegaConf.load(args_path)

        return {
            'data_sample_rate': args.data.data_sample_rate,
            'sample_rate': args.data.sample_rate,
            'seq_length': args.data.train_dataset.seq_length,
            'num_classes': args.model.model.num_classes,
            'label_names': list(args.data.label_names),
        }


def export_to_onnx(
    model: RavenExportModel,
    output_path: Union[str, Path],
    sample_rate: int,
    seq_length: float,
    opset_version: int = 14,
) -> Path:
    """
    Export a RavenExportModel to ONNX format.

    Args:
        model: The RavenExportModel to export
        output_path: Path for the output .onnx file
        sample_rate: Input sample rate (for calculating input size)
        seq_length: Sequence length in seconds
        opset_version: ONNX opset version

    Returns:
        Path to the exported ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Calculate input size: samples = sample_rate * seq_length
    num_samples = int(sample_rate * seq_length)

    # Create dummy input (batch_size=1)
    dummy_input = torch.randn(1, num_samples)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['audio'],
        output_names=['scores'],
        dynamic_axes={
            'audio': {0: 'batch_size'},
            'scores': {0: 'batch_size'},
        },
    )

    print(f"Exported ONNX model to {output_path}")
    return output_path


def export_to_torchscript(
    model: RavenExportModel,
    output_path: Union[str, Path],
    sample_rate: int,
    seq_length: float,
) -> Path:
    """
    Export a RavenExportModel to TorchScript format.

    TorchScript is preferred when ONNX export fails (e.g., STFT not supported).
    DJL supports loading TorchScript models via the PyTorch engine.

    Args:
        model: The RavenExportModel to export
        output_path: Path for the output .pt file
        sample_rate: Input sample rate (for calculating input size)
        seq_length: Sequence length in seconds

    Returns:
        Path to the exported TorchScript file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Calculate input size: samples = sample_rate * seq_length
    num_samples = int(sample_rate * seq_length)

    # Create dummy input (batch_size=1)
    dummy_input = torch.randn(1, num_samples)

    # Use tracing for export (works better with dynamic ops like STFT)
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)

    # Save the traced model
    traced_model.save(str(output_path))

    print(f"Exported TorchScript model to {output_path}")
    return output_path


def create_raven_model_package(
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str,
    apply_softmax: bool = True,
    export_format: str = 'torchscript',
) -> Path:
    """
    Create a complete Raven Intelligence model package from a soundbay checkpoint.

    This creates:
    - {model_name}.ravenmodel (JSON configuration)
    - {model_name}/model.pt or model.onnx (exported model)
    - {model_name}/labels.txt (class labels)

    Args:
        checkpoint_path: Path to the soundbay .pth checkpoint
        output_dir: Directory to create the model package in
        model_name: Name for the model
        apply_softmax: Whether to apply softmax in the model
        export_format: 'torchscript' (default, recommended) or 'onnx'

    Returns:
        Path to the .ravenmodel file
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)

    # Load args for configuration
    args_path = checkpoint_path.parent / 'args.yaml'
    args = OmegaConf.load(args_path)

    # Create the export model
    print(f"Loading checkpoint from {checkpoint_path}...")
    export_model = RavenExportModel.from_checkpoint(
        checkpoint_path=checkpoint_path,
        apply_softmax=apply_softmax,
    )

    # Get configuration
    config = export_model.get_config(checkpoint_path)

    # Create model directory
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Export model based on format
    if export_format == 'torchscript':
        model_path = model_dir / 'model.pt'
        export_to_torchscript(
            model=export_model,
            output_path=model_path,
            sample_rate=config['data_sample_rate'],
            seq_length=config['seq_length'],
        )
        engine = "PYTORCH"
    elif export_format == 'onnx':
        model_path = model_dir / 'model.onnx'
        export_to_onnx(
            model=export_model,
            output_path=model_path,
            sample_rate=config['data_sample_rate'],
            seq_length=config['seq_length'],
        )
        engine = "ONNX_RUNTIME"
    else:
        raise ValueError(f"Unknown export format: {export_format}. Use 'torchscript' or 'onnx'")

    # Create labels file
    labels_path = model_dir / 'labels.txt'
    with open(labels_path, 'w') as f:
        for label in config['label_names']:
            f.write(f"{label}\n")
    print(f"Created labels file at {labels_path}")

    # Calculate input dimensions
    num_samples = int(config['data_sample_rate'] * config['seq_length'])

    # Create .ravenmodel configuration
    ravenmodel_config = {
        "name": model_name,
        "engine": engine,
        "modelDirectory": {
            "path": model_name
        },
        "labelsFilePath": {
            "path": f"{model_name}/labels.txt"
        },
        "numLabels": config['num_classes'],
        "availableSignatures": ["default"],
        "chosenSignature": "default",
        "availableInputs": ["audio"],
        "availableOutputs": ["scores"],
        "inputTensors": {
            "audio": {
                "dimensions": [-1, num_samples],
                "dataType": "FLOAT32",
                "audio": True,
                "value": None,
                "file": None
            }
        },
        "outputTensors": {
            "scores": {
                "dimensions": [-1, config['num_classes']],
                "dataType": "FLOAT32",
                "audio": False,
                "value": None,
                "file": None
            }
        },
        "scoresTensor": "scores",
        "minSampleDuration": int(config['seq_length']),
        "maxSampleDuration": int(config['seq_length']),
        "minSampleRate": config['data_sample_rate'],
        "maxSampleRate": config['data_sample_rate']
    }

    # Write .ravenmodel file
    ravenmodel_path = output_dir / f"{model_name}.ravenmodel"
    with open(ravenmodel_path, 'w') as f:
        json.dump(ravenmodel_config, f, indent=2)
    print(f"Created .ravenmodel file at {ravenmodel_path}")

    print(f"\nRaven model package created successfully!")
    print(f"  Model directory: {model_dir}")
    print(f"  Configuration: {ravenmodel_path}")
    print(f"\nTo use in Raven Intelligence:")
    print(f"  1. Copy both '{model_name}.ravenmodel' and '{model_name}/' to:")
    print(f"     ~/Raven Workbench/Raven Intelligence/Models/")
    print(f"  2. Restart Raven Intelligence and select the model")

    return ravenmodel_path


def create_raven_model_package_from_torchscript(
    torchscript_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str,
    label_names: List[str],
    sample_rate: int,
    seq_length: float,
    input_channels: int = 1,
) -> Path:
    """
    Create a Raven Intelligence model package from a pre-existing TorchScript model.

    Use this when you have a TorchScript model (.pt) that already has preprocessing
    embedded, rather than a soundbay checkpoint.

    This creates:
    - {model_name}.ravenmodel (JSON configuration)
    - {model_name}/model.pt (copied TorchScript model)
    - {model_name}/labels.txt (class labels)

    Args:
        torchscript_path: Path to the TorchScript model (.pt file)
        output_dir: Directory to create the model package in
        model_name: Name for the model
        label_names: List of class label names (e.g., ['Noise', 'HUWH', 'RIWH'])
        sample_rate: Expected input sample rate in Hz
        seq_length: Expected input duration in seconds
        input_channels: Number of input audio channels (default: 1)

    Returns:
        Path to the .ravenmodel file
    """
    import shutil

    torchscript_path = Path(torchscript_path)
    output_dir = Path(output_dir)

    if not torchscript_path.exists():
        raise FileNotFoundError(f"TorchScript model not found: {torchscript_path}")

    # Create model directory
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Copy the TorchScript model
    model_dest = model_dir / 'model.pt'
    shutil.copy(torchscript_path, model_dest)
    print(f"Copied TorchScript model to {model_dest}")

    # Create labels file
    labels_path = model_dir / 'labels.txt'
    with open(labels_path, 'w') as f:
        for label in label_names:
            f.write(f"{label}\n")
    print(f"Created labels file at {labels_path}")

    # Calculate input dimensions
    num_samples = int(sample_rate * seq_length)
    num_classes = len(label_names)

    # Determine input tensor dimensions based on channels
    if input_channels == 1:
        input_dims = [-1, input_channels, num_samples]
    else:
        input_dims = [-1, input_channels, num_samples]

    # Create .ravenmodel configuration
    ravenmodel_config = {
        "name": model_name,
        "engine": "PYTORCH",
        "modelDirectory": {
            "path": model_name
        },
        "labelsFilePath": {
            "path": f"{model_name}/labels.txt"
        },
        "numLabels": num_classes,
        "availableSignatures": ["default"],
        "chosenSignature": "default",
        "availableInputs": ["audio"],
        "availableOutputs": ["scores"],
        "inputTensors": {
            "audio": {
                "dimensions": input_dims,
                "dataType": "FLOAT32",
                "audio": True,
                "value": None,
                "file": None
            }
        },
        "outputTensors": {
            "scores": {
                "dimensions": [-1, num_classes],
                "dataType": "FLOAT32",
                "audio": False,
                "value": None,
                "file": None
            }
        },
        "scoresTensor": "scores",
        "minSampleDuration": int(seq_length),
        "maxSampleDuration": int(seq_length),
        "minSampleRate": sample_rate,
        "maxSampleRate": sample_rate
    }

    # Write .ravenmodel file
    ravenmodel_path = output_dir / f"{model_name}.ravenmodel"
    with open(ravenmodel_path, 'w') as f:
        json.dump(ravenmodel_config, f, indent=2)
    print(f"Created .ravenmodel file at {ravenmodel_path}")

    print(f"\nRaven model package created successfully!")
    print(f"  Model directory: {model_dir}")
    print(f"  Configuration: {ravenmodel_path}")
    print(f"  Labels: {label_names}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {seq_length}s ({num_samples} samples)")

    return ravenmodel_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Export soundbay model to Raven Intelligence format'
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to the soundbay checkpoint (.pth file)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for the model package'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Model name (defaults to checkpoint directory name)'
    )
    parser.add_argument(
        '--no-softmax',
        action='store_true',
        help='Do not apply softmax to outputs'
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    model_name = args.name or checkpoint_path.parent.name

    create_raven_model_package(
        checkpoint_path=checkpoint_path,
        output_dir=args.output_dir,
        model_name=model_name,
        apply_softmax=not args.no_softmax,
    )
