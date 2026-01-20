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
