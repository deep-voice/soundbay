import torchvision.models as models
import torch.nn as nn
import torchaudio
import torch

class EfficientNet2D(nn.Module):
    """EfficientNet model for 3 channel ("RGB") input."""

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        dropout=0.5,
        hidden_dim=256,
        version="b7",
        melspec_kwargs: dict = None,
    ):
        super(EfficientNet2D, self).__init__()

        # Map version to corresponding model and weights
        model_map = {
            "b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            "b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            "b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
            "b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
            "b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
            "b5": (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT),
            "b6": (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT),
            "b7": (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
        }

        assert version in model_map, f"Unknown EfficientNet version: {version}, expected one of {list(model_map.keys())}"

        model_fn, weights = model_map[version]
        self.efficientnet = model_fn(weights=weights) if pretrained else model_fn(weights=None)

        # Replace the classification head to output the desired number of classes
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.melspec = torchaudio.transforms.MelSpectrogram(**melspec_kwargs)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
    def forward(self, x):
        x = self.melspec(x)
        x = self.amplitude_to_db(x)
        # Repeat channel to convert 1-channel to 3-channel input
        x = x.repeat(1, 3, 1, 1)
        # peak normalization
        x_min = torch.amin(x, dim=(1, 2, 3))
        x_max = torch.amax(x, dim=(1, 2, 3))
        x = (x - x_min) / (x_max - x_min + 1e-8)
        return self.efficientnet(x)
