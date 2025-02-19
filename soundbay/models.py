import importlib
from typing import Union

import torch
import torchaudio
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import ResNet, BasicBlock, conv3x3, Bottleneck
from torchvision.models.vgg import VGG
from torchvision.models import squeezenet, ResNet18_Weights
import torchvision.models as models

from soundbay.utils.files_handler import load_config
from transformers import AutoProcessor, ASTModel



class ResNet1Channel(ResNet):
    """ resnet model for 1 channel ("grayscale") """
    def __init__(self, block, *args, **kwargs):
        """
        initializes the block as a class instance
        Input:
            block: class of resnet block
            *args: the arguments of the parent class - layers, num_classes, etc.
        """
        _IN_PLANES = 64  # use 64 instead of self.inplanes, there is a nasty assignment in the original code to 64
        block = self._choose_block_class(block)
        super().__init__(block, *args, **kwargs)
        self.conv1 = nn.Conv2d(1, _IN_PLANES, kernel_size=7, stride=2, padding=3,
                               bias=False)

    @staticmethod
    def _choose_block_class(block):
        class_name = block.split('.')[-1]
        module_name = '.'.join(block.split('.')[:-1])
        return getattr(importlib.import_module(module_name), class_name)

    def freeze_layers(self):
        """
        Freeze all layers except the classifier and last layer block from training, as a condition for finetune
        """

        for param in self.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True
        for param in self.layer4.parameters():
            param.requires_grad = True


class AST(nn.Module):
    def __init__(self, weight_path, num_classes):
        super(AST, self).__init__()
        self.processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", max_length=1024)

        self.model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        state_dict = torch.load(weight_path)

        self.model.load_state_dict(state_dict)
        self.num_classes = num_classes

        self.fc = nn.Linear(self.model.config.hidden_size, self.num_classes)
        
    def get_embedding(self, x):
        import ipdb;
        sampling_rate = 16000 #TODO: add sample rate from config --> only takes this!  
        mel_spectogram = self.processor(x.squeeze(1).cpu().numpy(), sampling_rate = sampling_rate, return_tensors="pt")
        
        inputs = {key: val.to(x.device) for key, val in mel_spectogram.items()}  # Move to device

        embedding  = self.model(**inputs) # TODO: if we want to freeze...
        return embedding

    def forward(self, x):
        # Assuming x is the input that needs to be processed before passing to the model
        embedding = self.get_embedding(x)

        output = self.fc(embedding.pooler_output)

        return output


class SqueezeNet1D(squeezenet.SqueezeNet):

    def __init__(
        self,
        version: str = '1_1',
        num_classes: int = 2
    ) -> None:
        super(SqueezeNet1D, self).__init__(version, num_classes)
        sequential_list = list(self.features.children())
        if version == '1_0':
            sequential_list[0] = nn.Conv2d(1, 96, kernel_size=7, stride=2)
        elif version == '1_1':
            sequential_list[0] = nn.Conv2d(1, 64, kernel_size=3, stride=2)
        else:
            raise ValueError(f'Unknown SqueezeNet version: {version}, expected 1_0 or 1_1')
        self.features = nn.Sequential(*sequential_list)


class BottleneckDropout(Bottleneck):
    
    def __init__(self, *args, **kwargs):
        super(BottleneckDropout, self).__init__(*args, **kwargs)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)
        self.dropout4 = nn.Dropout(p=0.5)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.dropout1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout3(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.dropout4(out)

        return out


class VGG1Channel(VGG):
    """VGG model for 1 channel """
    def __init__(self, features, num_classes=2, init_weights=True):
        super().__init__(features, num_classes=num_classes, init_weights=init_weights)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG19Jasco(VGG1Channel):
    def __init__(self, num_classes=2, init_weights=True):
        super().__init__(features=make_layers(
            [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
            False),
            num_classes=num_classes, init_weights=init_weights)


class GoogleResNet50(ResNet1Channel):
    def __init__(self, *args, **kwargs):
        super().__init__('torchvision.models.resnet.Bottleneck', *args, layers=[3, 4, 6, 3], **kwargs)


class OrcaLabResNet18(ResNet1Channel):
    '''
    Resnet18 except without max-pool after the first residual layer (first module)
    '''

    def __init__(self, *args, **kwargs):
        super().__init__('torchvision.models.resnet.BasicBlock', *args, layers=[2, 2, 2, 2], **kwargs)
        layer1 = list(self.layer1.children())[:-1]
        self.layer1 = nn.Sequential(*layer1)


class ChristophCNN(nn.Module):
    '''
    CNN based on https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network by Christoph.
    Optimized with dedicated preprocessing and augmentations. Repo had no pytorch implementation so I used:
    https://missinglink.ai/guides/neural-network-concepts/convolutional-neural-network-build-one-keras-pytorch/ for help
    Hyperparameter Optimization was conducted using GridSearchCV's default 3-Fold Cross Validation to determine an
    optimum combination of hyperparameters for the CNN.
    '''

    def __init__(self, num_classes=2):
        super(ChristophCNN, self).__init__()
        self.drop_out = nn.Dropout(0.2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 15, kernel_size=7, stride=1),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(15, 30, kernel_size=7, stride=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.fc1 = nn.Linear(11 * 11 * 30, 200)
        # This deviates from the cchinchristopherj repo because dense has num_classes outputs instead of 1
        self.fc2 = nn.Linear(200, num_classes)
        self.drop_out = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.drop_out(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out


class ConvBlock2d(nn.Module):
    '''
    Block template usually used for classification. composed of 2dconv, normalization layer and
    activation.
    '''

    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd
        )
        self.activ = nn.LeakyReLU(0.1, inplace=True)
        self.norm = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activ(x)
        return x


class GenericClassifier(nn.Module):
    '''
    Structure of generic and basic CNN classifier. composed of stacked conv layers and fc at the end.
    For our use you'll probably need Resnet - no need to implement by yourself!
    Available at https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    https://pytorch.org/hub/pytorch_vision_resnet/
    '''

    def __init__(self,
                 kernels=(5, 5, 5, 5, 5, 5, 5, 5, 5, 1),
                 strides=(2, 2, 2, 2, 2, 2, 2, 2, 2, 1),
                 in_channels=(1, 128, 128, 128, 256, 256, 256, 512, 512, 512),
                 out_channels=(128, 128, 128, 256, 256, 256, 512, 512, 512, 1024),
                 num_classes=2,
                 features_in=2048,
                 ):
        super(GenericClassifier, self).__init__()
        self.conv_layers = torch.nn.ModuleList()
        for i in range(0, len(kernels)):
            conv_layer = [
                ConvBlock2d(in_channels[i], out_channels[i], kernels[i], padd=(kernels[i] - 1) // 2, stride=strides[i])
            ]
            self.conv_layers += conv_layer
        self.fc = nn.Linear(features_in, num_classes)

    def forward(self, x):

        for it, layer in enumerate(self.conv_layers):
            x = layer(x)
        x = x.view(x.size()[0], -1)

        return self.fc(x)


class ChristophCNNwithPCEN(ChristophCNN):
    '''
    same as ChristophCNN with first PCEN layer
    '''

    def __init__(self, num_classes=2, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, trainable=True):
        super().__init__(num_classes=num_classes)
        self.pcen_model = PCENTransform(eps, s, alpha, delta, r, trainable)

    def forward(self, x):
        out = self.pcen_model(x)
        out = super().forward(out)
        return out


class GoogleResNet50withPCEN(GoogleResNet50):
    '''
    same as GoogleResNet50 with first (non-trainable) PCEN layer
    '''

    def __init__(self, num_classes=2, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, trainable=False):
        super().__init__(num_classes=num_classes)
        self.pcen_model = PCENTransform(eps, s, alpha, delta, r, trainable)

    def forward(self, x):
        out = self.pcen_model(x)
        out = super().forward(out)
        return out


class PCENTransform(nn.Module):
    '''PCEN transform layer for learned parameters - a layer that inherits from nn.Module
    incorporated as a first layer of a module usually
    [s,alpha,delta, r] are learned parameters from:
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/8
    a75d472dc7286653a5245a80a7603a1db308af0.pdf'''
    def __init__(self, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, trainable=True):
        super().__init__()
        if trainable:
            self.log_s = nn.Parameter(torch.log(torch.Tensor([s])))
            self.log_alpha = nn.Parameter(torch.log(torch.Tensor([alpha])))
            self.log_delta = nn.Parameter(torch.log(torch.Tensor([delta])))
            self.log_r = nn.Parameter(torch.log(torch.Tensor([r])))
        else:
            self.s = s
            self.alpha = alpha
            self.delta = delta
            self.r = r
        self.eps = eps
        self.trainable = trainable

    @staticmethod
    def pcen(x, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, training=False):
        frames = x.split(1, -2)
        m_frames = []
        last_state = None
        for frame in frames:
            if last_state is None:
                last_state = s * frame
                m_frames.append(last_state)
                continue
            if training:
                m_frame = ((1 - s) * last_state).add_(s * frame)
            else:
                m_frame = (1 - s) * last_state + s * frame
            last_state = m_frame
            m_frames.append(m_frame)
        M = torch.cat(m_frames, 1)
        if training:
            pcen_ = (x / (M + eps).pow(alpha) + delta).pow(r) - delta ** r
        else:
            pcen_ = x.div_(M.add_(eps).pow_(alpha)).add_(delta).pow_(r).sub_(delta ** r)
        return pcen_

    def forward(self, x):
        x = x.permute((0, 1, 3, 2)).squeeze(dim=1)
        if self.trainable:
            x = PCENTransform.pcen(x, self.eps, torch.exp(self.log_s), torch.exp(self.log_alpha), torch.exp(self.log_delta), torch.exp(self.log_r), self.training and self.trainable)
        else:
            x = PCENTransform.pcen(x, self.eps, self.s, self.alpha, self.delta, self.r, self.training and self.trainable)
        x = x.unsqueeze(dim=1).permute((0, 1, 3, 2))
        return x


class Squeezenet2D(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(Squeezenet2D, self).__init__()

        self.squeezenet = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0',
                                         pretrained=pretrained)
        # number of features from existing squeezenet
        num_features = self.squeezenet.classifier[1].out_channels  # ==1000
        # extra classifier layer
        self.custom_classifier = nn.Sequential(
            nn.Linear(num_features, 256),  # Add a fully connected layer with 256 output units
            nn.ReLU(),  # Add a ReLU activation
            nn.Dropout(0.5),  # Add a dropout layer for regularization
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        rep = self.squeezenet(x)
        out = self.custom_classifier(rep)
        return out


class ResNet182D(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet182D, self).__init__()

        # Load a pre-trained ResNet-18
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT) if pretrained else models.resnet18(weights=None)

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


class EfficientNet2D(nn.Module):
    """EfficientNet model for 3 channel ("RGB") input."""

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        dropout=0.5,
        hidden_dim=256,
        version="b7",
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

    def forward(self, x):
        # Repeat channel to convert 1-channel to 3-channel input
        x = x.repeat(1, 3, 1, 1)
        return self.efficientnet(x)
 

class WAV2VEC2(nn.Module):
    def __init__(
            self,
            num_classes: int = 2,
            config: Union[str, dict] = torchaudio.pipelines.WAV2VEC2_BASE._params,
            path: str = f'https://download.pytorch.org/torchaudio/models/{torchaudio.pipelines.WAV2VEC2_BASE._path}',
            pretrained: bool = True,
            freeze_encoder: bool = False
    ):
        super(WAV2VEC2, self).__init__()
        if isinstance(config, str):
            config = load_config(config)
        config['aux_num_out'] = config.get('aux_num_out', None)
        embedding_dim = config['encoder_embed_dim']

        self.freeze_encoder = freeze_encoder
        self.wav2vec = torchaudio.models.wav2vec2_model(**config)
        if pretrained:
            # Load a pre-trained WAV2VEC2
            self.wav2vec.load_state_dict(torch.hub.load_state_dict_from_url(path))
        self.fc = nn.Linear(in_features=embedding_dim, out_features=num_classes)

    def forward(self, x):
        x = self.extract_features(x)
        return self.fc(x)

    def extract_features(self, x):
        # this is separated from forward to allow feature extraction.
        # sometimes for a batch of samples our raw input is [batch, 1, time]
        if len(x.shape) > 2:
            x = torch.squeeze(x, dim=1)
        x = self.wav2vec.extract_features(x)[0]
        # mean pooling over the layers and time: [layers, batch, time, features] -> [batch, features]
        x = torch.stack(x, dim=0).mean(dim=(0,2))
        return x


    def freeze_layers(self, ):
        # to avoid overfitting the feature extractor is frozen
        self.wav2vec.feature_extractor.requires_grad_(False)
        # it is possible to freeze the encoder as well
        # note that extract_features is using the encoder
        if self.freeze_encoder:
            for param in self.wav2vec.encoder.parameters():
                param.requires_grad = False
