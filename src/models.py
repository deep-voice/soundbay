import importlib
import os

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, conv3x3
from torchvision.models.vgg import VGG


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
        self.drop_out = nn.Dropout()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.drop_out = nn.Dropout()
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.drop_out(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
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
                 outputs_num=2,
                 features_in=2048,
                 ):
        super(GenericClassifier, self).__init__()
        self.conv_layers = torch.nn.ModuleList()
        for i in range(0, len(kernels)):
            conv_layer = [
                ConvBlock2d(in_channels[i], out_channels[i], kernels[i], padd=(kernels[i] - 1) // 2, stride=strides[i])
            ]
            self.conv_layers += conv_layer
        self.fc = nn.Linear(features_in, outputs_num)

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
        out = self.pcen_model.forward(x)
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
        out = self.pcen_model.forward(x)
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