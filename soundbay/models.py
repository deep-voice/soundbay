import importlib
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import ResNet, BasicBlock, conv3x3, Bottleneck
from torchvision.models.vgg import VGG
from torchvision.models import squeezenet
import torch.nn.init as init


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

"""Audio models."""

from slowfast.models import head_helper, resnet_helper, stem_helper
from slowfast.models.build import MODEL_REGISTRY
from slowfast.models.batchnorm_helper import get_norm
import slowfast.utils.weight_init_helper as init_helper

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "fast": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    "slow": [[1, 1]],
    "fast": [[1, 1]],
    "slowfast": [[1, 1], [1, 1]],
}

class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm2d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                 default is nn.BatchNorm2d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv2d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1],
            stride=[alpha, 1],
            padding=[fusion_kernel // 2, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


@MODEL_REGISTRY.register()
class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Evangelos Kazakos, Arsha Nagrani, Andrew Zisserman, Dima Damen.
    "Auditory Slow-Fast Networks for Audio Recognition"

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.AudioModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7], temp_kernel[0][1] + [7]],
            stride=[[2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3],
                [temp_kernel[0][1][0] // 2, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )
        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.FREQUENCY_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.FREQUENCY_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool2d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.FREQUENCY_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.FREQUENCY_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.FREQUENCY_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.FREQUENCY_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.FREQUENCY_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.FREQUENCY_DILATIONS[3],
            norm_module=self.norm_module,
        )

        self.head = head_helper.ResNetBasicHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            num_classes=cfg.MODEL.NUM_CLASSES if len(cfg.MODEL.NUM_CLASSES) > 1 else cfg.MODEL.NUM_CLASSES[0],
            pool_size=[
                [
                    cfg.AUDIO_DATA.NUM_FRAMES
                    // cfg.SLOWFAST.ALPHA // 4
                    // pool_size[0][0],
                    cfg.AUDIO_DATA.NUM_FREQUENCIES // 32 // pool_size[0][1],
                ],
                [
                    cfg.AUDIO_DATA.NUM_FRAMES // 4 // pool_size[1][0],
                    cfg.AUDIO_DATA.NUM_FREQUENCIES // 32 // pool_size[1][1],
                ],
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        x = self.head(x)
        return x

    def freeze_fn(self, freeze_mode):
        if freeze_mode == 'bn_parameters':
            print("Freezing all BN layers\' parameters except the first one.")
            for n, m in self.named_modules():
                if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm)) \
                        and ('s1.pathway0_stem.bn' not in n
                             and 's1.pathway1_stem.bn' not in n
                             and 's1_fuse.bn' not in n):
                    # shutdown parameters update in frozen mode
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
        elif freeze_mode == 'bn_statistics':
            print("Freezing all BN layers\' statistics except the first one.")
            for n, m in self.named_modules():
                if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm))\
                        and ('s1.pathway0_stem.bn' not in n
                             and 's1.pathway1_stem.bn' not in n
                             and 's1_fuse.bn' not in n):
                    # shutdown running statistics update in frozen mode
                    m.eval()


@MODEL_REGISTRY.register()
class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (Slow, Fast).

    Evangelos Kazakos, Arsha Nagrani, Andrew Zisserman, Dima Damen.
    "Auditory Slow-Fast Networks for Audio Recognition"

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.AudioModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7]],
            stride=[[2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.FREQUENCY_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.FREQUENCY_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool2d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.FREQUENCY_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.FREQUENCY_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.FREQUENCY_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.FREQUENCY_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.FREQUENCY_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.FREQUENCY_DILATIONS[3],
            norm_module=self.norm_module,
        )

        self.head = head_helper.ResNetBasicHead(
            dim_in=[width_per_group * 32],
            num_classes=cfg.MODEL.NUM_CLASSES if len(cfg.MODEL.NUM_CLASSES) > 1 else cfg.MODEL.NUM_CLASSES[0],
            pool_size=[
                [
                    cfg.AUDIO_DATA.NUM_FRAMES // 4 // pool_size[0][0],
                    cfg.AUDIO_DATA.NUM_FREQUENCIES // 32 // pool_size[0][1],
                ]
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.head(x)
        return x

    def freeze_fn(self, freeze_mode):
        if freeze_mode == 'bn_parameters':
            print("Freezing all BN layers\' parameters except the first one.")
            for n, m in self.named_modules():
                if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm))\
                        and ('s1.pathway0_stem.bn' not in n
                             and 's1.pathway1_stem.bn' not in n
                             and 's1_fuse.bn' not in n):
                    # shutdown parameters update in frozen mode
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
        elif freeze_mode == 'bn_statistics':
            print("Freezing all BN layers\' statistics except the first one.")
            for n, m in self.named_modules():
                if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm))\
                        and ('s1.pathway0_stem.bn' not in n
                             and 's1.pathway1_stem.bn' not in n
                             and 's1_fuse.bn' not in n):
                    # shutdown running statistics update in frozen mode
                    m.eval()


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
