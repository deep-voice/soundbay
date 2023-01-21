'''
Configuration dicts
-------
These dicts describe the allowed values of the soundbay framework
'''
from soundbay.models import ResNet1Channel, GoogleResNet50withPCEN
from soundbay.data import ClassifierDataset, BaseDataset
import torch
from audiomentations import PitchShift, BandStopFilter, TimeMask, TimeStretch



models_dict = {'models.ResNet1Channel':ResNet1Channel,
'models.GoogleResNet50withPCEN': GoogleResNet50withPCEN,
'models.ChristophCNN':ChristophCNN}


datasets_dict = {'soundbay.data.ClassifierDataset':ClassifierDataset}


optim_dict = {'torch.optim.Adam':torch.optim.Adam}



scheduler_dict = {'torch.optim.lr_scheduler.ExponentialLR':torch.optim.lr_scheduler.ExponentialLR}


criterion_dict = {'torch.nn.CrossEntropyLoss':torch.nn.CrossEntropyLoss(),
                'torch.nn.MSELoss':torch.nn.MSELoss()}


augmentations_dict = {'freq_shift': PitchShift,
                        'frequency_masking': BandStopFilter,
                        'time_masking': TimeMask,
                        'time_stretce': TimeStretch}
