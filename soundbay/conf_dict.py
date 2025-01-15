'''
Configuration dicts
-------
These dicts describe the allowed values of the soundbay framework
'''

from soundbay.models import ResNet1Channel, GoogleResNet50withPCEN, ChristophCNN, ResNet182D, Squeezenet2D, EfficientNet2D, WAV2VEC2
from soundbay.data import ClassifierDataset, InferenceDataset, NoBackGroundDataset
import torch
from audiomentations import PitchShift, BandStopFilter, TimeMask, TimeStretch

models_dict = {'models.ResNet1Channel': ResNet1Channel,
               'models.GoogleResNet50withPCEN': GoogleResNet50withPCEN,
               'models.ResNet182D': ResNet182D,
               'models.Squeezenet2D': Squeezenet2D,
               'models.ChristophCNN': ChristophCNN,
               'models.EfficientNet2D': EfficientNet2D, 
               'models.WAV2VEC2': WAV2VEC2}

datasets_dict = {'soundbay.data.ClassifierDataset': ClassifierDataset,
                 'soundbay.data.NoBackGroundDataset': NoBackGroundDataset,
                 'soundbay.data.InferenceDataset': InferenceDataset}

optim_dict = {'torch.optim.Adam': torch.optim.Adam, 'torch.optim.SGD': torch.optim.SGD}

scheduler_dict = {'torch.optim.lr_scheduler.ExponentialLR': torch.optim.lr_scheduler.ExponentialLR}

criterion_dict = {'torch.nn.CrossEntropyLoss': torch.nn.CrossEntropyLoss(),
                  'torch.nn.MSELoss': torch.nn.MSELoss(),
                  'torch.nn.BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss(),}

augmentations_dict = {'freq_shift': PitchShift,
                      'frequency_masking': BandStopFilter,
                      'time_masking': TimeMask,
                      'time_stretce': TimeStretch}
