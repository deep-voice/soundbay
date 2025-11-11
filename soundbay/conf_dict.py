'''
Configuration dicts
-------
These dicts describe the allowed values of the soundbay framework
'''

from soundbay.models import ResNet1Channel, GoogleResNet50withPCEN, ChristophCNN, ResNet182D, Squeezenet2D, EfficientNet2D, WAV2VEC2, AST
from soundbay.data import ClassifierDataset, InferenceDataset, NoBackGroundDataset
import torch
from audiomentations import PitchShift, BandStopFilter, TimeMask, TimeStretch

models_dict = {'ResNet1Channel': ResNet1Channel,
               'GoogleResNet50withPCEN': GoogleResNet50withPCEN,
               'ResNet182D': ResNet182D,
               'Squeezenet2D': Squeezenet2D,
               'ChristophCNN': ChristophCNN,
               'EfficientNet2D': EfficientNet2D, 
               'WAV2VEC2': WAV2VEC2,
               'AST': AST}

datasets_dict = {'ClassifierDataset': ClassifierDataset,
                 'NoBackGroundDataset': NoBackGroundDataset,
                 'InferenceDataset': InferenceDataset}


criterion_dict = {'cross_entropy': torch.nn.CrossEntropyLoss(),
                  'mse': torch.nn.MSELoss(),
                  'bce_with_logits': torch.nn.BCEWithLogitsLoss(),}

augmentations_dict = {'freq_shift': PitchShift,
                      'frequency_masking': BandStopFilter,
                      'time_masking': TimeMask,
                      'time_stretce': TimeStretch}
