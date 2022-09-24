from enum import Enum
import torch
from dataclasses import dataclass
from soundbay import data
from torch.utils.data import Dataset
class Scheduler(Enum):
    exponent = torch.optim.lr_scheduler.ExponentialLR
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau


class Criterion(Enum):
    cross_entropy = torch.nn.CrossEntropyLoss
    binary_cross_entropy = torch.nn.BCELoss


class dataset(Enum):
    classifier_dataset = data.ClassifierDataset
    base_dataset = data.BaseDataset 

# @dataclass
# class Dataset2:
#     kind: Dataset=data.ClassifierDataset
#     data_path: str='../tests/assets/data/'
#     mode: str='train'
#     metadata_path: str='../tests/assets/annotations/sample_annotations.csv'
#     augmentations_p: float=0.8