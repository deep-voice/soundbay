from enum import Enum
import torch

class scheduler(Enum):
    exponent = torch.optim.lr_scheduler.ExponentialLR
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau


class criterion(Enum):
    cross_entropy = torch.nn.CrossEntropyLoss
    binary_cross_entropy = torch.nn.BCELoss
