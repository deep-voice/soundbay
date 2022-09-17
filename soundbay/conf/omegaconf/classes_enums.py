from enum import Enum
import torch

class scheduler(Enum, str):
    exponent = torch.optim.lr_scheduler.ExponentialLR
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau


class criterion(Enum, str):
    cross_entropy = torch.nn.CrossEntropyLoss
    binary_cross_entropy = torch.nn.BCELoss
