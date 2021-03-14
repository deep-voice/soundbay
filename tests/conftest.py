import pytest
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(20, 2)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, x):
        return self.softmax(self.linear1(x))


@pytest.fixture
def model():
    return Model()


@pytest.fixture
def optimizer(model):
    return torch.optim.Adam(model.parameters())


@pytest.fixture
def scheduler(optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=0.98)


@pytest.fixture
def data_loader():
    inputs = torch.randn(20, 20)
    targets = torch.randint(0, 2, (20,)).long()
    batch = [(inputs, targets)]
    return batch


@pytest.fixture
def criterion():
    return torch.nn.CrossEntropyLoss()