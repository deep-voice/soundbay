import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import sys

sys.path.append('../src')
from models import ResNet1Channel, GoogleResNet50, OrcaLabResNet18, ChristophCNN, ChristophCNNwithPCEN, \
    GoogleResNet50withPCEN


@pytest.fixture(scope="module", params=[ResNet1Channel('torchvision.models.resnet.Bottleneck', [3, 4, 6, 3]),
                                        GoogleResNet50(),
                                        OrcaLabResNet18(),
                                        # ChristophCNN(),
                                        # ChristophCNNwithPCEN(),
                                        GoogleResNet50withPCEN()])
def own_model(request):
    return request.param


def test_model(own_model):
    data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                         transforms.ToTensor()])
    dataset = ImageFolder('assets/demi_image_data', transform=data_transform)
    data = DataLoader(dataset, batch_size=16, shuffle=True)
    assert torch.sum(own_model(next(iter(data))[0])).detach().numpy() != 0
