import pytest
import soundfile as sf
from soundbay.data_augmentation import *


# Each test calling this fixture will run one time per each element in the params list
@pytest.fixture(scope="module", params=[AddGaussianNoise(p=1), TemporalMasking(p=1), FrequencyMasking(p=1),
                ChainedAugmentations([AddGaussianNoise(p=1), TemporalMasking(p=1), FrequencyMasking(p=1)], p=1)])
def augmenter(request) -> RandomAugmenter:
    return request.param


def test_augmenter(augmenter):
    with open('assets/data/sample.wav', 'rb') as f:
        wav, _ = sf.read(f, 100000)
    wav = torch.from_numpy(wav.T).float().unsqueeze(0)
    augmented_wav_file = augmenter(wav.clone())
    assert augmented_wav_file is not None
    assert isinstance(augmented_wav_file, torch.Tensor)
    assert not torch.all(wav.eq(augmented_wav_file))
