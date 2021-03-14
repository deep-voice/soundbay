import sys
sys.path.append('../src')
from hydra.experimental import compose, initialize
import pytest
from random import randint
from random import seed
from data import ClassifierDataset
import numpy as np


def test_dataloader() -> None:
    seed(1)
    with initialize(config_path="../src/conf"):
        # config is relative to a module
        cfg = compose(config_name="runs/main")
        data_loader = ClassifierDataset(cfg.data.train_dataset.data_path, cfg.data.train_dataset.metadata_path,
                                        augmentations=cfg.data.train_dataset.augmentations,
                                        augmentations_p=cfg.data.train_dataset.augmentations_p,
                                        preprocessors=cfg.data.train_dataset.preprocessors)
        assert data_loader.metadata.shape[1] == 5  # make sure metadata has 5 columns
        assert data_loader.metadata.shape[0] > 0  # make sure metadata is not empty
        data_size = data_loader.metadata.shape[0]
        value = randint(0, data_size)
        sample = data_loader[value]
        assert np.issubdtype(sample[1], np.integer)
        if 'spectrogram' in cfg.data.train_dataset.preprocessors:
            assert len(sample[0].shape) == 3
            if 'utils.LibrosaMelSpectrogram' in cfg.data.train_dataset.preprocessors.spectrogram._target_:
                assert sample[0].shape[1] == cfg.data.train_dataset.preprocessors.spectrogram.n_mels
            else:
                assert sample[0].shape[1] == (cfg.data.train_dataset.preprocessors.spectrogram.n_fft // 2 + 1)
        else:
            assert sample[0].shape[1] == 1
