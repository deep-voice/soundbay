import pathlib
import os

from hydra import compose, initialize
from random import randint
from random import seed
from soundbay.data import ClassifierDataset
import numpy as np


def test_dataloader() -> None:
    seed(1)
    with initialize(config_path=os.path.join("..", 'soundbay', 'conf/runs/'), version_base='1.2'):
        # config is relative to a module
        cfg = compose(config_name="main")
        dataset = ClassifierDataset(cfg.data.train_dataset.data_path, cfg.data.train_dataset.metadata_path,
                                        augmentations=cfg.data.train_dataset.augmentations,
                                        augmentations_p=cfg.data.train_dataset.augmentations_p,
                                        preprocessors=cfg.data.train_dataset.preprocessors)
        assert dataset.metadata.shape[1] in (5, 6)  # make sure metadata has 5/6 columns (account for channel)
        assert dataset.metadata.shape[0] > 0  # make sure metadata is not empty
        data_size = dataset.metadata.shape[0]
        value = randint(0, data_size)
        sample = dataset[value]
        assert np.issubdtype(sample[1], np.integer)
        if 'spectrogram' in cfg.data.train_dataset.preprocessors:
            assert len(sample[0].shape) == 3
            if 'utils.LibrosaMelSpectrogram' in cfg.data.train_dataset.preprocessors.spectrogram._target_:
                assert sample[0].shape[1] == cfg.data.train_dataset.preprocessors.spectrogram.n_mels
            else:
                assert sample[0].shape[1] == (cfg.data.train_dataset.preprocessors.spectrogram.n_fft // 2 + 1)
        else:
            assert sample[0].shape[1] == 1
