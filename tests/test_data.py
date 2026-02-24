import pathlib
import os

from soundbay.config import create_training_config
from soundbay.preprocessing import Preprocessor
from soundbay.augmentations import Augmentor
from soundbay.data import ClassifierDataset
from random import seed, randint
import numpy as np


def test_dataloader() -> None:
    seed(1)
    # Load config from main.yaml relative to this test file
    config_path = pathlib.Path(__file__).parent.parent / 'soundbay' / 'conf' / 'runs' / 'main.yaml'
    cfg = create_training_config(config_path=str(config_path))
    
    # Create valid dummy preprocessor/augmentor for testing
    preprocessor = Preprocessor(
        audio_representation=cfg.data.audio_representation,
        normalization=cfg.data.normalization,
        resize=cfg.data.resize,
        size=cfg.data.size,
        sample_rate=cfg.data.sample_rate,
        min_freq=cfg.data.min_freq,
        n_fft=cfg.data.n_fft,
        hop_length=cfg.data.hop_length,
        n_mels=cfg.data.n_mels
    )

    augmentor = Augmentor(
        pitch_shift_p=cfg.data.train_dataset.augmentations_config.pitch_shift_p,
        time_stretch_p=cfg.data.train_dataset.augmentations_config.time_stretch_p,
        time_masking_p=cfg.data.train_dataset.augmentations_config.time_masking_p,
        frequency_masking_p=cfg.data.train_dataset.augmentations_config.frequency_masking_p,
        min_semitones=cfg.data.train_dataset.augmentations_config.min_semitones,
        max_semitones=cfg.data.train_dataset.augmentations_config.max_semitones,
        min_rate=cfg.data.train_dataset.augmentations_config.min_rate,
        max_rate=cfg.data.train_dataset.augmentations_config.max_rate,
        min_band_part=cfg.data.train_dataset.augmentations_config.min_band_part,
        max_band_part=cfg.data.train_dataset.augmentations_config.max_band_part,
        min_bandwidth_fraction=cfg.data.train_dataset.augmentations_config.min_bandwidth_fraction,
        max_bandwidth_fraction=cfg.data.train_dataset.augmentations_config.max_bandwidth_fraction,
        add_multichannel_background_noise_p=cfg.data.train_dataset.augmentations_config.add_multichannel_background_noise_p,
        min_snr_in_db=cfg.data.train_dataset.augmentations_config.min_snr_in_db,
        max_snr_in_db=cfg.data.train_dataset.augmentations_config.max_snr_in_db,
        lru_cache_size=cfg.data.train_dataset.augmentations_config.lru_cache_size,
        sounds_path=cfg.data.train_dataset.augmentations_config.sounds_path,
        min_center_freq=cfg.data.min_freq
    )

    dataset = ClassifierDataset(
        data_path=cfg.data.train_dataset.data_path,
        metadata_path=cfg.data.train_dataset.metadata_path,
        augmentor=augmentor,
        augmentations_p=cfg.data.train_dataset.augmentations_p,
        preprocessor=preprocessor,
        label_type=cfg.data.label_type
    )

    assert dataset.metadata.shape[1] in (5, 6)  # make sure metadata has 5/6 columns (account for channel)
    assert dataset.metadata.shape[0] > 0  # make sure metadata is not empty
    data_size = dataset.metadata.shape[0]
    value = randint(0, data_size - 1) # Fix potential off-by-one error in original test
    sample = dataset[value]
    assert np.issubdtype(sample[1], np.integer)
    
    if cfg.data.audio_representation == 'spectrogram':
        assert len(sample[0].shape) == 3
        # Preprocessor returns spectrograms, shape depends on n_fft or n_mels if mel_spectrogram
        # But here we default to spectrogram
        assert sample[0].shape[1] == (cfg.data.n_fft // 2 + 1)
    else:
        # if raw waveform
        assert sample[0].shape[1] == 1
