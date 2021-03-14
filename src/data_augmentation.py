from typing import List

import torch
from torchaudio import sox_effects
import torchaudio

import random
import numpy as np


class RandomAugmenter:
    def __init__(self, p):
        self.p = p

    def __call__(self, x_orig: torch.Tensor) -> torch.Tensor:
        x = x_orig.clone()
        if self.p < random.random():
            return x
        return self.apply_augmentation(x)

    def apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AddGaussianNoise(RandomAugmenter):
    """
    An augmenter that adds random gaussian noise to the given signal.
    """

    def __init__(self, p: float = 0.5, sr: int = 16000, snr: float = 15):
        super().__init__(p)
        self.snr = snr
        self.sr = sr

    def apply_augmentation(self, x):
        self.x_size = x.shape
        gaussian_noise = self.generate_gaussian_noise()
        x_std_amp = torch.std(x)
        x_augmented = x.add(gaussian_noise * x_std_amp)
        return x_augmented

    def generate_gaussian_noise(self):
        gaussian_noise_vec = torch.randn(self.x_size)
        noise_std_amp = torch.std(gaussian_noise_vec)
        gaussian_noise_w_snr_applied = gaussian_noise_vec / noise_std_amp * 10 ** (-self.snr / 20)
        return gaussian_noise_w_snr_applied


class TemporalMasking(RandomAugmenter):
    """
    An augmenter that works by choosing a random time frame in the given tensor and masking it.
    At the moment, masking is done by replacing with zeros.
    """

    def __init__(self, p: float = 0.5, sr: int = 16000, max_secs: float = 0.2, mask_value: int = 0):
        super().__init__(p)
        self.sr = sr
        self.max_secs = max_secs
        self.mask_value = mask_value

    def apply_augmentation(self, x):
        x_size = x.size(-1)
        max_mask_length = min(self.max_secs * self.sr, x_size)
        actual_mask_length = np.random.randint(max_mask_length)
        mask_start = np.random.randint(x_size - actual_mask_length)
        mask_end = mask_start + actual_mask_length
        x[:, mask_start:mask_end] = self.mask_value
        return x


class FrequencyMasking(RandomAugmenter):
    """
    An augmenter that works by choosing a random frequency range and masking it.
    At the moment, masking is done by filtering out the selected frequency range.
    """

    def __init__(self, p: float = 0.5, sr: int = 16000, min_freq_in_segment: int = 500, max_freq_in_segment: int = 1500,
                 attenuation_gain: int = 120, bandwidth_to_filter_out: int = 50):
        super().__init__(p)
        self.sr = sr
        self.min_freq_in_segment = min_freq_in_segment
        self.max_freq_in_segment = max_freq_in_segment
        self.attenuation_gain = attenuation_gain
        self.bandwidth_to_filter_out = bandwidth_to_filter_out

    def apply_augmentation(self, x):
        lower_freq_boundary = random.randint(
            self.min_freq_in_segment, self.max_freq_in_segment - self.bandwidth_to_filter_out)
        higher_freq_boundary = lower_freq_boundary + self.bandwidth_to_filter_out
        effects_list = [['sinc', '-a', f'{self.attenuation_gain}', f'{higher_freq_boundary}-{lower_freq_boundary}']]
        x_augmented = sox_effects.apply_effects_tensor(x, sample_rate=self.sr, effects=effects_list)[0]
        return x_augmented


class ChainedAugmentations(RandomAugmenter):
    def __init__(self, augmentation_list: List[RandomAugmenter], p: float):
        super().__init__(p=p)
        self.augmentation_list = augmentation_list

    def apply_augmentation(self, x):
        for augmentation in self.augmentation_list:
            x = augmentation(x)
        return x
