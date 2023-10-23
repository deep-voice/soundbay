"""soundbay additional custom audio augmentations, based on https://github.com/iver56/audiomentations"""
import os
import warnings
from functools import lru_cache

from audiomentations import AddBackgroundNoise
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    calculate_desired_noise_rms,
    calculate_rms,
    convert_decibels_to_amplitude_ratio
)


import numpy as np
import librosa
from scipy.interpolate import interp1d


class AtmosphericFilter(BaseWaveformTransform):
    """Apply an atmospheric filter to the audio."""

    supports_multichannel = True

    def __init__(
        self,
        n_fft=128,
        p=0.5,
    ):

        super().__init__(p)
        self.n_fft = n_fft

    @staticmethod
    @lru_cache(maxsize=1)
    def load_filter():
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        path = os.path.join(base_path, r'quad\filter1.npy')
        filter_data = np.load(
            path, allow_pickle=True
        ).item()
        return filter_data

    def time_domain_augmentation(self, audio_data, sample_rate):
        filter_data = self.load_filter()

        audio_data = audio_data.astype(np.float32)

        n_fft = self.n_fft
        # Compute the short-time Fourier transform
        y_pad = librosa.util.fix_length(audio_data, size=len(audio_data) + n_fft // 2)
        Sxx = librosa.stft(y_pad, n_fft=n_fft)

        # Get the frequencies from the STFT
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

        # Interp filter to match the spectrogram (extrapolate if needed)
        filter_interp = interp1d(
            filter_data["freq_vector"][0],
            filter_data["spl_vector"][0],
            kind="linear",
            fill_value="extrapolate",
        )(frequencies)

        spec_shape = np.shape(Sxx)

        linear_vector = 10 ** (filter_interp / 20)

        filter_interp = np.tile(linear_vector, (spec_shape[1], 1)).T

        Sxx = Sxx * filter_interp

        # Convert back to time domain
        time_domain_audio = librosa.istft(Sxx, length=len(audio_data))

        return time_domain_audio

    def apply(self, samples: np.array, sample_rate: int):
        return self.time_domain_augmentation(samples, sample_rate)


class AddMultichannelBackgroundNoise(AddBackgroundNoise):
    """A version of AddBackgroundNoise that can handle multiple channel noise files.

    Note that the noise file is loaded as mono - all channels are mixed together.
    """

    supports_multichannel = True

    def apply(self, samples: np.array, sample_rate: int):
        noise_sound, _ = self._load_sound(
            self.parameters["noise_file_path"], sample_rate
        )
        noise_sound = noise_sound[
            self.parameters["noise_start_index"] : self.parameters["noise_end_index"]
        ]

        if self.noise_transform:
            noise_sound = self.noise_transform(noise_sound, sample_rate)

        noise_rms = calculate_rms(noise_sound)
        if noise_rms < 1e-9:
            warnings.warn(
                "The file {} is too silent to be added as noise. Returning the input"
                " unchanged.".format(self.parameters["noise_file_path"])
            )
            return samples

        clean_rms = calculate_rms(samples)

        if self.noise_rms == "relative":
            desired_noise_rms = calculate_desired_noise_rms(
                clean_rms, self.parameters["snr_in_db"]
            )

            # Adjust the noise to match the desired noise RMS
            noise_sound = noise_sound * (desired_noise_rms / noise_rms)

        if self.noise_rms == "absolute":
            desired_noise_rms_db = self.parameters["rms_in_db"]
            desired_noise_rms_amp = convert_decibels_to_amplitude_ratio(
                desired_noise_rms_db
            )
            gain = desired_noise_rms_amp / noise_rms
            noise_sound = noise_sound * gain

        if len(samples.shape) > 1:
            num_samples = samples.shape[1]  # Audiomentations uses (channels, samples) for non mono sound
            n_channels = samples.shape[0]
        else:
            num_samples = samples
            n_channels = 1

        # Repeat the sound if it shorter than the input sound
        while len(noise_sound) < num_samples:
            noise_sound = np.concatenate((noise_sound, noise_sound))

        if len(noise_sound) > num_samples:
            noise_sound = noise_sound[0:num_samples]

        # Return a mix of the input sound and the background noise sound
        return samples + np.tile(noise_sound, (n_channels, 1))
