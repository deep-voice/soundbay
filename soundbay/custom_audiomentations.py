"""soundbay additional custom audio augmentations, based on https://github.com/iver56/audiomentations"""

import warnings
import numpy as np

from audiomentations import AddBackgroundNoise
from audiomentations.core.utils import (
    calculate_desired_noise_rms,
    calculate_rms,
    convert_decibels_to_amplitude_ratio
)


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
