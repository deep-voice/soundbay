from audiomentations import PitchShift, BandStopFilter, TimeMask, TimeStretch
from soundbay.custom_audiomentations import AddMultichannelBackgroundNoise
from typing import Optional


class Augmentor:
    def __init__(self, pitch_shift_p: float, time_stretch_p: float, time_masking_p: float, frequency_masking_p: float,
                       min_semitones: int, max_semitones: int, min_rate: float, max_rate: float, min_band_part: float, 
                       max_band_part: float, min_center_freq: int, min_bandwidth_fraction: float, max_bandwidth_fraction: float,
                       add_multichannel_background_noise_p: float, min_snr_in_db: int, max_snr_in_db: int, 
                       lru_cache_size: int, sounds_path: Optional[str] = None):
        if sounds_path is not None:
            self.add_multichannel_background_noise = AddMultichannelBackgroundNoise(p=add_multichannel_background_noise_p,
                                                    sounds_path=sounds_path,
                                                    min_snr_in_db=min_snr_in_db,
                                                    max_snr_in_db=max_snr_in_db,
                                                    lru_cache_size=lru_cache_size)
        else:
            self.add_multichannel_background_noise = None
        self.pitch_shift = PitchShift(min_semitones=min_semitones, max_semitones=max_semitones, p=pitch_shift_p)
        self.time_stretch = TimeStretch(min_rate=min_rate, max_rate=max_rate, p=time_stretch_p)
        self.time_masking = TimeMask(min_band_part=min_band_part, max_band_part=max_band_part, p=time_masking_p)
        self.frequency_masking = BandStopFilter(min_center_freq=min_center_freq, 
                                                min_bandwidth_fraction=min_bandwidth_fraction, 
                                                max_bandwidth_fraction=max_bandwidth_fraction, 
                                                p=frequency_masking_p)
                                                
    def __call__(self, x, sample_rate: int):
        if self.add_multichannel_background_noise is not None:
            x = self.add_multichannel_background_noise(x, sample_rate)
        x = self.pitch_shift(x, sample_rate)
        x = self.time_stretch(x, sample_rate)
        x = self.time_masking(x, sample_rate)
        x = self.frequency_masking(x, sample_rate)
        return x