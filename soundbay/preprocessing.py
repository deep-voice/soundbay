import torch
import numpy as np
import random
from torchvision.transforms import Compose, Resize
from torchaudio.transforms import Spectrogram, MelSpectrogram, AmplitudeToDB


class PeakNormalize:
    """Convert array to lay between 0 to 1"""

    def __call__(self, sample):

        return (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)


class MinFreqFiltering:
    """Cut the spectrogram frequency axis to make it start from min_freq
    ***Note: In case a MaxFreqFiltering is implemented, the max_freq should be greater than min_freq***

    input:
        min_freq_filtering - int
        sample_rate - int

    output:
        spectrogram - pytorch tensor (3-D array)
    """

    def __init__(self, min_freq_filtering, sample_rate):
        self.min_freq_filtering = min_freq_filtering
        self.sample_rate = sample_rate

    def edit_spectrogram_axis(self, sample):
        if self.min_freq_filtering > self.sample_rate / 2 or self.min_freq_filtering < 0:
            raise ValueError("min_freq_filtering should be greater than 0, and smaller than sample_rate/2")
        max_freq_in_spectrogram = self.sample_rate / 2
        min_value = sample.size(dim=1) * self.min_freq_filtering / max_freq_in_spectrogram
        min_value = int(np.floor(min_value))
        sample = sample[:, min_value:, :]

        return sample

    def __call__(self, sample):

        return self.edit_spectrogram_axis(sample)


class UnitNormalize:
    """Remove mean and divide by std to normalize samples"""

    def __call__(self, sample):

        return (sample - sample.mean()) / (sample.std() + 1e-8)


class SlidingWindowNormalize:
    """ Based on Sliding window augmentations of
    https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/whale_cnn.py
        Translated to torch
        Has 50/50 chance of activating H sliding window or V sliding window

        Must come after spectrogram and before AmplitudeToDB
    """

    def __init__(self, sr: float, n_fft: int, lower_cutoff: float = 50, norm=True,
                 inner_ratio: float = 0.06, outer_ratio: float = 0.5):
        self.sr = sr
        self.n_fft = n_fft
        self.lower_cutoff = lower_cutoff
        self.norm = norm
        self.inner_ratio = inner_ratio
        self.outer_ratio = outer_ratio

    def spectrogram_norm(self, spect):

        min_f_ind = int((self.lower_cutoff / (self.sr / 2)) * self.n_fft)

        mval, sval = np.mean(spect[min_f_ind:, :]), np.std(spect[min_f_ind:, :])
        fact_ = 1.5
        spect[spect > mval + fact_ * sval] = mval + fact_ * sval
        spect[spect < mval - fact_ * sval] = mval - fact_ * sval
        spect[:min_f_ind, :] = mval

        return spect

    # slidingWindowV Function from: https://github.com/nmkridler/moby2/blob/master/metrics.py
    def slidingWindow(self, torch_spectrogram, dim=0):
        ''' slidingWindow Method
                Enhance the contrast vertically (along frequency dimension) for dim=0 and
                horizontally (along temporal dimension) for dim=1

                Args:
                    torch_spectrogram: 2-D numpy array image
                    dim: dimension to do the sliding window across
                Returns:
                    Q: 2-D numpy array image with vertically-enhanced contrast

        '''
        if dim not in {0, 1}:
            raise ValueError('dim must be 0 or 1')

        spect = torch_spectrogram.cpu().clone().numpy()
        spect_shape = spect.shape
        spect = spect.squeeze()
        if self.norm:
            spect = self.spectrogram_norm(spect)

        # Set up the local mean window
        wInner = np.ones(int(self.inner_ratio * spect.shape[dim]))
        # Set up the overall mean window
        wOuter = np.ones(int(self.outer_ratio * spect.shape[dim]))
        # Remove overall mean and local mean using np.convolve
        for i in range(spect.shape[1-dim]):
            if dim == 0:
                spect[:, i] = spect[:, i] - (
                        np.convolve(spect[:, i], wOuter, 'same') - np.convolve(spect[:, i], wInner, 'same')) / (
                            wOuter.shape[0] - wInner.shape[0])
            elif dim == 1:
                spect[i, :] = spect[i, :] - (
                        np.convolve(spect[i, :], wOuter, 'same') - np.convolve(spect[i, :], wInner, 'same')) / (
                                      wOuter.shape[0] - wInner.shape[0])

        spect[spect < 0] = 0.
        return torch.from_numpy(spect).reshape(spect_shape)

    def __call__(self, x):

        if random.random() < 0.5:
            return self.slidingWindow(x, dim=0)
        else:
            return self.slidingWindow(x, dim=1)


class Preprocessor:
    def __init__(self, audio_representation: str, normalization: str, resize: bool, 
                       size: tuple[int, int], sample_rate: int, min_freq: int, n_fft: int, 
                       hop_length: int, n_mels: int, amplitude_2_db: bool):
        self.audio_processor = self.set_audio_processor(audio_representation, min_freq, n_fft, 
                                                        hop_length, sample_rate, n_mels)
        self.amplitude_2_db = self.set_amplitude_2_db(amplitude_2_db)
        self.normalization = self.set_normalization(normalization)
        self.resize = self.set_resize(resize, size)
        

    def __call__(self, x):
        x = self.audio_processor(x)
        x = self.amplitude_2_db(x)
        x = self.normalization(x)
        x = self.resize(x)
        return x
    
    def __repr__(self):
        return f"Preprocessor({self.audio_representation}, {self.normalization}, {self.resize})"

    def set_audio_processor(self, audio_representation, min_freq, n_fft, hop_length, sample_rate, n_mels):
        if audio_representation == "spectrogram":
            spec =  Spectrogram(n_fft=n_fft, hop_length=hop_length)
            spec_filtering = MinFreqFiltering(min_freq, sample_rate)
            return Compose([spec, spec_filtering])
        elif audio_representation == "mel_spectrogram":
            return MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, f_min=min_freq,
                                  win_length=n_fft, pad_mode='constant', n_mels=n_mels)
        elif audio_representation == "sliding_window":
            spec =  Spectrogram(n_fft=256, hop_length=64)
            return Compose([spec, SlidingWindowNormalize(sr=sample_rate, n_fft=n_fft)])
        else:
            raise ValueError(f"Invalid audio representation: {audio_representation}")

    def set_normalization(self, normalization):
        if normalization == "peak":
            return PeakNormalize()
        elif normalization == "unit":
            return UnitNormalize()
        elif normalization == None:
            return torch.nn.Identity()
        else:
            raise ValueError(f"Invalid normalization: {normalization}")
        
    def set_resize(self, resize, size):
        if resize:
            return Resize(size)
        else:
            return torch.nn.Identity()
        
    def set_amplitude_2_db(self, amplitude_2_db):
        if amplitude_2_db:
            return AmplitudeToDB()
        else:
            return torch.nn.Identity()