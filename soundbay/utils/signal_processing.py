import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt



def plot_stft(audio_data, sr):
    fig = plt.figure(figsize=(15,7))
    nfft = 256
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data, n_fft=nfft, hop_length=nfft // 4)), ref=np.max)
    display.specshow(D, sr=sr, fmax=10000)
    plt.colorbar(format='%+2.0f dB')

class LibrosaMelSpectrogram:
    """
    Defining and computing a mel-spectrogram transform using pre-defined parameters
    Input:
        init args:
            sr: sample rate
            n_mels: number of mel frequency bins
            fmin: minimum frequency
        call args:
            sample: audio segment vector
    Output:
        melspectrogram: numpy array with size n_mels*t (signal length)
    """

    def __init__(self, sr, n_mels, fmin):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_mels*20
        self.hop_length = int(sr / n_mels)
        self.fmin = fmin
        self.fmax = sr//2

    def __call__(self,  sample):
        slice_len = 1
        sample_numpy = sample.numpy().ravel() # convert to numpy and flatten
        melspectrogram = librosa.feature.melspectrogram(sample_numpy, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft,
                                                        hop_length=self.hop_length, fmin=self.fmin, fmax=self.fmax)
        return melspectrogram




class LibrosaPcen:
    """
    Defining and computing pcen (Per-Channel Energy Normalization) transform using pre-defined parameters
    Input:
        init args:
            sr: sample rate
            n_mels: number of mel-frequency bins
            fmin: minimum frequency
        call args:
            sample: audio segment vector
    Output:
         pcen: per-channel energy normalized version of signal - torch tensor
    """

    def __init__(self, sr, n_mels, fmin):
        self.sr = sr
        self.n_mels = n_mels
        self.slice_len = 1
        self.hop_length = int(self.sr / (self.n_mels / self.slice_len))
        self.fmin = fmin
        self.fmax = sr//2

    # sample,

    def __call__(self,  sample, gain=0.6, bias=0.1, power=0.2, time_constant=0.4, eps=1e-9):
        hop_length = int(self.sr / (self.n_mels / self.slice_len))
        pcen_librosa = librosa.core.pcen(sample, sr=self.sr, hop_length=self.hop_length, gain=gain, bias=bias, power=power,
                                         time_constant=time_constant, eps=eps)
        pcen_librosa = np.expand_dims(pcen_librosa, 0)  # expand dims to greyscale


        return torch.from_numpy(pcen_librosa).float()
