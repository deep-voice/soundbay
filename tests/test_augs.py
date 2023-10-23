import wave

import numpy as np
import matplotlib.pyplot as plt
import scipy

from soundbay.custom_audiomentations import AtmosphericFilter


def test_atmospheric_augmentations():
    """Test the atmospheric augmentations."""
    file = r'C:\Users\noam\whale\soundbay\quad\test4.wav'
    with wave.open(file, "rb") as wav_file:
        n_frames = wav_file.getnframes()
        frames = wav_file.readframes(n_frames)
        audio_data = np.frombuffer(frames, dtype=np.int16)
        fs = wav_file.getframerate()

    audio_data = np.tile(audio_data, (2, 1))
    augmented_sig = AtmosphericFilter(n_fft=128, p=1.0)(audio_data, fs)
    augmented_sig = augmented_sig[0]
    audio_data = audio_data[0]

    # original spectrogram
    frequencies, times, Sxx = scipy.signal.spectrogram(audio_data, fs)
    Sxx = 10 * np.log10(Sxx)
    plt.pcolormesh(Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    # augmented spectrogram
    frequencies, times, Sxx = scipy.signal.spectrogram(augmented_sig, fs)
    Sxx = 10 * np.log10(Sxx)
    plt.pcolormesh(Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    print()
