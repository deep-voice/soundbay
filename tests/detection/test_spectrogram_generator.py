import tempfile
import os
import numpy as np
import soundfile as sf
from soundbay.detection.spectrogram_generator import generate_spectrograms


def test_generate_spectrograms_creates_correct_number_of_chunks():
    """A 45s audio file with 15s chunks and 50% overlap should produce 5 chunks."""
    sr = 8000
    duration = 45.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "test.wav")
        sf.write(wav_path, audio, sr)

        output_dir = os.path.join(tmpdir, "output")
        paths = generate_spectrograms(
            wav_path=wav_path,
            output_dir=output_dir,
            chunk_duration=15.0,
            overlap=0.5,
            target_sr=8000,
            n_fft=1024,
            hop_length=128,
            freq_min=50.0,
            freq_max=4000.0,
            img_size=640,
        )

        # 45s with 15s chunks, 7.5s hop → chunks at 0, 7.5, 15, 22.5, 30 = 5 chunks
        assert len(paths) == 5
        for p in paths:
            assert os.path.exists(p)
            assert p.endswith(".png")


def test_generate_spectrograms_image_dimensions():
    """Output PNG should be img_size x img_size."""
    from PIL import Image

    sr = 8000
    audio = np.random.randn(sr * 15).astype(np.float32) * 0.1

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "test.wav")
        sf.write(wav_path, audio, sr)

        output_dir = os.path.join(tmpdir, "output")
        paths = generate_spectrograms(
            wav_path=wav_path,
            output_dir=output_dir,
            chunk_duration=15.0,
            overlap=0.0,
            target_sr=8000,
            n_fft=1024,
            hop_length=128,
            freq_min=50.0,
            freq_max=4000.0,
            img_size=640,
        )

        img = Image.open(paths[0])
        assert img.size == (640, 640)
        img.close()
