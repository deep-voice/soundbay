"""Generate spectrogram PNG images from WAV files for YOLO training."""
import json
import os
from typing import Dict, List

import librosa
import numpy as np
from PIL import Image


# Spectrogram geometry must match between dataset preparation and inference:
# a model trained on 5s/3000Hz chunks scores poorly when fed 15s/4000Hz ones.
DEFAULT_PARAMS = {
    "chunk_duration": 15.0,
    "overlap": 0.5,
    "target_sr": 8000,
    "n_fft": 1024,
    "hop_length": 128,
    "freq_min": 50.0,
    "freq_max": 4000.0,
    "img_size": 640,
}

PARAM_KEYS = tuple(DEFAULT_PARAMS)

SIDECAR_SUFFIX = ".spectrogram.json"


def sidecar_path_for_model(model_path: str) -> str:
    """Path of the spectrogram-params sidecar that belongs next to a checkpoint."""
    return os.path.splitext(model_path)[0] + SIDECAR_SUFFIX


def load_params_sidecar(model_path: str) -> Dict:
    """Read the spectrogram params recorded beside a checkpoint, if any."""
    path = sidecar_path_for_model(model_path)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def resolve_params(sidecar: Dict = None, overrides: Dict = None) -> Dict:
    """Merge spectrogram params: defaults < sidecar < explicit overrides.

    ``None`` values in ``overrides`` mean "not supplied" (argparse default) and
    are ignored so they cannot wipe a sidecar value.
    """
    resolved = dict(DEFAULT_PARAMS)
    for source in (sidecar or {}, overrides or {}):
        for key, value in source.items():
            if key not in DEFAULT_PARAMS:
                raise ValueError(f"unknown spectrogram param: {key!r}")
            if value is not None:
                resolved[key] = value
    return resolved


def generate_spectrograms(
    wav_path: str,
    output_dir: str,
    chunk_duration: float = 15.0,
    overlap: float = 0.5,
    target_sr: int = 8000,
    n_fft: int = 1024,
    hop_length: int = 128,
    freq_min: float = 50.0,
    freq_max: float = 4000.0,
    img_size: int = 640,
) -> List[str]:
    """Split a WAV file into spectrogram PNG chunks.

    Returns list of output PNG paths. Filenames encode chunk start time
    for later label alignment: {stem}_{start_ms:08d}.png
    """
    os.makedirs(output_dir, exist_ok=True)

    audio, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    total_duration = len(audio) / sr
    hop_seconds = chunk_duration * (1 - overlap)

    # Frequency bin indices for cropping
    freqs = librosa.fft_frequencies(sr=target_sr, n_fft=n_fft)
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)

    stem = os.path.splitext(os.path.basename(wav_path))[0]
    output_paths = []

    chunk_start = 0.0
    while chunk_start + chunk_duration <= total_duration + 0.01:
        start_sample = int(chunk_start * sr)
        end_sample = int((chunk_start + chunk_duration) * sr)
        chunk_audio = audio[start_sample:end_sample]

        # Pad if slightly short
        expected_samples = int(chunk_duration * sr)
        if len(chunk_audio) < expected_samples:
            chunk_audio = np.pad(chunk_audio, (0, expected_samples - len(chunk_audio)))

        # Compute magnitude spectrogram
        S = np.abs(librosa.stft(chunk_audio, n_fft=n_fft, hop_length=hop_length))
        S = S[freq_mask, :]  # Crop to frequency range

        # Normalize: per-chunk 98th percentile
        p98 = np.percentile(S, 98)
        if p98 > 0:
            S = np.clip(S / p98, 0, 1)
        else:
            S = np.zeros_like(S)

        # Flip vertically so high freq is at top (y=0)
        S = S[::-1, :]

        # Convert to uint8 image and resize
        img_array = (S * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")
        img = img.resize((img_size, img_size), Image.BILINEAR)

        # Save
        start_ms = int(chunk_start * 1000)
        filename = f"{stem}_{start_ms:08d}.png"
        out_path = os.path.join(output_dir, filename)
        img.save(out_path)
        output_paths.append(out_path)

        chunk_start += hop_seconds

    return output_paths


def get_chunk_start_from_filename(filename: str) -> float:
    """Extract chunk start time (seconds) from spectrogram filename."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    ms_str = stem.rsplit("_", 1)[-1]
    return int(ms_str) / 1000.0
