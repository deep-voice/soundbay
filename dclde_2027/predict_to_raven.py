#!/usr/bin/env python3
"""
Run the bioacoustic detector on local WAV files and write Raven selection tables.

Usage (from project root or dclde_2027):
  python dclde_2027/predict_to_raven.py --checkpoint path/to/best.pt --input /path/to/file.wav
  python dclde_2027/predict_to_raven.py --checkpoint path/to/best.pt --input /path/to/folder/
  python dclde_2027/predict_to_raven.py --checkpoint path/to/best.pt --input file1.wav file2.wav --output_dir ./raven_out

Output: For each input WAV, a Raven-compatible .txt file is written (by default
next to the WAV with suffix .Table.1.selections.txt, or in --output_dir).
"""

import argparse
import sys
import tempfile
from pathlib import Path

import torch
import torchaudio

# Allow running from repo root or from dclde_2027
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import Config
from model import (
    BioacousticDetector,
    BioacousticDetectorBEATS,
    load_state_dict_compat,
    get_state_dict_from_checkpoint,
)
from utils import (
    frames_to_raven_table,
    merge_adjacent_detections,
    split_long_detections,
    filter_short_detections,
    save_raven_table,
)


def get_class_names(config):
    """List of class names indexed by class index (from config.label_map)."""
    return [name for name, _ in sorted(config.label_map.items(), key=lambda x: x[1])]


def load_model(checkpoint_path, config):
    """Load model from checkpoint. Returns (model, device)."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if config.model_type == "beats":
        model = BioacousticDetectorBEATS(config).to(device)
    elif config.model_type == "perch":
        model = BioacousticDetector(config).to(device)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict_raw = get_state_dict_from_checkpoint(checkpoint)
    state_dict, strict = load_state_dict_compat(state_dict_raw, model)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model, device


def load_and_resample(wav_path, target_sr):
    """
    Load WAV and resample to target_sr. Returns (waveform, target_sr).
    E.g. 96 kHz files are downsampled to 32 kHz (model's training rate); high frequencies above 16 kHz are lost.
    """
    waveform, sr = torchaudio.load(str(wav_path))
    # Mono: (1, samples) or (samples,) after squeeze
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr


def load_wav_from_bytes(data: bytes, target_sr: int):
    """Load WAV from in-memory bytes and resample to target_sr. Returns (waveform, target_sr), waveform (1, samples)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        return load_and_resample(path, target_sr)
    finally:
        Path(path).unlink(missing_ok=True)


def run_inference_on_waveform(
    waveform, model, device, config,
    threshold=0.5,
    overlap_ratio=0.0,
    max_call_sec=1.75,
    min_duration_by_class=None,
):
    """
    Run model on a single waveform (1, samples) at config.target_sr.
    Uses overlapping windows when overlap_ratio > 0. Detections longer than max_call_sec are split.

    Returns list of detection dicts with 'Begin Time (s)', 'End Time (s)', 'Class'.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    samples_per_window = int(config.window_sec * config.target_sr)
    step_samples = int((1.0 - overlap_ratio) * samples_per_window)
    if step_samples < 1:
        step_samples = 1
    num_samples = waveform.shape[1]
    class_names = get_class_names(config)
    frames_per_sec = config.frames_per_sec
    all_detections = []

    with torch.no_grad():
        start = 0
        while start < num_samples:
            end = min(start + samples_per_window, num_samples)
            chunk = waveform[:, start:end]
            if chunk.shape[1] < samples_per_window:
                pad = torch.zeros(1, samples_per_window - chunk.shape[1], device=chunk.device, dtype=chunk.dtype)
                chunk = torch.cat([chunk, pad], dim=1)
            chunk = chunk.to(device)
            out = model(chunk)
            probs = torch.sigmoid(out.squeeze(0)).cpu()
            window_start_sec = start / config.target_sr
            dets = frames_to_raven_table(
                probs,
                window_start=window_start_sec,
                frames_per_sec=frames_per_sec,
                class_names=class_names,
                threshold=threshold,
            )
            all_detections.extend(dets)
            start += step_samples

    # Merge overlapping/adjacent same-class detections (from overlap or window boundaries)
    merged = merge_adjacent_detections(all_detections, gap_sec=0.5)
    # Split any call longer than max_call_sec into segments of at most max_call_sec
    split = split_long_detections(merged, max_duration_sec=max_call_sec)
    # Drop detections shorter than per-class minimum (e.g. KW < 0.4s)
    return filter_short_detections(split, min_duration_by_class=min_duration_by_class or {})


def run_inference_on_file(
    wav_path, model, device, config,
    threshold=0.5,
    overlap_ratio=0.0,
    max_call_sec=1.75,
    min_duration_by_class=None,
):
    """
    Run model on a single WAV file. Audio is split into windows (optionally overlapping).
    Long detections are split; short ones can be filtered by per-class minimum duration.

    Returns:
        list of detection dicts with 'Begin Time (s)', 'End Time (s)', 'Class'
    """
    waveform, sr = load_and_resample(wav_path, config.target_sr)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return run_inference_on_waveform(
        waveform, model, device, config,
        threshold=threshold,
        overlap_ratio=overlap_ratio,
        max_call_sec=max_call_sec,
        min_duration_by_class=min_duration_by_class,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run bioacoustic detector on WAV files and export Raven selection tables."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt). Default: dclde_2027/checkpoints/best.pt if present.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more WAV files or a single directory (will use all .wav inside).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output .txt files. If not set, save next to each input WAV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold (default: 0.5).",
    )
    parser.add_argument(
        "--config_onnx",
        type=str,
        default=None,
        help="Override config.onnx_path if needed (e.g. for Perch encoder).",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated class names to include in Raven table (e.g. KW or KW,HW). Default: all.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        metavar="RATIO",
        help="Window overlap ratio 0–0.9 (default 0.5 = 50%% overlap). 0 = no overlap.",
    )
    parser.add_argument(
        "--max-call-duration",
        type=float,
        default=1.75,
        metavar="SEC",
        help="Split detections longer than this (seconds) into segments (default 1.75). 0 = no split.",
    )
    parser.add_argument(
        "--min-duration",
        type=str,
        default=None,
        metavar="CLASS:SEC[,CLASS:SEC...]",
        help="Drop detections shorter than this per class (e.g. KW:0.4 or KW:0.4,HW:0.3).",
    )
    args = parser.parse_args()

    # Resolve input paths
    inputs = []
    for p in args.input:
        path = Path(p).expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() in (".wav", ".wave"):
                inputs.append(path)
            else:
                print(f"Skip (not WAV): {path}")
        elif path.is_dir():
            inputs.extend(sorted(path.glob("*.wav")) + sorted(path.glob("*.WAV")))
        else:
            print(f"Not found: {path}")

    if not inputs:
        print("No WAV files to process.")
        sys.exit(1)

    # Checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    else:
        checkpoint_path = SCRIPT_DIR / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    config = Config()
    # Use paths relative to script dir so it works on any machine (e.g. your Mac)
    if not args.config_onnx:
        config.onnx_path = str(SCRIPT_DIR / "perch_v2.onnx")
    else:
        config.onnx_path = args.config_onnx

    classes_filter = None
    if args.classes:
        classes_filter = [c.strip() for c in args.classes.split(",") if c.strip()]
        valid = set(config.label_map.keys())
        invalid = set(classes_filter) - valid
        if invalid:
            print(f"Invalid --classes: {invalid}. Valid: {list(valid)}")
            sys.exit(1)
        print(f"Filtering to classes: {classes_filter}")

    overlap_ratio = max(0.0, min(0.9, args.overlap))
    if overlap_ratio > 0:
        print(f"Using {overlap_ratio*100:.0f}% window overlap.")
    max_call_sec = getattr(args, "max_call_duration", 1.75)
    if max_call_sec and max_call_sec > 0:
        print(f"Splitting calls longer than {max_call_sec}s.")

    min_duration_by_class = {}
    if getattr(args, "min_duration", None):
        for part in args.min_duration.split(","):
            part = part.strip()
            if ":" in part:
                cls, sec = part.split(":", 1)
                cls, sec = cls.strip(), sec.strip()
                try:
                    min_duration_by_class[cls] = float(sec)
                except ValueError:
                    print(f"Invalid --min-duration part: {part!r}")
                    sys.exit(1)
        if min_duration_by_class:
            valid = set(config.label_map.keys())
            invalid = set(min_duration_by_class) - valid
            if invalid:
                print(f"Unknown class in --min-duration: {invalid}. Valid: {list(valid)}")
                sys.exit(1)
            print(f"Min duration per class: {min_duration_by_class}")

    print(f"Loading model from {checkpoint_path} ...")
    model, device = load_model(checkpoint_path, config)
    print(f"Processing {len(inputs)} file(s) (threshold={args.threshold}) ...")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for wav_path in inputs:
        try:
            detections = run_inference_on_file(
                wav_path, model, device, config,
                threshold=args.threshold,
                overlap_ratio=overlap_ratio,
                max_call_sec=max_call_sec if max_call_sec and max_call_sec > 0 else None,
                min_duration_by_class=min_duration_by_class or None,
            )
            if classes_filter:
                detections = [d for d in detections if d["Class"] in classes_filter]
            base = wav_path.stem
            if output_dir:
                out_path = output_dir / f"{base}.Table.1.selections.txt"
            else:
                out_path = wav_path.parent / f"{base}.Table.1.selections.txt"
            file_ref = wav_path.name
            save_raven_table(detections, str(out_path), file_path=file_ref)
            print(f"  {wav_path.name} -> {len(detections)} selections -> {out_path}")
        except Exception as e:
            print(f"  ERROR {wav_path}: {e}")
            raise

    print("Done.")


if __name__ == "__main__":
    main()
