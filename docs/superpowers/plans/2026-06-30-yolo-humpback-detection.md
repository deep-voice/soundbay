# YOLOv11 Humpback Whale Call Detection — Implementation Plan

> **Status (2026-08-08): Phase 1 (Tasks 1–8) complete and committed** — see `soundbay/detection/`. Checkboxes below marked `[x]` retroactively; this plan was executed but not committed until now. Phase 2 (Task 9, EC2 training) also happened — see `runs/detect/` on the training EC2 instance and the follow-up plan `docs/superpowers/plans/2026-07-21-yolo-postprocessing-filters.md` for the post-training postprocessing work driven by biologist feedback.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a YOLOv11m object detector to locate humpback whale calls as bounding boxes in time-frequency spectrograms, using Mozambique 2021 as primary training data and Costa Rica 2022 as test/domain-adaptation data.

**Architecture:** Adapt the biodcase-2026 champion pipeline (YOLOv11m, spectrogram PNGs, YOLO-format labels) to humpback whale calls. Generate spectrograms from WAV files, convert Raven selection tables to YOLO bounding-box labels, train with spectrogram-safe augmentation, evaluate with IoU in both time and frequency dimensions.

**Tech Stack:** Python 3.10+, ultralytics (YOLOv11), librosa, numpy, pandas, matplotlib, boto3, gdown, wandb

---

## Context & Key Decisions

### Data Inventory

| Dataset | Recordings | Annotated | Role | Location |
|---------|-----------|-----------|------|----------|
| Mozambique 2021 | 96 WAVs (2 tracks each) | 29 have Raven .txt files | Primary train + val | Google Drive → S3 |
| Costa Rica 2022 | 12 WAVs | 3 Done + 2 In Progress | Test + domain adaptation | Google Drive → S3 |

### Architecture Decisions (from biodcase-2026 champion)

- **Model:** `yolo11m.pt` (pretrained ImageNet → fine-tune)
- **Optimizer:** AdamW, lr=0.001 (NOT default SGD/0.01 — diverges on fine-tune)
- **Image size:** 640 (larger than biodcase's 320 because humpback calls span wider freq range)
- **Epochs:** 50 (humpbacks have more varied call types than blue whales)
- **Augmentation:** HSV-value=0.3, translate=0.05, scale=0.1, erasing=0.2, NO flips/rotation/mosaic
- **Spectrogram:** Linear magnitude, per-chunk 98th-percentile normalization, saved as PNG

### Spectrogram Parameters for Humpbacks

Humpback songs span ~80 Hz to ~4000 Hz with units lasting 1–5 seconds. This differs from biodcase's 0–125 Hz blue whales:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sample rate | 8000 Hz (resample) | Nyquist 4000 Hz covers humpback range |
| n_fft | 1024 | ~0.128s window → good freq resolution for tonal calls |
| hop_length | 128 | ~0.016s → good time resolution |
| Chunk duration | 15 seconds | Humpback units are 1–5s; 15s gives 2–3 calls per image |
| Freq range | 50–4000 Hz | Covers all humpback song units |
| Output image | 640×640 PNG | Matches YOLO imgsz, frequency on Y, time on X |

### Data Split Strategy

Given the domain difference (Mozambique = training site, Costa Rica = test site):

1. **Train (80%):** ~23 annotated Mozambique recordings
2. **Validation (20%):** ~6 annotated Mozambique recordings (random split by recording)
3. **Test:** 3 fully annotated Costa Rica recordings (pure out-of-domain eval)
4. **Domain adaptation (optional):** If initial cross-domain performance is poor, move 1–2 Costa Rica "In Progress" recordings to training as fine-tuning data (record which ones)

### Class Mapping

Single class detection: `humpback_call` (class 0). If Raven annotations contain sub-types (song units, social calls, etc.), we collapse them all to one class initially. Multi-class can be a follow-up experiment.

---

## File Structure

```
soundbay/
├── detection/                          # NEW: YOLO detection module
│   ├── __init__.py
│   ├── spectrogram_generator.py        # WAV → spectrogram PNG chunks
│   ├── raven_to_yolo.py                # Raven .txt → YOLO .txt labels
│   ├── prepare_dataset.py              # End-to-end data preparation CLI
│   ├── train.py                        # YOLO training wrapper
│   ├── inference.py                    # Run trained model on new audio
│   └── postprocess.py                  # NMS, temporal merging, Raven export
├── conf/
│   └── detection/
│       ├── humpback_mozambique.yaml    # Dataset YAML for YOLO
│       └── train_humpback.yaml         # Training hyperparameters
├── scripts/
│   ├── download_humpback_data.py       # gdown from Google Drive
│   └── upload_to_s3.py                 # Upload prepared dataset to S3
└── docs/
    └── superpowers/plans/
        └── 2026-06-30-yolo-humpback-detection.md  # This plan
```

**On EC2 (runtime layout):**
```
/data/humpback_yolo/
├── images/
│   ├── train/          # Spectrogram PNGs from Mozambique
│   └── val/            # Spectrogram PNGs from Mozambique (val split)
├── labels/
│   ├── train/          # YOLO .txt files matching images
│   └── val/            # YOLO .txt files matching images
├── test/
│   ├── images/         # Costa Rica spectrograms
│   └── labels/         # Costa Rica YOLO labels
└── humpback.yaml       # YOLO dataset config pointing to above
```

---

## Phase 1: Data Pipeline (Local Development)

### Task 1: Download Data from Google Drive

**Files:**
- Create: `soundbay/scripts/download_humpback_data.py`

- [x] **Step 1: Install gdown in .venv**

```bash
source .venv/Scripts/activate  # Windows
pip install gdown
```

- [x] **Step 2: Write download script**

```python
"""Download humpback whale recordings from Google Drive to local/S3."""
import argparse
import os
import gdown


MOZAMBIQUE_FOLDER_ID = "1245QCyv2twFnVOsHmBkcz0TvDcNUGPpo"
COSTA_RICA_FOLDER_ID = "1PFJuSEC3fQC0uAcv4bMQgdPa5YACS0HP"


def download_folder(folder_id: str, output_dir: str):
    """Download entire Google Drive folder."""
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url, output=output_dir, quiet=False)


def main():
    parser = argparse.ArgumentParser(description="Download humpback data from Google Drive")
    parser.add_argument("--dataset", choices=["mozambique", "costa_rica", "both"], default="both")
    parser.add_argument("--output-dir", default="./data/raw")
    args = parser.parse_args()

    if args.dataset in ("mozambique", "both"):
        print("Downloading Mozambique 2021...")
        download_folder(MOZAMBIQUE_FOLDER_ID, os.path.join(args.output_dir, "mozambique_2021"))

    if args.dataset in ("costa_rica", "both"):
        print("Downloading Costa Rica 2022...")
        download_folder(COSTA_RICA_FOLDER_ID, os.path.join(args.output_dir, "costa_rica_2022"))

    print("Done.")


if __name__ == "__main__":
    main()
```

- [x] **Step 3: Test download of a single file to verify access**

```bash
python scripts/download_humpback_data.py --dataset mozambique --output-dir ./data/raw
```

Expected: WAV files + Raven .txt annotation files download to `./data/raw/mozambique_2021/`

- [x] **Step 4: Commit**

```bash
git add scripts/download_humpback_data.py
git commit -m "feat(detection): add Google Drive download script for humpback data"
```

---

### Task 2: Raven Selection Table → YOLO Label Converter

**Files:**
- Create: `soundbay/detection/__init__.py`
- Create: `soundbay/detection/raven_to_yolo.py`
- Create: `tests/detection/test_raven_to_yolo.py`

- [x] **Step 1: Write the failing test**

```python
# tests/detection/test_raven_to_yolo.py
import tempfile
import os
from soundbay.detection.raven_to_yolo import convert_raven_to_yolo, parse_raven_file


SAMPLE_RAVEN = (
    "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tAnnotation\n"
    "1\tSpectrogram 1\t1\t2.500\t4.800\t150.0\t1200.0\thumpback\n"
    "2\tSpectrogram 1\t1\t8.100\t10.300\t200.0\t3500.0\thumpback\n"
    "3\tSpectrogram 1\t1\t12.000\t13.500\t80.0\t900.0\tunknown\n"
)


def test_parse_raven_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SAMPLE_RAVEN)
        f.flush()
        annotations = parse_raven_file(f.name)

    assert len(annotations) == 3
    assert annotations[0]["begin_time"] == 2.5
    assert annotations[0]["end_time"] == 4.8
    assert annotations[0]["low_freq"] == 150.0
    assert annotations[0]["high_freq"] == 1200.0
    os.unlink(f.name)


def test_convert_raven_to_yolo_single_chunk():
    """A 15s chunk starting at t=0 should capture annotations 1 and 2."""
    annotations = [
        {"begin_time": 2.5, "end_time": 4.8, "low_freq": 150.0, "high_freq": 1200.0, "label": "humpback"},
        {"begin_time": 8.1, "end_time": 10.3, "low_freq": 200.0, "high_freq": 3500.0, "label": "humpback"},
    ]
    chunk_start = 0.0
    chunk_duration = 15.0
    freq_min = 50.0
    freq_max = 4000.0

    yolo_labels = convert_raven_to_yolo(
        annotations, chunk_start, chunk_duration, freq_min, freq_max, class_id=0
    )

    assert len(yolo_labels) == 2
    # First box: x_center = (2.5+4.8)/(2*15) = 0.243, width = 2.3/15 = 0.153
    # y is inverted (high freq = top = y=0): y_center = 1 - (150+1200)/(2*4000-50)... 
    # Just check format and bounds
    for label in yolo_labels:
        parts = label.split()
        assert len(parts) == 5
        assert parts[0] == "0"  # class_id
        x_c, y_c, w, h = [float(p) for p in parts[1:]]
        assert 0 <= x_c <= 1
        assert 0 <= y_c <= 1
        assert 0 < w <= 1
        assert 0 < h <= 1
```

- [x] **Step 2: Run test to verify it fails**

```bash
pytest tests/detection/test_raven_to_yolo.py -v
```

Expected: ImportError — module doesn't exist yet.

- [x] **Step 3: Implement raven_to_yolo.py**

```python
# soundbay/detection/__init__.py
"""YOLO-based bioacoustic object detection module."""

# soundbay/detection/raven_to_yolo.py
"""Convert Raven Pro selection tables to YOLO bounding box format."""
import pandas as pd
from typing import List, Dict


def parse_raven_file(filepath: str) -> List[Dict]:
    """Parse a Raven selection table (.txt) into a list of annotation dicts."""
    df = pd.read_csv(filepath, sep="\t")
    col_map = {
        "Begin Time (s)": "begin_time",
        "End Time (s)": "end_time",
        "Low Freq (Hz)": "low_freq",
        "High Freq (Hz)": "high_freq",
    }
    annotations = []
    for _, row in df.iterrows():
        ann = {col_map[k]: float(row[k]) for k in col_map if k in df.columns}
        ann["label"] = str(row.get("Annotation", row.get("Type", "unknown"))).strip()
        annotations.append(ann)
    return annotations


def convert_raven_to_yolo(
    annotations: List[Dict],
    chunk_start: float,
    chunk_duration: float,
    freq_min: float,
    freq_max: float,
    class_id: int = 0,
    min_overlap_fraction: float = 0.5,
) -> List[str]:
    """Convert annotations overlapping a time chunk to YOLO format lines.

    YOLO format: <class> <x_center> <y_center> <width> <height> (all normalized 0-1)
    X axis = time, Y axis = frequency (inverted: high freq at top = y=0)
    """
    chunk_end = chunk_start + chunk_duration
    freq_range = freq_max - freq_min
    lines = []

    for ann in annotations:
        t_start = max(ann["begin_time"], chunk_start)
        t_end = min(ann["end_time"], chunk_end)
        overlap = t_end - t_start
        ann_duration = ann["end_time"] - ann["begin_time"]

        if overlap <= 0 or overlap / ann_duration < min_overlap_fraction:
            continue

        # Clip frequency to spectrogram bounds
        f_low = max(ann["low_freq"], freq_min)
        f_high = min(ann["high_freq"], freq_max)
        if f_high <= f_low:
            continue

        # Normalize to [0, 1]
        x_center = ((t_start + t_end) / 2 - chunk_start) / chunk_duration
        width = (t_end - t_start) / chunk_duration

        # Y axis inverted: high freq at y=0
        y_center = 1.0 - ((f_low + f_high) / 2 - freq_min) / freq_range
        height = (f_high - f_low) / freq_range

        # Clamp
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.001, min(1.0, width))
        height = max(0.001, min(1.0, height))

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines
```

- [x] **Step 4: Run tests**

```bash
pytest tests/detection/test_raven_to_yolo.py -v
```

Expected: PASS

- [x] **Step 5: Commit**

```bash
git add soundbay/detection/__init__.py soundbay/detection/raven_to_yolo.py tests/detection/test_raven_to_yolo.py
git commit -m "feat(detection): Raven selection table to YOLO label converter"
```

---

### Task 3: Spectrogram Generator (WAV → PNG chunks)

**Files:**
- Create: `soundbay/detection/spectrogram_generator.py`
- Create: `tests/detection/test_spectrogram_generator.py`

- [x] **Step 1: Write the failing test**

```python
# tests/detection/test_spectrogram_generator.py
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
```

- [x] **Step 2: Run test to verify it fails**

```bash
pytest tests/detection/test_spectrogram_generator.py -v
```

- [x] **Step 3: Implement spectrogram_generator.py**

```python
# soundbay/detection/spectrogram_generator.py
"""Generate spectrogram PNG images from WAV files for YOLO training."""
import os
from typing import List, Tuple

import librosa
import numpy as np
from PIL import Image


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
```

- [x] **Step 4: Run tests**

```bash
pytest tests/detection/test_spectrogram_generator.py -v
```

Expected: PASS

- [x] **Step 5: Commit**

```bash
git add soundbay/detection/spectrogram_generator.py tests/detection/test_spectrogram_generator.py
git commit -m "feat(detection): spectrogram chunk generator for YOLO training"
```

---

### Task 4: End-to-End Dataset Preparation Script

**Files:**
- Create: `soundbay/detection/prepare_dataset.py`

- [x] **Step 1: Write the dataset preparation CLI**

```python
# soundbay/detection/prepare_dataset.py
"""End-to-end dataset preparation: WAV + Raven → YOLO images + labels."""
import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

from soundbay.detection.spectrogram_generator import generate_spectrograms, get_chunk_start_from_filename
from soundbay.detection.raven_to_yolo import parse_raven_file, convert_raven_to_yolo


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


def find_annotation_file(wav_path: str, annotations_dir: str) -> str | None:
    """Find Raven .txt file matching a WAV file."""
    stem = Path(wav_path).stem
    # Try common naming patterns
    candidates = [
        os.path.join(annotations_dir, f"{stem}.txt"),
        os.path.join(annotations_dir, f"{stem}.Table.1.selections.txt"),
    ]
    # Also search for partial stem match
    if annotations_dir and os.path.isdir(annotations_dir):
        for f in os.listdir(annotations_dir):
            if stem in f and f.endswith(".txt"):
                candidates.append(os.path.join(annotations_dir, f))

    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def prepare_recording(
    wav_path: str,
    annotation_path: str | None,
    output_images_dir: str,
    output_labels_dir: str,
    params: dict = None,
) -> int:
    """Process one recording: generate spectrograms + labels. Returns chunk count."""
    params = params or DEFAULT_PARAMS

    # Generate spectrograms
    img_paths = generate_spectrograms(
        wav_path=wav_path,
        output_dir=output_images_dir,
        **params,
    )

    # Parse annotations if available
    annotations = []
    if annotation_path:
        annotations = parse_raven_file(annotation_path)

    # Generate YOLO labels for each chunk
    os.makedirs(output_labels_dir, exist_ok=True)
    for img_path in img_paths:
        chunk_start = get_chunk_start_from_filename(img_path)
        yolo_lines = convert_raven_to_yolo(
            annotations=annotations,
            chunk_start=chunk_start,
            chunk_duration=params["chunk_duration"],
            freq_min=params["freq_min"],
            freq_max=params["freq_max"],
            class_id=0,
        )
        label_path = os.path.join(
            output_labels_dir,
            Path(img_path).stem + ".txt",
        )
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

    return len(img_paths)


def prepare_dataset(
    data_dir: str,
    annotations_dir: str,
    output_dir: str,
    val_fraction: float = 0.2,
    seed: int = 42,
    params: dict = None,
):
    """Prepare full YOLO dataset from a directory of WAVs + annotations."""
    params = params or DEFAULT_PARAMS

    # Find all WAV files
    wav_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".wav")
    ])

    # Split by recording (not by chunk) to avoid data leakage
    random.seed(seed)
    random.shuffle(wav_files)
    n_val = max(1, int(len(wav_files) * val_fraction))
    val_wavs = set(wav_files[:n_val])
    train_wavs = set(wav_files[n_val:])

    total_train, total_val = 0, 0
    for wav_path in wav_files:
        split = "val" if wav_path in val_wavs else "train"
        ann_path = find_annotation_file(wav_path, annotations_dir)

        n_chunks = prepare_recording(
            wav_path=wav_path,
            annotation_path=ann_path,
            output_images_dir=os.path.join(output_dir, "images", split),
            output_labels_dir=os.path.join(output_dir, "labels", split),
            params=params,
        )

        if split == "train":
            total_train += n_chunks
        else:
            total_val += n_chunks

        status = "✓ annotated" if ann_path else "○ no annotations"
        print(f"  [{split}] {os.path.basename(wav_path)} → {n_chunks} chunks ({status})")

    print(f"\nDataset ready: {total_train} train + {total_val} val chunks in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO detection dataset from WAV + Raven annotations")
    parser.add_argument("--data-dir", required=True, help="Directory with WAV files")
    parser.add_argument("--annotations-dir", required=True, help="Directory with Raven .txt files")
    parser.add_argument("--output-dir", required=True, help="Output YOLO dataset directory")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-duration", type=float, default=15.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=640)
    args = parser.parse_args()

    params = {**DEFAULT_PARAMS, "chunk_duration": args.chunk_duration, "overlap": args.overlap, "img_size": args.img_size}
    prepare_dataset(args.data_dir, args.annotations_dir, args.output_dir, args.val_fraction, args.seed, params)


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Commit**

```bash
git add soundbay/detection/prepare_dataset.py
git commit -m "feat(detection): end-to-end dataset preparation CLI"
```

---

### Task 5: YOLO Dataset Configuration

**Files:**
- Create: `soundbay/conf/detection/humpback_mozambique.yaml`
- Create: `soundbay/conf/detection/train_humpback.yaml`

- [x] **Step 1: Create YOLO dataset YAML**

```yaml
# soundbay/conf/detection/humpback_mozambique.yaml
# YOLO dataset configuration for humpback whale call detection
# This file is referenced by ultralytics during training

path: /data/humpback_yolo  # root dataset dir (on EC2)
train: images/train
val: images/val

# Class names
names:
  0: humpback_call

# Number of classes
nc: 1
```

- [x] **Step 2: Create training hyperparameter config**

```yaml
# soundbay/conf/detection/train_humpback.yaml
# Training hyperparameters adapted from biodcase-2026 champion

# Model
model: yolo11m.pt
imgsz: 640
batch: 16  # Adjust based on GPU memory (16 for 16GB, 32 for 24GB+)

# Optimizer (critical: NOT default SGD)
optimizer: AdamW
lr0: 0.001
lrf: 0.01  # cosine decay to lr0 * lrf
weight_decay: 0.0005
warmup_epochs: 2

# Training schedule
epochs: 50
patience: 15
seed: 42

# Augmentation (spectrogram-safe ONLY)
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.3       # brightness/gain variation
translate: 0.05   # small positional shift
scale: 0.1        # small zoom
erasing: 0.2      # random erasing (simulates noise patches)
# DISABLED (would break spectrogram semantics):
flipud: 0.0
fliplr: 0.0
mosaic: 0.0
mixup: 0.0
degrees: 0.0
perspective: 0.0
shear: 0.0

# NMS
iou: 0.4          # IoU threshold for NMS during training
conf: 0.25        # confidence threshold

# Tracking
project: humpback_detection
name: mozambique_yolo11m
```

- [x] **Step 3: Commit**

```bash
git add soundbay/conf/detection/
git commit -m "feat(detection): YOLO dataset and training configs for humpback"
```

---

### Task 6: Training Wrapper

**Files:**
- Create: `soundbay/detection/train.py`

- [x] **Step 1: Write training script**

```python
# soundbay/detection/train.py
"""YOLO training wrapper for humpback call detection."""
import argparse
import yaml
from ultralytics import YOLO


def train(dataset_yaml: str, config_yaml: str, resume: str = None):
    """Train YOLOv11 model on humpback spectrogram dataset."""
    with open(config_yaml) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.pop("model", "yolo11m.pt")

    if resume:
        model = YOLO(resume)
        model.train(resume=True)
    else:
        model = YOLO(model_name)
        model.train(data=dataset_yaml, **cfg)


def main():
    parser = argparse.ArgumentParser(description="Train YOLO humpback detector")
    parser.add_argument("--dataset", required=True, help="Path to dataset YAML")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(args.dataset, args.config, args.resume)


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Commit**

```bash
git add soundbay/detection/train.py
git commit -m "feat(detection): YOLO training wrapper script"
```

---

### Task 7: Inference & Post-processing

**Files:**
- Create: `soundbay/detection/inference.py`
- Create: `soundbay/detection/postprocess.py`

- [x] **Step 1: Write inference script**

```python
# soundbay/detection/inference.py
"""Run trained YOLO model on new audio files."""
import argparse
import os
from pathlib import Path

from ultralytics import YOLO

from soundbay.detection.spectrogram_generator import generate_spectrograms, get_chunk_start_from_filename
from soundbay.detection.postprocess import merge_detections, export_to_raven


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


def infer_audio(
    model_path: str,
    wav_path: str,
    output_dir: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.4,
    params: dict = None,
):
    """Run detection on a single audio file."""
    params = params or DEFAULT_PARAMS
    model = YOLO(model_path)

    # Generate spectrograms
    temp_img_dir = os.path.join(output_dir, "temp_spectrograms")
    img_paths = generate_spectrograms(wav_path=wav_path, output_dir=temp_img_dir, **params)

    # Run inference on all chunks
    raw_detections = []
    for img_path in img_paths:
        chunk_start = get_chunk_start_from_filename(img_path)
        results = model.predict(img_path, conf=conf_threshold, iou=iou_threshold, verbose=False)

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x_c, y_c, w, h = box.xywhn[0].tolist()
                conf = box.conf[0].item()

                # Convert normalized image coords back to time-frequency
                t_start = chunk_start + (x_c - w / 2) * params["chunk_duration"]
                t_end = chunk_start + (x_c + w / 2) * params["chunk_duration"]
                freq_range = params["freq_max"] - params["freq_min"]
                # Y is inverted in our spectrograms
                f_high = params["freq_max"] - (y_c - h / 2) * freq_range
                f_low = params["freq_max"] - (y_c + h / 2) * freq_range

                raw_detections.append({
                    "begin_time": t_start,
                    "end_time": t_end,
                    "low_freq": f_low,
                    "high_freq": f_high,
                    "confidence": conf,
                    "class": "humpback_call",
                })

    # Post-process: merge overlapping detections from overlapping chunks
    merged = merge_detections(raw_detections, iou_threshold=0.3)

    # Export
    stem = Path(wav_path).stem
    raven_path = os.path.join(output_dir, f"{stem}_detections.txt")
    export_to_raven(merged, raven_path)

    print(f"  {stem}: {len(merged)} detections → {raven_path}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Run humpback YOLO detector on audio")
    parser.add_argument("--model", required=True, help="Path to trained .pt model")
    parser.add_argument("--input", required=True, help="WAV file or directory of WAVs")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.4)
    args = parser.parse_args()

    if os.path.isfile(args.input):
        infer_audio(args.model, args.input, args.output_dir, args.conf, args.iou)
    else:
        for f in sorted(os.listdir(args.input)):
            if f.lower().endswith(".wav"):
                infer_audio(args.model, os.path.join(args.input, f), args.output_dir, args.conf, args.iou)


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Write post-processing module**

```python
# soundbay/detection/postprocess.py
"""Post-processing: NMS across chunks, temporal merging, Raven export."""
from typing import List, Dict


def compute_iou_tf(det_a: Dict, det_b: Dict) -> float:
    """Compute IoU in time-frequency space (2D)."""
    # Time overlap
    t_inter_start = max(det_a["begin_time"], det_b["begin_time"])
    t_inter_end = min(det_a["end_time"], det_b["end_time"])
    t_overlap = max(0, t_inter_end - t_inter_start)

    # Frequency overlap
    f_inter_low = max(det_a["low_freq"], det_b["low_freq"])
    f_inter_high = min(det_a["high_freq"], det_b["high_freq"])
    f_overlap = max(0, f_inter_high - f_inter_low)

    intersection = t_overlap * f_overlap

    area_a = (det_a["end_time"] - det_a["begin_time"]) * (det_a["high_freq"] - det_a["low_freq"])
    area_b = (det_b["end_time"] - det_b["begin_time"]) * (det_b["high_freq"] - det_b["low_freq"])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def merge_detections(
    detections: List[Dict],
    iou_threshold: float = 0.3,
) -> List[Dict]:
    """Greedy NMS in time-frequency space. Keeps highest confidence."""
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []

    for det in sorted_dets:
        suppress = False
        for kept_det in kept:
            if compute_iou_tf(det, kept_det) >= iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(det)

    return kept


def export_to_raven(detections: List[Dict], output_path: str):
    """Export detections as a Raven Pro selection table."""
    header = "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tAnnotation\tConfidence\n"
    lines = [header]

    for i, det in enumerate(sorted(detections, key=lambda d: d["begin_time"]), 1):
        line = (
            f"{i}\tSpectrogram 1\t1\t"
            f"{det['begin_time']:.3f}\t{det['end_time']:.3f}\t"
            f"{det['low_freq']:.1f}\t{det['high_freq']:.1f}\t"
            f"{det['class']}\t{det['confidence']:.4f}\n"
        )
        lines.append(line)

    with open(output_path, "w") as f:
        f.writelines(lines)
```

- [x] **Step 3: Commit**

```bash
git add soundbay/detection/inference.py soundbay/detection/postprocess.py
git commit -m "feat(detection): inference pipeline with NMS and Raven export"
```

---

### Task 8: S3 Upload Script

**Files:**
- Create: `soundbay/scripts/upload_to_s3.py`

- [x] **Step 1: Write S3 upload utility**

```python
# soundbay/scripts/upload_to_s3.py
"""Upload prepared YOLO dataset to S3 for EC2 training."""
import argparse
import os
import boto3
from pathlib import Path


S3_BUCKET = "deepvoice-external"
S3_PREFIX = "humpback_yolo_detection"


def upload_directory(local_dir: str, bucket: str, prefix: str):
    """Upload a local directory to S3 recursively."""
    s3 = boto3.client("s3")
    local_path = Path(local_dir)

    files = [f for f in local_path.rglob("*") if f.is_file()]
    print(f"Uploading {len(files)} files to s3://{bucket}/{prefix}/")

    for i, filepath in enumerate(files):
        relative = filepath.relative_to(local_path)
        s3_key = f"{prefix}/{relative.as_posix()}"
        s3.upload_file(str(filepath), bucket, s3_key)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)} uploaded...")

    print(f"Done. Dataset at s3://{bucket}/{prefix}/")


def main():
    parser = argparse.ArgumentParser(description="Upload YOLO dataset to S3")
    parser.add_argument("--local-dir", required=True, help="Local dataset directory")
    parser.add_argument("--bucket", default=S3_BUCKET)
    parser.add_argument("--prefix", default=S3_PREFIX)
    args = parser.parse_args()

    upload_directory(args.local_dir, args.bucket, args.prefix)


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Commit**

```bash
git add scripts/upload_to_s3.py
git commit -m "feat: S3 upload script for YOLO dataset"
```

---

## Phase 2: EC2 Training Execution

### Task 9: EC2 Setup & Training Run

This task is executed on the EC2 instance after launch.

- [x] **Step 1: EC2 instance setup**

```bash
# SSH into EC2 instance
# Recommended: g5.xlarge (1x A10G 24GB) or g4dn.xlarge (1x T4 16GB)

# Install system deps
sudo apt update && sudo apt install -y python3.10 python3.10-venv ffmpeg libsndfile1

# Clone and setup
git clone https://github.com/deep-voice/soundbay.git
cd soundbay
git checkout feature/yolo-humpback-detection
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ultralytics gdown librosa soundfile pillow pyyaml boto3
```

- [x] **Step 2: Download data to EC2**

```bash
# Option A: From S3 (if already uploaded)
aws s3 sync s3://deepvoice-external/humpback_yolo_detection /data/humpback_yolo

# Option B: From Google Drive directly on EC2
python scripts/download_humpback_data.py --output-dir /data/raw

# Prepare YOLO dataset
python -m soundbay.detection.prepare_dataset \
    --data-dir /data/raw/mozambique_2021 \
    --annotations-dir /data/raw/mozambique_2021 \
    --output-dir /data/humpback_yolo \
    --val-fraction 0.2 \
    --seed 42
```

- [x] **Step 3: Prepare Costa Rica test set separately**

```bash
python -m soundbay.detection.prepare_dataset \
    --data-dir /data/raw/costa_rica_2022 \
    --annotations-dir /data/raw/costa_rica_2022 \
    --output-dir /data/humpback_yolo/test \
    --val-fraction 0.0 \
    --seed 42
```

- [x] **Step 4: Verify dataset structure**

```bash
echo "Train images:" && ls /data/humpback_yolo/images/train/ | wc -l
echo "Train labels:" && ls /data/humpback_yolo/labels/train/ | wc -l
echo "Val images:" && ls /data/humpback_yolo/images/val/ | wc -l
echo "Val labels:" && ls /data/humpback_yolo/labels/val/ | wc -l

# Verify labels have content (not all empty)
find /data/humpback_yolo/labels/train -name "*.txt" -size +0 | wc -l
```

- [x] **Step 5: Update dataset YAML path and launch training**

```bash
# Edit the dataset yaml to point to /data/humpback_yolo
# Then train:
python -m soundbay.detection.train \
    --dataset soundbay/conf/detection/humpback_mozambique.yaml \
    --config soundbay/conf/detection/train_humpback.yaml
```

Expected: Training runs for up to 50 epochs with early stopping (patience=15). On A10G with batch=16, expect ~2-3 min/epoch.

- [x] **Step 6: Evaluate on Costa Rica test set**

```bash
python -m soundbay.detection.inference \
    --model runs/detect/mozambique_yolo11m/weights/best.pt \
    --input /data/raw/costa_rica_2022 \
    --output-dir /data/results/costa_rica_test \
    --conf 0.25
```

---

### Task 10: Domain Adaptation (if needed)

If Costa Rica test performance is poor (expected due to domain shift between Mozambique and Costa Rica humpback populations):

- [x] **Step 1: Add 1-2 Costa Rica recordings to training**

```bash
# Move 2 "In Progress" Costa Rica recordings to a fine-tuning split
# Re-prepare with mixed dataset:
python -m soundbay.detection.prepare_dataset \
    --data-dir /data/raw/costa_rica_finetune \
    --annotations-dir /data/raw/costa_rica_finetune \
    --output-dir /data/humpback_yolo_mixed \
    --val-fraction 0.0 \
    --seed 42

# Copy into main train split
cp /data/humpback_yolo_mixed/images/train/* /data/humpback_yolo/images/train/
cp /data/humpback_yolo_mixed/labels/train/* /data/humpback_yolo/labels/train/
```

- [x] **Step 2: Fine-tune from best Mozambique checkpoint**

```yaml
# Modify train config for fine-tuning:
# epochs: 20, lr0: 0.0003, patience: 10
```

```bash
python -m soundbay.detection.train \
    --dataset soundbay/conf/detection/humpback_mozambique.yaml \
    --config soundbay/conf/detection/train_humpback_finetune.yaml \
    --resume runs/detect/mozambique_yolo11m/weights/best.pt
```

---

## Summary: Execution Order

| # | Phase | Where | What |
|---|-------|-------|------|
| 1 | Data download | Local/EC2 | Download WAVs + annotations from Google Drive |
| 2 | Code development | Local (this branch) | Tasks 2–8: detection module, configs, scripts |
| 3 | Upload data | Local → S3 | Upload prepared dataset (or run data prep on EC2) |
| 4 | EC2 setup | EC2 | Pull branch, install deps, download data |
| 5 | Training | EC2 | Run YOLO training (~1-2 hours on A10G) |
| 6 | Evaluation | EC2 | Test on Costa Rica, check IoU metrics |
| 7 | Domain adapt | EC2 | If needed: add CR recordings, fine-tune |
| 8 | Results | Local | Pull model weights, run inference locally |

## Key Risks & Mitigations

1. **Raven annotation format varies**: The `parse_raven_file` function handles flexible column names. Run `head` on actual .txt files to verify before batch processing.
2. **Few annotated recordings (29 Mozambique)**: With 15s chunks at 50% overlap, each multi-hour recording yields 100s of chunks. Background chunks (no calls) serve as hard negatives.
3. **Domain shift Mozambique→Costa Rica**: Different populations have different song dialects. Plan includes fine-tuning fallback.
4. **Google Drive download limits**: For large batches, gdown may hit rate limits. Alternative: manual download + S3 upload.
