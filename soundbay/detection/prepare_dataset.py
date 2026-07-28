"""End-to-end dataset preparation: WAV + Raven -> YOLO images + labels."""
import argparse
import json
import os
import random
from pathlib import Path
from typing import List

from soundbay.detection.spectrogram_generator import (
    DEFAULT_PARAMS,
    generate_spectrograms,
    get_chunk_start_from_filename,
)
from soundbay.detection.raven_to_yolo import parse_raven_file, convert_raven_to_yolo


def find_annotation_file(wav_path: str, annotations_dir: str) -> str | None:
    """Find Raven .txt file matching a WAV file."""
    stem = Path(wav_path).stem
    candidates = [
        os.path.join(annotations_dir, f"{stem}.txt"),
        os.path.join(annotations_dir, f"{stem}.Table.1.selections.txt"),
    ]
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

    img_paths = generate_spectrograms(
        wav_path=wav_path,
        output_dir=output_images_dir,
        **params,
    )

    annotations = []
    if annotation_path:
        annotations = parse_raven_file(annotation_path)

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

    wav_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".wav")
    ])

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

        status = "annotated" if ann_path else "no annotations"
        print(f"  [{split}] {os.path.basename(wav_path)} -> {n_chunks} chunks ({status})")

    # Record the geometry so inference can reproduce it (copy next to the checkpoint
    # as <weights>.spectrogram.json).
    params_path = os.path.join(output_dir, "spectrogram_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nDataset ready: {total_train} train + {total_val} val chunks in {output_dir}")
    print(f"Spectrogram params written to {params_path}")


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
