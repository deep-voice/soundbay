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

    temp_img_dir = os.path.join(output_dir, "temp_spectrograms")
    img_paths = generate_spectrograms(wav_path=wav_path, output_dir=temp_img_dir, **params)

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

                t_start = chunk_start + (x_c - w / 2) * params["chunk_duration"]
                t_end = chunk_start + (x_c + w / 2) * params["chunk_duration"]
                freq_range = params["freq_max"] - params["freq_min"]
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

    merged = merge_detections(raw_detections, iou_threshold=0.3)

    stem = Path(wav_path).stem
    raven_path = os.path.join(output_dir, f"{stem}_detections.txt")
    export_to_raven(merged, raven_path)

    print(f"  {stem}: {len(merged)} detections -> {raven_path}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Run humpback YOLO detector on audio")
    parser.add_argument("--model", required=True, help="Path to trained .pt model")
    parser.add_argument("--input", required=True, help="WAV file or directory of WAVs")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isfile(args.input):
        infer_audio(args.model, args.input, args.output_dir, args.conf, args.iou)
    else:
        for f in sorted(os.listdir(args.input)):
            if f.lower().endswith(".wav"):
                infer_audio(args.model, os.path.join(args.input, f), args.output_dir, args.conf, args.iou)


if __name__ == "__main__":
    main()
