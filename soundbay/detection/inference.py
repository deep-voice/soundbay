"""Run trained YOLO model on new audio files."""
import argparse
import os
from pathlib import Path

from ultralytics import YOLO

from soundbay.detection.spectrogram_generator import (
    DEFAULT_PARAMS,
    PARAM_KEYS,
    generate_spectrograms,
    get_chunk_start_from_filename,
    load_params_sidecar,
    resolve_params,
)
from soundbay.detection.postprocess import apply_postprocessing, export_to_raven


def infer_audio(
    model_path: str,
    wav_path: str,
    output_dir: str,
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.4,
    overlap_iomin: float = 0.6,
    harmonic_time_overlap: float = 0.6,
    harmonic_freq_margin: float = 50.0,
    params: dict = None,
):
    """Run detection on a single audio file.

    ``params`` are the spectrogram geometry settings; when omitted they are read
    from the checkpoint's sidecar (see ``load_params_sidecar``) falling back to
    ``DEFAULT_PARAMS``. These MUST match the settings the model was trained on.
    """
    if params is None:
        params = resolve_params(sidecar=load_params_sidecar(model_path))
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

    merged = apply_postprocessing(
        raw_detections,
        overlap_iomin=overlap_iomin,
        harmonic_time_overlap=harmonic_time_overlap,
        harmonic_freq_margin=harmonic_freq_margin,
        merge_iou=0.3,
    )

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
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--overlap-iomin", type=float, default=0.6)
    parser.add_argument("--harmonic-time-overlap", type=float, default=0.6)
    parser.add_argument("--harmonic-freq-margin", type=float, default=50.0)

    # Spectrogram geometry. Defaults come from the checkpoint sidecar when present,
    # else DEFAULT_PARAMS; these flags override either. Must match training.
    spec = parser.add_argument_group("spectrogram params (must match training)")
    spec.add_argument("--chunk-duration", type=float, default=None)
    spec.add_argument("--overlap", type=float, default=None)
    spec.add_argument("--target-sr", type=int, default=None)
    spec.add_argument("--n-fft", type=int, default=None)
    spec.add_argument("--hop-length", type=int, default=None)
    spec.add_argument("--freq-min", type=float, default=None)
    spec.add_argument("--freq-max", type=float, default=None)
    spec.add_argument("--img-size", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sidecar = load_params_sidecar(args.model)
    overrides = {key: getattr(args, key) for key in PARAM_KEYS}
    params = resolve_params(sidecar=sidecar, overrides=overrides)
    source = f"sidecar {sorted(sidecar)}" if sidecar else "defaults"
    print(f"Spectrogram params ({source} + CLI overrides): {params}")
    if params == DEFAULT_PARAMS and not sidecar:
        print(
            "  WARNING: using DEFAULT_PARAMS with no sidecar for this checkpoint. "
            "If it was trained on different spectrogram geometry, detections will be wrong."
        )

    common = dict(
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        overlap_iomin=args.overlap_iomin,
        harmonic_time_overlap=args.harmonic_time_overlap,
        harmonic_freq_margin=args.harmonic_freq_margin,
        params=params,
    )
    if os.path.isfile(args.input):
        infer_audio(args.model, args.input, args.output_dir, **common)
    else:
        for f in sorted(os.listdir(args.input)):
            if f.lower().endswith(".wav"):
                infer_audio(args.model, os.path.join(args.input, f), args.output_dir, **common)


if __name__ == "__main__":
    main()
