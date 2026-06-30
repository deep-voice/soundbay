"""Convert Raven Pro selection tables to YOLO bounding box format."""
import pandas as pd
from typing import List, Dict


def parse_raven_file(filepath: str) -> List[Dict]:
    """Parse a Raven selection table (.txt) into a list of annotation dicts."""
    df = pd.read_csv(filepath, sep="\t", encoding_errors="replace")
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

        f_low = max(ann["low_freq"], freq_min)
        f_high = min(ann["high_freq"], freq_max)
        if f_high <= f_low:
            continue

        x_center = ((t_start + t_end) / 2 - chunk_start) / chunk_duration
        width = (t_end - t_start) / chunk_duration

        y_center = 1.0 - ((f_low + f_high) / 2 - freq_min) / freq_range
        height = (f_high - f_low) / freq_range

        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.001, min(1.0, width))
        height = max(0.001, min(1.0, height))

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines
