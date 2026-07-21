"""Post-processing: NMS across chunks, temporal merging, Raven export."""
from typing import List, Dict


def compute_iou_tf(det_a: Dict, det_b: Dict) -> float:
    """Compute IoU in time-frequency space (2D)."""
    t_inter_start = max(det_a["begin_time"], det_b["begin_time"])
    t_inter_end = min(det_a["end_time"], det_b["end_time"])
    t_overlap = max(0, t_inter_end - t_inter_start)

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


def compute_time_overlap_ratio(det_a: Dict, det_b: Dict) -> float:
    """Time-intersection divided by the shorter box's duration."""
    inter_start = max(det_a["begin_time"], det_b["begin_time"])
    inter_end = min(det_a["end_time"], det_b["end_time"])
    overlap = max(0.0, inter_end - inter_start)
    dur_a = det_a["end_time"] - det_a["begin_time"]
    dur_b = det_b["end_time"] - det_b["begin_time"]
    shorter = min(dur_a, dur_b)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def compute_iomin(det_a: Dict, det_b: Dict) -> float:
    """2D time-frequency intersection over the smaller box's area."""
    t_overlap = max(0.0, min(det_a["end_time"], det_b["end_time"])
                    - max(det_a["begin_time"], det_b["begin_time"]))
    f_overlap = max(0.0, min(det_a["high_freq"], det_b["high_freq"])
                    - max(det_a["low_freq"], det_b["low_freq"]))
    intersection = t_overlap * f_overlap
    area_a = (det_a["end_time"] - det_a["begin_time"]) * (det_a["high_freq"] - det_a["low_freq"])
    area_b = (det_b["end_time"] - det_b["begin_time"]) * (det_b["high_freq"] - det_b["low_freq"])
    smaller = min(area_a, area_b)
    if smaller <= 0:
        return 0.0
    return intersection / smaller


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
