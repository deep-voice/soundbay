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


def suppress_overlapping(
    detections: List[Dict],
    iomin_threshold: float = 0.6,
) -> List[Dict]:
    """Collapse boxes overlapping in both time and frequency (same call).

    Two boxes whose intersection-over-minimum-area >= iomin_threshold are
    treated as the same call. The longer-duration box survives (tie-break:
    higher confidence). Catches nested / half-call boxes that plain IoU misses.
    """
    if not detections:
        return []

    def _duration(d):
        return d["end_time"] - d["begin_time"]

    # Process longest-first (tie-break: higher confidence) so survivors are
    # the boxes we want to keep.
    ordered = sorted(detections, key=lambda d: (_duration(d), d["confidence"]), reverse=True)
    kept = []
    for det in ordered:
        if any(compute_iomin(det, k) >= iomin_threshold for k in kept):
            continue
        kept.append(det)
    return kept


def suppress_harmonics(
    detections: List[Dict],
    time_overlap_threshold: float = 0.6,
    freq_margin: float = 50.0,
) -> List[Dict]:
    """Suppress the higher-frequency box of a time-overlapping stacked pair.

    When two boxes overlap in time by >= time_overlap_threshold and one sits
    entirely above the other in frequency (its low_freq is at or above the
    lower box's high_freq minus freq_margin, tolerating fuzzy box edges), the
    higher box is treated as a harmonic and dropped; the lower (fundamental)
    is kept regardless of confidence.

    Known limitation: A genuine simultaneous call in a higher frequency band
    that overlaps in time will be indistinguishable from a harmonic by this
    geometric heuristic and will be suppressed. This is an accepted
    precision-over-recall trade-off.
    """
    if not detections:
        return []

    suppressed = set()  # indices marked as harmonics
    for i, di in enumerate(detections):
        for j, dj in enumerate(detections):
            if i == j or i in suppressed or j in suppressed:
                continue
            if compute_time_overlap_ratio(di, dj) < time_overlap_threshold:
                continue
            # Identify which box is higher in frequency.
            lower, higher, higher_idx = (di, dj, j) if di["low_freq"] <= dj["low_freq"] else (dj, di, i)
            if higher["low_freq"] >= lower["high_freq"] - freq_margin:
                suppressed.add(higher_idx)

    return [d for k, d in enumerate(detections) if k not in suppressed]


def apply_postprocessing(
    detections: List[Dict],
    overlap_iomin: float = 0.6,
    harmonic_time_overlap: float = 0.6,
    harmonic_freq_margin: float = 50.0,
    merge_iou: float = 0.3,
) -> List[Dict]:
    """Full post-processing chain: overlap merge -> harmonics -> IoU-NMS."""
    result = suppress_overlapping(detections, iomin_threshold=overlap_iomin)
    result = suppress_harmonics(
        result,
        time_overlap_threshold=harmonic_time_overlap,
        freq_margin=harmonic_freq_margin,
    )
    # Note: "longest-duration wins" from suppress_overlapping holds strictly only
    # for IoMin >= overlap_iomin; final IoU-NMS may pick higher-confidence box for moderate overlaps.
    result = merge_detections(result, iou_threshold=merge_iou)
    return result


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
