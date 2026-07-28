"""Score detection Raven table against ground-truth Raven table using time-IoU matching."""
import argparse
import csv


def load_raven(path):
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({
                "begin": float(row["Begin Time (s)"]),
                "end": float(row["End Time (s)"]),
            })
    return rows


def time_iou(a, b):
    inter = max(0.0, min(a["end"], b["end"]) - max(a["begin"], b["begin"]))
    union = (a["end"] - a["begin"]) + (b["end"] - b["begin"]) - inter
    return inter / union if union > 0 else 0.0


def score(gt_path, det_path, iou_threshold=0.1):
    gt = load_raven(gt_path)
    det = load_raven(det_path)

    matched_gt = set()
    matched_det = set()
    for i, d in enumerate(det):
        best_j, best_iou = None, 0.0
        for j, g in enumerate(gt):
            if j in matched_gt:
                continue
            iou = time_iou(d, g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j is not None and best_iou >= iou_threshold:
            matched_gt.add(best_j)
            matched_det.add(i)

    tp = len(matched_det)
    fp = len(det) - tp
    fn = len(gt) - len(matched_gt)
    precision = tp / len(det) if det else 0.0
    recall = tp / len(gt) if gt else 0.0
    print(f"GT: {len(gt)}  Detections: {len(det)}")
    print(f"TP={tp} FP={fp} FN={fn}")
    print(f"Precision={precision:.3f} Recall={recall:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--det", required=True)
    parser.add_argument("--iou", type=float, default=0.1)
    args = parser.parse_args()
    score(args.gt, args.det, args.iou)


if __name__ == "__main__":
    main()
