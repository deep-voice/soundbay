# Humpback YOLO detector — Costa Rica 2022 results

**Date:** 2026-07-28
**Model:** YOLOv11m, fine-tuned on Costa Rica (`cr_finetune`), starting from the
Mozambique-trained detector.
**Confidence threshold used for the attached tables:** **0.15**

## Files

| File | Detections |
|------|-----------|
| `220824-120311_Tr1_detections.txt` | 407 |
| `220825-120510_Tr1_detections.txt` | 552 |

Both are Raven Pro selection tables — open them alongside the corresponding WAV in
Raven. The last column, `Confidence`, is the detector's score (0–1); you can sort or
filter on it to review the least-certain detections first.

Note the two recordings are not equivalent as a test:

- **`220824-120311_Tr1` is fully held out** — the model never saw it during training,
  so its numbers are the honest estimate of performance on new recordings.
- **`220825-120510_Tr1` was used for fine-tuning**, so treat its numbers as a
  sanity check rather than as evidence of generalization.

## How threshold affects precision and recall

Scored against your manual annotations by time overlap (a detection counts as
correct if it overlaps a annotated call by IoU ≥ 0.1 in time).

- **Precision** = of the detections we report, the fraction that are real calls
  (high precision → little noise to wade through).
- **Recall** = of the calls you annotated, the fraction we found
  (high recall → few missed calls).

### `220824-120311_Tr1` — held out, 374 annotated calls

| Confidence | Detections | Precision | Recall |
|-----------|-----------|-----------|--------|
| 0.45 | 202 | 0.970 | 0.524 |
| 0.25 | 336 | 0.845 | 0.759 |
| **0.15** | **407** | **0.737** | **0.802** |
| 0.10 | 445 | 0.692 | 0.824 |
| 0.05 | 570 | 0.595 | 0.906 |

### `220825-120510_Tr1` — used for fine-tuning, 804 annotated calls

| Confidence | Detections | Precision | Recall |
|-----------|-----------|-----------|--------|
| 0.45 | 155 | 0.974 | 0.188 |
| 0.25 | 424 | 0.948 | 0.500 |
| **0.15** | **552** | **0.899** | **0.617** |
| 0.10 | 621 | 0.876 | 0.677 |
| 0.05 | 743 | 0.817 | 0.755 |

## Why 0.15

It sits at the knee of the curve. Going from 0.25 → 0.15 buys a solid recall gain
(+4 pts on the held-out file, +12 pts on the other) for a moderate precision cost.
Pushing further to 0.10 or 0.05 keeps adding recall but precision falls off faster
than recall improves, so you would spend most of your review time rejecting noise.

**This is a judgement call and easy for us to change** — tell us which way you'd
rather err:

- **You'd rather not miss calls** (recall first) → we move to 0.10 or 0.05 and you
  accept reviewing more false positives.
- **You'd rather review a clean list** (precision first) → we move to 0.25 or 0.45.
  Note 0.45 finds only about half the calls on the held-out file and under a fifth
  on the other, so we don't recommend it.

Since the `Confidence` column is in the attached tables, you can also explore this
yourself: filtering to `Confidence >= 0.25` in Raven reproduces the 0.25 row above
without us re-running anything.

## What changed in the post-processing

Following your earlier feedback, the detector now:

1. **Merges duplicate boxes on the same call** — where a call was previously marked
   both as one long box and as several partial boxes, the longest box wins.
2. **Suppresses harmonics** — when a detection sits directly above another in
   frequency and overlaps it in time, we keep only the lower (fundamental) one.

One known trade-off: if two *genuinely different* calls happen simultaneously in
stacked frequency bands, rule 2 will discard the upper one, since geometry alone
can't distinguish that from a harmonic. If you see this happening in your review,
let us know and we'll revisit.

## Caveats

- Recall on the fine-tuning recording (0.617) being *lower* than on the held-out one
  (0.802) is unexpected and we haven't yet explained it. `220825-120510_Tr1` has
  more than twice as many annotated calls (804 vs 374), so it may simply be a denser,
  harder recording. Worth keeping in mind when interpreting these numbers.
- Only these two recordings have complete manual annotations, so all the figures above
  rest on a small sample.
- Matching uses time overlap only; it ignores whether the frequency bounds of a box are
  right. A detection counted as correct may still have imprecise frequency extent.
