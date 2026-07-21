# YOLO Humpback Detection — Post-Processing Filters

**Date:** 2026-07-21
**Branch:** feature/yolo-humpback-detection
**Status:** Design — pending user review

## Context

We trained a YOLO object detector to find humpback calls in spectrogram images —
a departure from the repo's traditional time-domain classifiers. Training data was
mostly Mozambique with a small amount of Costa Rica (roughly half an annotated call);
the rest of Costa Rica was held out as a test set. Inference produces Raven Pro
selection tables for the biologist.

The biologist reviewed the Costa Rica test results and raised three issues:

1. **Harmonics** — a call and its harmonic(s) are both marked as separate detections.
2. **Overlapping / nested boxes** — the model draws a precise rectangle on a call
   *and* a larger box around part of the same call.
3. **False positives** — enough spurious detections that a higher confidence
   threshold is warranted.

These are the cheap improvements to try before re-training with different spectrogram
parameters or augmentations.

Ground truth: only a **partial** set of Costa Rica test files is annotated. We validate
post-processing on that annotated subset and apply the tuned settings to the rest.

## Goals

- Reduce harmonic duplicate detections.
- Collapse overlapping/nested boxes that describe the same call.
- Reduce false positives via a higher default confidence threshold.
- Keep every filter independently testable and tunable (togglable via CLI args).

## Non-Goals

- No re-training, spectrogram-parameter changes, or augmentation work (that comes later).
- No formal threshold-sweep tool. The new default confidence is chosen by inspecting
  precision/recall on the annotated CR subset, not by an automated sweep.
- No changes to the legacy time-domain pipeline (`soundbay/inference.py`). All work is
  confined to `soundbay/detection/`.

## Approach

**Sequential filter chain** (chosen over a single unified suppression pass or a
config-driven pipeline). Each filter is a pure function on a list of detection dicts,
living in `soundbay/detection/postprocess.py` alongside the existing `merge_detections`.
`inference.py` composes them in a fixed order. This matches the existing code style,
keeps each biological concept (harmonic vs. same-call duplicate) separate and testable,
and lets any single filter be disabled.

A "detection dict" has the existing shape:
`{begin_time, end_time, low_freq, high_freq, confidence, class}`.

## Pipeline Order

Applied to the raw per-chunk detections inside `infer_audio`, in this order:

1. **Confidence threshold** — applied at `model.predict(conf=...)` as today, but the
   default rises from `0.25` to a higher value (chosen from the annotated subset;
   expected ~0.4–0.5). Cheapest filter, so it runs first and reduces the box count
   for later steps.

2. **Same-call overlap merge** — `suppress_overlapping()`. For any two boxes that
   overlap in **both** time and frequency (intersection-over-minimum-area ≥
   `overlap_iomin`, default 0.6), suppress one. **The survivor is the box with the
   longer time duration** (not the higher confidence). Rationale: the precise full-call
   rectangle is longer in time than a box drawn around half the call, so it wins.
   Tie-break: if durations are equal, keep the higher-confidence box.

3. **Harmonic suppression** — `suppress_harmonics()`. Runs *after* the overlap merge so
   it operates on clean, well-formed boxes. For two boxes overlapping in **time** by
   ≥ `harmonic_time_overlap` (default 0.6, measured as time-intersection over the
   shorter box's duration) where one box sits **entirely above** the other in frequency
   (B.low_freq ≥ A.high_freq − `freq_margin`), suppress the higher box and keep the
   lower (fundamental). The fundamental is kept regardless of confidence.

4. **Existing IoU-NMS** — `merge_detections(iou_threshold=0.3)` runs last as a final
   cleanup of true near-duplicates across overlapping chunks.

## Components

### `soundbay/detection/postprocess.py`

Existing (unchanged): `compute_iou_tf`, `merge_detections`, `export_to_raven`.

New:

- `compute_time_overlap_ratio(a, b) -> float` — time-intersection divided by the
  shorter box's duration. Helper for both new filters.
- `compute_iomin(a, b) -> float` — 2D time-frequency intersection over the smaller
  box's area. Helper for overlap merge.
- `suppress_overlapping(detections, iomin_threshold=0.6) -> List[Dict]` — greedy pass;
  survivor chosen by longest duration, tie-break by confidence.
- `suppress_harmonics(detections, time_overlap_threshold=0.6, freq_margin=50.0)
  -> List[Dict]` — suppress higher-frequency box of a time-overlapping stacked pair.
  `freq_margin` (Hz) tolerates box-edge fuzziness in the "entirely above" test;
  default 50 Hz, tunable during validation.

### `soundbay/detection/inference.py`

- Compose the new filters into `infer_audio` in the order above.
- New CLI args: `--overlap-iomin` (default 0.6), `--harmonic-time-overlap` (default 0.6),
  and raised `--conf` default. Merge-IoU unchanged.
- Each new filter is skippable (e.g. threshold of 0 or a `--no-*` flag) so the biologist
  can compare with/without.

## Testing

Unit tests (pure functions, hand-built detection dicts — no model run required):

- **Nested box**: full-call rectangle + smaller box inside it → one survives, the longer.
- **Half-call box**: precise rectangle + box around half the call → precise (longer) one
  survives.
- **Harmonic stack**: fundamental + harmonic above it, time-overlapping → harmonic
  suppressed, fundamental kept.
- **Genuine simultaneous calls**: two boxes overlapping in time but *not* satisfying the
  same-call or harmonic criteria → both kept (no false merge).
- **Confidence keep-lower-fundamental**: harmonic has higher confidence than fundamental
  → fundamental still kept.

## Validation

Run updated inference on the annotated CR subset, score detections (precision / recall /
F1) against the manual Raven annotations, and pick the confidence default from that.
Local PC should handle YOLO inference (slower); fall back to the training EC2 if it fails.

## Open Questions

- None blocking. `freq_margin` default (50 Hz) and the new confidence default will be
  refined during validation against the annotated CR subset.
