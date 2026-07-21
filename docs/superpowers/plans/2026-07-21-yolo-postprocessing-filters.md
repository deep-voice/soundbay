# YOLO Post-Processing Filters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add harmonic suppression, same-call overlap merging, and a raised confidence threshold to the humpback YOLO detector's post-processing, to address biologist feedback on Costa Rica test results.

**Architecture:** Three independent, pure filter functions on lists of detection dicts, added to `soundbay/detection/postprocess.py` alongside the existing `merge_detections`. `inference.py` composes them in a fixed order: confidence (at predict) → overlap merge → harmonic suppression → existing IoU-NMS. Each filter is unit-tested with hand-built dicts and tunable via CLI args.

**Tech Stack:** Python, pytest, ultralytics YOLO (only for the inference wiring, not the unit tests).

## Global Constraints

- Detection dict shape (verbatim from existing code): `{"begin_time": float, "end_time": float, "low_freq": float, "high_freq": float, "confidence": float, "class": str}`.
- All new filter functions are **pure**: take a `List[Dict]`, return a new `List[Dict]`, never mutate inputs, never call the model.
- Frequency convention: `high_freq > low_freq` (Hz). Time convention: `end_time > begin_time` (seconds).
- Follow existing test style: plain pytest functions (no classes), direct imports from `soundbay.detection.postprocess`, tests in `tests/detection/`.
- Do NOT touch the legacy `soundbay/inference.py`. All work is in `soundbay/detection/`.

---

### Task 1: Geometry helpers (`compute_time_overlap_ratio`, `compute_iomin`)

**Files:**
- Modify: `soundbay/detection/postprocess.py` (add two functions after `compute_iou_tf`)
- Test: `tests/detection/test_postprocess.py` (create)

**Interfaces:**
- Consumes: nothing (uses only detection dict fields).
- Produces:
  - `compute_time_overlap_ratio(a: Dict, b: Dict) -> float` — time-intersection divided by the SHORTER box's duration; 0.0 if no time overlap or a zero-duration box.
  - `compute_iomin(a: Dict, b: Dict) -> float` — 2D time-frequency intersection area divided by the SMALLER box's area; 0.0 if no overlap or a zero-area box.

- [ ] **Step 1: Write the failing tests**

```python
# tests/detection/test_postprocess.py
from soundbay.detection.postprocess import (
    compute_time_overlap_ratio,
    compute_iomin,
)


def _det(begin, end, low, high, conf=0.5, cls="humpback_call"):
    return {
        "begin_time": begin, "end_time": end,
        "low_freq": low, "high_freq": high,
        "confidence": conf, "class": cls,
    }


def test_time_overlap_ratio_full_containment():
    # b (1s) fully inside a's time span (4s) -> ratio over shorter (b) = 1.0
    a = _det(0.0, 4.0, 100, 500)
    b = _det(1.0, 2.0, 600, 900)
    assert compute_time_overlap_ratio(a, b) == 1.0


def test_time_overlap_ratio_no_overlap():
    a = _det(0.0, 2.0, 100, 500)
    b = _det(3.0, 5.0, 100, 500)
    assert compute_time_overlap_ratio(a, b) == 0.0


def test_time_overlap_ratio_partial():
    # overlap 1.0s (2.0->3.0); shorter box is b (2.0s) -> 0.5
    a = _det(0.0, 3.0, 100, 500)
    b = _det(2.0, 4.0, 100, 500)
    assert compute_time_overlap_ratio(a, b) == 0.5


def test_iomin_nested_box():
    # b fully inside a in both time and freq -> intersection == area(b) -> 1.0
    a = _det(0.0, 4.0, 100, 900)
    b = _det(1.0, 2.0, 300, 500)
    assert compute_iomin(a, b) == 1.0


def test_iomin_no_overlap():
    a = _det(0.0, 2.0, 100, 500)
    b = _det(3.0, 5.0, 600, 900)
    assert compute_iomin(a, b) == 0.0


def test_iomin_half_call_partial():
    # a: full call 0-4s / 100-500Hz (area 4*400=1600)
    # b: half-call box 0-2s / 100-500Hz (area 2*400=800), fully within a's freq band
    # intersection = 2*400 = 800; smaller area = 800 -> 1.0
    a = _det(0.0, 4.0, 100, 500)
    b = _det(0.0, 2.0, 100, 500)
    assert compute_iomin(a, b) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/detection/test_postprocess.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_time_overlap_ratio'`

- [ ] **Step 3: Implement the helpers**

Add to `soundbay/detection/postprocess.py` (after `compute_iou_tf`, before `merge_detections`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/detection/test_postprocess.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add soundbay/detection/postprocess.py tests/detection/test_postprocess.py
git commit -m "feat(detection): time-overlap and IoMin geometry helpers"
```

---

### Task 2: `suppress_overlapping` (same-call overlap merge, longest-duration wins)

**Files:**
- Modify: `soundbay/detection/postprocess.py` (add after `compute_iomin`)
- Test: `tests/detection/test_postprocess.py` (append)

**Interfaces:**
- Consumes: `compute_iomin` (Task 1).
- Produces: `suppress_overlapping(detections: List[Dict], iomin_threshold: float = 0.6) -> List[Dict]` — greedy pass; for any pair with `compute_iomin >= iomin_threshold`, keep the LONGER-duration box; tie-break by higher confidence. Returns a new list; input order among survivors preserved by descending duration processing.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/detection/test_postprocess.py
from soundbay.detection.postprocess import suppress_overlapping


def test_suppress_overlapping_keeps_longer_box():
    # precise full-call rectangle (4s) + box around half the call (2s), same freq band
    full = _det(0.0, 4.0, 100, 500, conf=0.6)
    half = _det(0.0, 2.0, 100, 500, conf=0.9)  # higher conf but shorter
    kept = suppress_overlapping([half, full], iomin_threshold=0.6)
    assert len(kept) == 1
    assert kept[0]["end_time"] == 4.0  # the longer box survives despite lower conf


def test_suppress_overlapping_nested_box():
    outer = _det(0.0, 4.0, 100, 900, conf=0.5)
    inner = _det(1.0, 2.0, 300, 500, conf=0.8)  # fully inside outer
    kept = suppress_overlapping([outer, inner], iomin_threshold=0.6)
    assert len(kept) == 1
    assert kept[0]["begin_time"] == 0.0 and kept[0]["end_time"] == 4.0


def test_suppress_overlapping_keeps_distinct_calls():
    # two calls separated in time -> IoMin 0 -> both kept
    a = _det(0.0, 2.0, 100, 500)
    b = _det(5.0, 7.0, 100, 500)
    kept = suppress_overlapping([a, b], iomin_threshold=0.6)
    assert len(kept) == 2


def test_suppress_overlapping_tie_break_by_confidence():
    # equal duration, heavy overlap -> higher confidence wins
    a = _det(0.0, 2.0, 100, 500, conf=0.4)
    b = _det(0.0, 2.0, 110, 510, conf=0.7)
    kept = suppress_overlapping([a, b], iomin_threshold=0.6)
    assert len(kept) == 1
    assert kept[0]["confidence"] == 0.7


def test_suppress_overlapping_empty():
    assert suppress_overlapping([]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/detection/test_postprocess.py -v`
Expected: FAIL with `ImportError: cannot import name 'suppress_overlapping'`

- [ ] **Step 3: Implement `suppress_overlapping`**

Add to `soundbay/detection/postprocess.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/detection/test_postprocess.py -v`
Expected: PASS (all Task 1 + Task 2 tests)

- [ ] **Step 5: Commit**

```bash
git add soundbay/detection/postprocess.py tests/detection/test_postprocess.py
git commit -m "feat(detection): suppress same-call overlapping boxes (longest wins)"
```

---

### Task 3: `suppress_harmonics` (drop higher-frequency harmonic of a time-overlapping pair)

**Files:**
- Modify: `soundbay/detection/postprocess.py` (add after `suppress_overlapping`)
- Test: `tests/detection/test_postprocess.py` (append)

**Interfaces:**
- Consumes: `compute_time_overlap_ratio` (Task 1).
- Produces: `suppress_harmonics(detections: List[Dict], time_overlap_threshold: float = 0.6, freq_margin: float = 50.0) -> List[Dict]` — for any pair overlapping in time by `>= time_overlap_threshold` where one box sits entirely above the other in frequency (`higher.low_freq >= lower.high_freq - freq_margin`), suppress the higher box and keep the lower (fundamental), regardless of confidence.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/detection/test_postprocess.py
from soundbay.detection.postprocess import suppress_harmonics


def test_suppress_harmonics_drops_higher_band():
    # fundamental 100-500Hz, harmonic 550-1000Hz, both 0-3s (full time overlap)
    fundamental = _det(0.0, 3.0, 100, 500, conf=0.6)
    harmonic = _det(0.0, 3.0, 550, 1000, conf=0.8)
    kept = suppress_harmonics([fundamental, harmonic])
    assert len(kept) == 1
    assert kept[0]["low_freq"] == 100  # fundamental kept even though lower conf


def test_suppress_harmonics_keeps_when_time_disjoint():
    # stacked in freq but not overlapping in time -> not a harmonic pair
    low = _det(0.0, 1.0, 100, 500)
    high = _det(5.0, 6.0, 550, 1000)
    kept = suppress_harmonics([low, high])
    assert len(kept) == 2


def test_suppress_harmonics_keeps_freq_overlapping_pair():
    # bands overlap heavily (not "entirely above") -> left to overlap merge, not harmonics
    a = _det(0.0, 3.0, 100, 500)
    b = _det(0.0, 3.0, 400, 800)  # 400 < 500 - 50 -> not above by margin
    kept = suppress_harmonics([a, b], freq_margin=50.0)
    assert len(kept) == 2


def test_suppress_harmonics_margin_tolerates_fuzzy_edge():
    # harmonic starts 30Hz below fundamental's top, within 50Hz margin -> still a harmonic
    fundamental = _det(0.0, 3.0, 100, 500)
    harmonic = _det(0.0, 3.0, 470, 900)  # 470 >= 500 - 50
    kept = suppress_harmonics([fundamental, harmonic], freq_margin=50.0)
    assert len(kept) == 1
    assert kept[0]["low_freq"] == 100


def test_suppress_harmonics_empty():
    assert suppress_harmonics([]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/detection/test_postprocess.py -v`
Expected: FAIL with `ImportError: cannot import name 'suppress_harmonics'`

- [ ] **Step 3: Implement `suppress_harmonics`**

Add to `soundbay/detection/postprocess.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/detection/test_postprocess.py -v`
Expected: PASS (all tests through Task 3)

- [ ] **Step 5: Commit**

```bash
git add soundbay/detection/postprocess.py tests/detection/test_postprocess.py
git commit -m "feat(detection): suppress harmonic detections above fundamental"
```

---

### Task 4: Wire filters into `inference.py` with CLI args and raised threshold

**Files:**
- Modify: `soundbay/detection/inference.py`
- Test: `tests/detection/test_postprocess.py` (append one integration-style ordering test that does NOT load a model)

**Interfaces:**
- Consumes: `suppress_overlapping`, `suppress_harmonics`, `merge_detections` (all from `postprocess`).
- Produces: updated `infer_audio(...)` signature adding `overlap_iomin: float = 0.6`, `harmonic_time_overlap: float = 0.6`, `harmonic_freq_margin: float = 50.0`; raised `conf_threshold` default to `0.45`; new CLI args `--overlap-iomin`, `--harmonic-time-overlap`, `--harmonic-freq-margin`, and `--conf` default `0.45`.

- [ ] **Step 1: Write the failing test (pipeline order, model-free)**

This test imports the composition helper we will extract so the ordering logic is testable without a YOLO model. Add:

```python
# append to tests/detection/test_postprocess.py
from soundbay.detection.postprocess import apply_postprocessing


def test_apply_postprocessing_order_overlap_then_harmonics_then_nms():
    # A: full call (4s, 100-500Hz)
    # B: half-call box inside A (2s, 100-500Hz) -> removed by overlap merge
    # C: harmonic above A (4s, 600-1000Hz) -> removed by harmonic suppression
    # D: distinct later call (5-7s, 100-500Hz) -> kept
    A = _det(0.0, 4.0, 100, 500, conf=0.6)
    B = _det(0.0, 2.0, 100, 500, conf=0.9)
    C = _det(0.0, 4.0, 600, 1000, conf=0.8)
    D = _det(5.0, 7.0, 100, 500, conf=0.7)
    kept = apply_postprocessing(
        [A, B, C, D],
        overlap_iomin=0.6,
        harmonic_time_overlap=0.6,
        harmonic_freq_margin=50.0,
        merge_iou=0.3,
    )
    begins = sorted(d["begin_time"] for d in kept)
    assert begins == [0.0, 5.0]  # only A and D remain
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/detection/test_postprocess.py::test_apply_postprocessing_order_overlap_then_harmonics_then_nms -v`
Expected: FAIL with `ImportError: cannot import name 'apply_postprocessing'`

- [ ] **Step 3: Add `apply_postprocessing` to `postprocess.py`**

```python
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
    result = merge_detections(result, iou_threshold=merge_iou)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/detection/test_postprocess.py::test_apply_postprocessing_order_overlap_then_harmonics_then_nms -v`
Expected: PASS

- [ ] **Step 5: Wire into `inference.py`**

In `soundbay/detection/inference.py`:

Replace the import line:
```python
from soundbay.detection.postprocess import merge_detections, export_to_raven
```
with:
```python
from soundbay.detection.postprocess import apply_postprocessing, export_to_raven
```

Update `infer_audio` signature (lines ~24-31):
```python
def infer_audio(
    model_path: str,
    wav_path: str,
    output_dir: str,
    conf_threshold: float = 0.45,
    iou_threshold: float = 0.4,
    overlap_iomin: float = 0.6,
    harmonic_time_overlap: float = 0.6,
    harmonic_freq_margin: float = 50.0,
    params: dict = None,
):
```

Replace the merge line (`merged = merge_detections(raw_detections, iou_threshold=0.3)`) with:
```python
    merged = apply_postprocessing(
        raw_detections,
        overlap_iomin=overlap_iomin,
        harmonic_time_overlap=harmonic_time_overlap,
        harmonic_freq_margin=harmonic_freq_margin,
        merge_iou=0.3,
    )
```

Update `main()` arg parsing:
```python
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--overlap-iomin", type=float, default=0.6)
    parser.add_argument("--harmonic-time-overlap", type=float, default=0.6)
    parser.add_argument("--harmonic-freq-margin", type=float, default=50.0)
```

And update both `infer_audio(...)` calls in `main()` to pass the new args:
```python
    common = dict(
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        overlap_iomin=args.overlap_iomin,
        harmonic_time_overlap=args.harmonic_time_overlap,
        harmonic_freq_margin=args.harmonic_freq_margin,
    )
    if os.path.isfile(args.input):
        infer_audio(args.model, args.input, args.output_dir, **common)
    else:
        for f in sorted(os.listdir(args.input)):
            if f.lower().endswith(".wav"):
                infer_audio(args.model, os.path.join(args.input, f), args.output_dir, **common)
```

- [ ] **Step 6: Run the full test suite and confirm imports resolve**

Run: `pytest tests/detection/ -v`
Expected: PASS (all postprocess tests)

Run: `python -c "import soundbay.detection.inference"`
Expected: no error (imports resolve)

- [ ] **Step 7: Commit**

```bash
git add soundbay/detection/postprocess.py soundbay/detection/inference.py tests/detection/test_postprocess.py
git commit -m "feat(detection): wire post-processing chain and raise default conf to 0.45"
```

---

### Task 5: Validate on annotated Costa Rica subset

**Files:**
- None committed unless we add a scoring helper; this task is a manual verification step.

**Interfaces:**
- Consumes: the wired `inference.py` CLI.

- [ ] **Step 1: Run inference on the annotated CR test files**

Run (user provides model path and the annotated-subset WAV dir):
```bash
python -m soundbay.detection.inference \
  --model <path/to/best.pt> \
  --input <path/to/annotated_cr_wavs> \
  --output-dir datasets/humpback_cr_inference/postproc_test
```
Expected: one `*_detections.txt` Raven table per WAV, fewer detections than before.
If the local PC fails on YOLO inference, boot the training EC2 and run there.

- [ ] **Step 2: Compare against manual annotations**

Load the new Raven tables and the manual annotation tables; eyeball precision/recall
(and count harmonic/nested duplicates removed). Confirm the `--conf 0.45` default looks
right; adjust if precision/recall on the subset suggests a different value.

- [ ] **Step 3: Record findings**

Note the chosen final thresholds in the design doc's Validation section and, if useful,
save them to project memory for future recall.

---

## Self-Review

**Spec coverage:**
- Harmonic suppression → Task 3. ✓
- Same-call overlap merge (longest-duration wins, tie-break confidence) → Task 2. ✓
- Raised confidence threshold → Task 4 (default 0.45, tunable). ✓
- Pipeline order (conf → overlap → harmonics → NMS) → Task 4 `apply_postprocessing`. ✓
- Filters tunable/skippable via CLI → Task 4 args. ✓ (skip = set threshold to 0 for suppress passes; harmonic/overlap with threshold 0 would over-merge, so "skip" is best done by high thresholds — noted as a possible refinement, not required by spec.)
- Pure functions on dicts, no mutation → Tasks 1-3. ✓
- Unit tests incl. genuine-simultaneous-calls no-merge case → Task 2 `test_suppress_overlapping_keeps_distinct_calls` + Task 3 `test_suppress_harmonics_keeps_when_time_disjoint`. ✓
- Keep-lower-fundamental-despite-higher-confidence → Task 3 `test_suppress_harmonics_drops_higher_band`. ✓
- Validation on annotated subset, EC2 fallback → Task 5. ✓
- No changes to legacy `soundbay/inference.py` → respected throughout. ✓

**Placeholder scan:** No TBD/TODO; all code steps show full code.

**Type consistency:** `compute_iomin`, `compute_time_overlap_ratio`, `suppress_overlapping`, `suppress_harmonics`, `apply_postprocessing` names and signatures are consistent across tasks and the wiring in Task 4.

**Note on "skippable filters":** The spec mentions filters should be skippable for A/B comparison. The plan makes them *tunable* via CLI. True on/off flags (`--no-harmonics` etc.) are a trivial future addition but not implemented here to keep scope tight; the biologist can compare by setting thresholds. Flagged for the user in case explicit flags are wanted.
