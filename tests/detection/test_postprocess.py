from soundbay.detection.postprocess import (
    compute_time_overlap_ratio,
    compute_iomin,
    suppress_overlapping,
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
