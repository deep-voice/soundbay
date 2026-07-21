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
