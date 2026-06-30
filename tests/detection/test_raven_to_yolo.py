import tempfile
import os
from soundbay.detection.raven_to_yolo import convert_raven_to_yolo, parse_raven_file


SAMPLE_RAVEN = (
    "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tAnnotation\n"
    "1\tSpectrogram 1\t1\t2.500\t4.800\t150.0\t1200.0\thumpback\n"
    "2\tSpectrogram 1\t1\t8.100\t10.300\t200.0\t3500.0\thumpback\n"
    "3\tSpectrogram 1\t1\t12.000\t13.500\t80.0\t900.0\tunknown\n"
)


def test_parse_raven_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SAMPLE_RAVEN)
        f.flush()
        annotations = parse_raven_file(f.name)

    assert len(annotations) == 3
    assert annotations[0]["begin_time"] == 2.5
    assert annotations[0]["end_time"] == 4.8
    assert annotations[0]["low_freq"] == 150.0
    assert annotations[0]["high_freq"] == 1200.0
    os.unlink(f.name)


def test_convert_raven_to_yolo_single_chunk():
    """A 15s chunk starting at t=0 should capture annotations 1 and 2."""
    annotations = [
        {"begin_time": 2.5, "end_time": 4.8, "low_freq": 150.0, "high_freq": 1200.0, "label": "humpback"},
        {"begin_time": 8.1, "end_time": 10.3, "low_freq": 200.0, "high_freq": 3500.0, "label": "humpback"},
    ]
    chunk_start = 0.0
    chunk_duration = 15.0
    freq_min = 50.0
    freq_max = 4000.0

    yolo_labels = convert_raven_to_yolo(
        annotations, chunk_start, chunk_duration, freq_min, freq_max, class_id=0
    )

    assert len(yolo_labels) == 2
    for label in yolo_labels:
        parts = label.split()
        assert len(parts) == 5
        assert parts[0] == "0"
        x_c, y_c, w, h = [float(p) for p in parts[1:]]
        assert 0 <= x_c <= 1
        assert 0 <= y_c <= 1
        assert 0 < w <= 1
        assert 0 < h <= 1
