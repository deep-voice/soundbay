import json
import os
import tempfile

import pytest

from soundbay.detection.spectrogram_generator import (
    DEFAULT_PARAMS,
    PARAM_KEYS,
    load_params_sidecar,
    resolve_params,
    sidecar_path_for_model,
)


def test_default_params_has_all_keys():
    assert set(DEFAULT_PARAMS) == set(PARAM_KEYS)


def test_resolve_params_returns_copy_of_defaults():
    resolved = resolve_params()
    assert resolved == DEFAULT_PARAMS
    resolved["chunk_duration"] = 99.0
    assert DEFAULT_PARAMS["chunk_duration"] != 99.0


def test_resolve_params_overrides_win_over_sidecar():
    sidecar = {"chunk_duration": 5.0, "freq_max": 3000.0}
    resolved = resolve_params(sidecar=sidecar, overrides={"freq_max": 1000.0})
    assert resolved["chunk_duration"] == 5.0  # from sidecar
    assert resolved["freq_max"] == 1000.0  # override wins
    assert resolved["img_size"] == DEFAULT_PARAMS["img_size"]  # untouched default


def test_resolve_params_ignores_none_overrides():
    resolved = resolve_params(overrides={"chunk_duration": None})
    assert resolved["chunk_duration"] == DEFAULT_PARAMS["chunk_duration"]


def test_resolve_params_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown"):
        resolve_params(sidecar={"bogus_key": 1})


def test_sidecar_path_for_model_sits_beside_checkpoint():
    path = sidecar_path_for_model(os.path.join("runs", "weights", "best.pt"))
    assert path == os.path.join("runs", "weights", "best.spectrogram.json")


def test_load_params_sidecar_missing_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        assert load_params_sidecar(os.path.join(tmp, "best.pt")) == {}


def test_load_params_sidecar_reads_json_beside_model():
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "best.pt")
        with open(sidecar_path_for_model(model_path), "w") as f:
            json.dump({"chunk_duration": 5.0, "freq_max": 3000.0}, f)
        assert load_params_sidecar(model_path) == {"chunk_duration": 5.0, "freq_max": 3000.0}
