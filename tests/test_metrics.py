import pytest
import numpy as np

from soundbay.utils.logging import Logger


def test_get_metrics_dict():
    labels = np.random.choice([0, 1, 2, 3], 1000).tolist()
    pred = np.random.choice([0, 1, 2, 3], 1000).tolist()
    proba = np.random.rand(1000, 4)
    proba = proba / proba.sum(axis=1, keepdims=True)

    metrics = Logger.get_metrics_dict(labels, pred, proba)

    assert isinstance(metrics, dict)
    assert 'bg_auc' in metrics['global']
    assert 'call_auc_macro' in metrics['global']
    for key in metrics['global'].keys():
        assert isinstance(metrics['global'][key], float)

