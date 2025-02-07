import pytest
import numpy as np

from soundbay.utils.logging import Logger
from soundbay.utils.metrics import MetricsCalculator


def asserts_on_metric_dict(metrics, no_nan=True):
    assert isinstance(metrics, dict)
    assert 'bg_auc' in metrics['global']
    assert 'call_auc_macro' in metrics['global']
    for key in metrics['global'].keys():
        assert isinstance(metrics['global'][key], float)
        if no_nan:
            assert 0 <= metrics['global'][key] <= 1


def test_metrics_calculator():
    n_samples = 100
    labels = np.random.choice([0, 1, 2, 3], n_samples).tolist()
    pred = np.random.choice([0, 1, 2, 3], n_samples).tolist()
    proba = np.random.rand(n_samples, 4)
    proba = proba / proba.sum(axis=1, keepdims=True)
    label_type = 'single_label'

    metrics = MetricsCalculator(labels, pred, proba, label_type).calc_all_metrics()
    asserts_on_metric_dict(metrics)

    # Test only single class in labels
    labels = [0] * n_samples
    metrics = MetricsCalculator(labels, pred, proba, label_type).calc_all_metrics()
    asserts_on_metric_dict(metrics, no_nan=False)

    # Test only single class in predictions
    labels = np.random.choice([0, 1, 2, 3], n_samples).tolist()
    pred = [0] * n_samples
    metrics = MetricsCalculator(labels, pred, proba, label_type).calc_all_metrics()
    asserts_on_metric_dict(metrics, no_nan=False)

    # Test both labels and pred have single class
    labels = [0] * n_samples
    pred = [2] * n_samples
    metrics = MetricsCalculator(labels, pred, proba, label_type).calc_all_metrics()
    asserts_on_metric_dict(metrics, no_nan=False)
