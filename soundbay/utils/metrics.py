import numpy as np
from sklearn import metrics
from typing import Dict


class MetricsCalculator:
    """class for metrics calculators."""

    def __init__(self, label_list: list, pred_list: list, pred_proba_list: list, label_type: str):
        """
        Initialize the base metrics calculator.

        Args:
            label_list: Ground truth labels
            pred_list: Predicted labels
            pred_proba_list: Prediction probabilities array
        """
        self.label_type = label_type
        self.label_list = np.array(label_list)
        self.pred_list = np.array(pred_list)
        self.pred_proba_array = np.array(np.concatenate(pred_proba_list))
        self.num_classes = self.pred_proba_array.shape[1]
        self.metrics_dict = {
            'global': {},
            'calls': {}
        }

    @staticmethod
    def nan_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUC with NaN handling for invalid cases."""
        try:
            return metrics.roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def _calc_average_precision(self, labels: np.ndarray, proba: np.ndarray) -> float:
        """Calculate average precision score."""
        return metrics.average_precision_score(labels, proba)

    def _calc_f1(self, labels: np.ndarray, preds: np.ndarray) -> float:
        """Calculate F1 score."""
        return metrics.f1_score(labels, preds)

    def _calc_precision(self, labels: np.ndarray, preds: np.ndarray) -> float:
        """Calculate precision score."""
        return metrics.precision_score(labels, preds)

    def _calc_recall(self, labels: np.ndarray, preds: np.ndarray) -> float:
        """Calculate recall score."""
        return metrics.recall_score(labels, preds)

    def _calc_base_metrics(self, labels: np.ndarray, preds: np.ndarray, proba: np.ndarray) -> Dict:
        """Calculate basic metrics for any class."""
        return {
            'precision': self._calc_precision(labels, preds),
            'recall': self._calc_recall(labels, preds),
            'f1': self._calc_f1(labels, preds),
            'auc': self.nan_auc(labels, proba),
            'average_precision': self._calc_average_precision(labels, proba)
        }

    def _get_background_mask(self) -> tuple:
        """Get mask for background class."""
        if self.label_type == 'single_label':
            return self.label_list == 0, self.pred_list == 0
        elif self.label_type == 'multi_label':
            return (self.label_list == 0).all(axis=1), (self.pred_list == 0).all(axis=1)
        else:
            raise ValueError(f"Label type {self.label_type} is not supported")

    def _get_class_masks(self, class_id: int) -> tuple:
        """Get masks for specific class."""
        if self.label_type == 'single_label':
            labels = self.label_list == class_id
            preds = self.pred_list == class_id
        elif self.label_type == 'multi_label':
            labels = self.label_list[:, class_id]
            preds = self.pred_list[:, class_id]
        else:
            raise ValueError(f"Label type {self.label_type} is not supported")
        return labels, preds

    def calc_global_metrics(self) -> None:
        """Calculate global metrics."""
        # Get background masks
        bg_labels, bg_preds = self._get_background_mask()

        # Calculate background metrics
        bg_metrics = self._calc_base_metrics(labels=bg_labels, preds=bg_preds, proba=self.pred_proba_array[:, 0])

        # Store background metrics
        for metric, value in bg_metrics.items():
            self.metrics_dict['global'][f'bg_{metric}'] = value

        # Calculate class metrics
        pos_auc_list = []
        ap_list = []

        for i in range(1, self.num_classes):
            class_labels, _ = self._get_class_masks(i)
            pos_auc_list.append(self.nan_auc(class_labels, self.pred_proba_array[:, i]))
            ap_list.append(self._calc_average_precision(class_labels, self.pred_proba_array[:, i]))

        # Store macro metrics
        self.metrics_dict['global']['call_auc_macro'] = np.nanmean(pos_auc_list)
        self.metrics_dict['global']['call_average_precision_macro'] = np.nanmean(ap_list)
        self.metrics_dict['global']['call_f1_macro'] = metrics.f1_score(
            self.label_list.flatten(),
            self.pred_list.flatten(),
            average='macro',
            labels=list(range(1, self.num_classes)) if len(self.label_list.shape) == 1 else [1]
        )

        # Calculate accuracy
        self.metrics_dict['global']['accuracy'] = metrics.accuracy_score(
            self.label_list.flatten(), self.pred_list.flatten()
        )

    def calc_class_metrics(self) -> None:
        """Calculate per-class metrics."""
        for class_id in range(1, self.num_classes):
            class_labels, class_preds = self._get_class_masks(class_id)
            self.metrics_dict['calls'][class_id] = self._calc_base_metrics(
                class_labels,
                class_preds,
                self.pred_proba_array[:, class_id]
            )

    def calc_all_metrics(self) -> Dict:
        """Calculate all metrics at once."""
        self.calc_global_metrics()
        self.calc_class_metrics()
        return self.metrics_dict
