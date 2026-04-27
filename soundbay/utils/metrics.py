import numpy as np
from sklearn import metrics
from typing import Dict

import torch


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
        self.label_list = np.asarray(label_list)
        self.pred_list = np.asarray(pred_list)
        if isinstance(pred_proba_list, list): # if pred_proba is already a numpy array, we don't need to convert it twice
            self.pred_proba_array = np.asarray(np.concatenate(pred_proba_list))
        elif isinstance(pred_proba_list, np.ndarray):
            self.pred_proba_array = pred_proba_list
        else:
            raise ValueError("pred_proba_list must be a list or numpy array")
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
    
# from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

def calculate_giou(pred_boxes, target_boxes):
    """
    pred_boxes/target_boxes: [N, 4] -> (x, y, w, h)
    """
    # 1. Get standard coordinates (x1, y1, x2, y2)
    p_x1, p_y1 = pred_boxes[:, 0] - pred_boxes[:, 2]/2, pred_boxes[:, 1] - pred_boxes[:, 3]/2
    p_x2, p_y2 = pred_boxes[:, 0] + pred_boxes[:, 2]/2, pred_boxes[:, 1] + pred_boxes[:, 3]/2
    t_x1, t_y1 = target_boxes[:, 0] - target_boxes[:, 2]/2, target_boxes[:, 1] - target_boxes[:, 3]/2
    t_x2, t_y2 = target_boxes[:, 0] + target_boxes[:, 2]/2, target_boxes[:, 1] + target_boxes[:, 3]/2

    # 2. Standard Intersection and Union
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
    t_area = (t_x2 - t_x1) * (t_y2 - t_y1)
    union_area = p_area + t_area - inter_area + 1e-7
    iou = inter_area / union_area

    # 3. GIoU: Find the smallest enclosing box (C)
    c_x1 = torch.min(p_x1, t_x1)
    c_y1 = torch.min(p_y1, t_y1)
    c_x2 = torch.max(p_x2, t_x2)
    c_y2 = torch.max(p_y2, t_y2)
    
    # Area of the enclosing box
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + 1e-7
    
    # GIoU calculation
    giou = iou - (c_area - union_area) / c_area
    
    return giou # Returns values between -1 and 1

class MetricsCalculator2Detection:
    """ 
    class for 2d detection metrics calculation.
    calculating: mAP, Object Count, RMSE of centroid, f1 score
    """
    def __init__(self, target_boxes: np.ndarray, pred_boxes: np.ndarray, 
                 iou_threshold: float = 0.5, n_classes: int = 2):
        """
        Initialize the base metrics calculator.
        Args:
                target_boxes: [N, 6] -> (x, y, w, h, class_id, conf)
                pred_boxes: [N, 6] -> (x, y, w, h, class_id, conf)
                iou_threshold: threshold to consider a prediction as true positive
        """
        self.target_boxes = target_boxes
        self.pred_boxes = pred_boxes
        self.iou_threshold = iou_threshold

        self.num_classes = n_classes
        self.metrics_dict = {
            'global': {},
            'calls': {}
        }

    def _get_class_masks(self, class_id: int) -> tuple:
        """Get masks for specific class."""
        target_mask = self.target_boxes[:, 4] == class_id
        pred_mask = self.pred_boxes[:, 4] == class_id
        return target_mask, pred_mask
    
    def _get_box_mask(self, boxes: np.ndarray) -> np.ndarray:
        return torch.sum(boxes[:, -1] > confidence_threshold, dim=0)
    
    def _get_n_boxes(self, boxes: np.ndarray) -> int:
        """Get the number of boxes in the given array."""
        return np.sum(boxes[:, -1] > self.iou_threshold)
    
    def _calc_box_count(self, target_boxes: np.ndarray, pred_boxes: np.ndarray, threshold = 0.5) -> float:
        """Return 1 if the number of predicted boxes matches the number of target boxes, else return 0."""
        correct_count = []
        for i in range(len(target_boxes)):
                # check the number of boxes with confidence > threshold:
                n_target_boxes = self._get_n_boxes(target_boxes[i], threshold)
                n_pred_boxes = self._get_n_boxes(pred_boxes[i], threshold)
                correct_count.append(n_target_boxes == n_pred_boxes)
        return np.mean(np.array(correct_count, dtype=np.int8))
    
    def _get_giou(self, pred_boxes: np.ndarray, target_boxes: np.ndarray, confidence_threshold: float = 0.5) -> np.ndarray:
        batch_size = pred_boxes.shape[0]
        giou_scores = []
        
        for i in range(batch_size):
            # Get mask for valid boxes
            pred_mask = self._get_box_mask(pred_boxes[i], confidence_threshold)
            target_mask = self._get_box_mask(target_boxes[i], confidence_threshold)
            
            # Get valid boxes
            valid_pred_boxes = pred_boxes[i][pred_mask][:, :4]  # (x, y, w, h)
            valid_target_boxes = target_boxes[i][target_mask][:, :4]  # (x, y, w, h)
            
            if valid_pred_boxes.shape[0] == 0 or valid_target_boxes.shape[0] == 0:
                giou_scores.append(torch.tensor(0.0))  # No valid boxes, GIoU is 0
                continue
            
            # Calculate GIoU for each pair of valid boxes and take the mean
            giou_sum = 0.0
            count = 0
            for p_box in valid_pred_boxes:
                for t_box in valid_target_boxes:
                    giou_sum += calculate_giou(p_box.unsqueeze(0), t_box.unsqueeze(0)).item()
                    count += 1
            
            avg_giou = giou_sum / count if count > 0 else 0.0
            giou_scores.append(torch.tensor(avg_giou))
        
        return torch.stack(giou_scores)
    
    def _calc_avg_giou(self, target_boxes: np.ndarray, pred_boxes: np.ndarray) -> float:
        """Calculate average GIoU across the batch."""
        giou_scores = self._get_giou(pred_boxes, target_boxes)
        return giou_scores.mean().item()
    
    def _calc_rmse_of_centroids(self, target_boxes: np.ndarray, pred_boxes: np.ndarray, confidence_threshold: float = 0.5) -> float:
        batch_size = pred_boxes.shape[0]
        rmse_sum = 0.0
        count = 0
        
        for i in range(batch_size):
            # Get valid boxes based on confidence threshold
            pred_mask = self._get_box_mask(pred_boxes[i], confidence_threshold)
            target_mask = self._get_box_mask(target_boxes[i], confidence_threshold)
            
            valid_pred_boxes = pred_boxes[i][pred_mask][:, :4]  # (x, y, w, h)
            valid_target_boxes = target_boxes[i][target_mask][:, :4]  # (x, y, w, h)
            
            if valid_pred_boxes.shape[0] == 0 or valid_target_boxes.shape[0] == 0:
                continue
            
            for p_box in valid_pred_boxes:
                for t_box in valid_target_boxes:
                    rmse_sum += torch.sqrt(torch.mean((p_box[:2] + p_box[2:4] / 2 - t_box[:2] - t_box[2:4] / 2) ** 2)).item()
                    count += 1
        
        rmse = rmse_sum / count if count > 0 else 0.0
        return rmse
    
    def _calc_precision_recall_f1(self, target_boxes: np.ndarray, pred_boxes: np.ndarray, 
                                  confidence_threshold: float = 0.5, giou_threshold: float = 0.5) -> Dict:
        batch_size = pred_boxes.shape[0]
        true_positives = 0
        false_positives = 0
        
        for i in range(batch_size):
            # Get valid boxes based on confidence threshold
            pred_mask = self._get_box_mask(pred_boxes[i], confidence_threshold)
            target_mask = self._get_box_mask(target_boxes[i], confidence_threshold)
            
            valid_pred_boxes = pred_boxes[i][pred_mask][:, :4]  # (x, y, w, h)
            valid_target_boxes = target_boxes[i][target_mask][:, :4]  # (x, y, w, h)
            
            if valid_pred_boxes.shape[0] == 0:
                continue
            
            for p_box in valid_pred_boxes:
                matched = False
                for t_box in valid_target_boxes:
                    giou_score = calculate_giou(p_box.unsqueeze(0), t_box.unsqueeze(0)).item()
                    if giou_score >= giou_threshold:
                        true_positives += 1
                        matched = True
                        break
                if not matched:
                    false_positives += 1
        
        precision = true_positives / (true_positives + false_positives + 1e-7)
        recall = true_positives / (true_positives + false_positives + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1': f1}

    def _calc_base_metrics(self, target_boxes: np.ndarray, pred_boxes: np.ndarray, 
                           confidence_threshold: float = 0.5, giou_threshold: float = 0.5) -> Dict:
        f1_group = self._calc_precision_recall_f1(target_boxes, pred_boxes, confidence_threshold, giou_threshold)
        return {
            'precision': f1_group['precision'],
            'recall': f1_group['recall'],
            'f1': f1_group['f1'],
            'avg_giou': self._calc_avg_giou(target_boxes, pred_boxes, confidence_threshold),
            'rmse_centroid': self._calc_rmse_of_centroids(target_boxes, pred_boxes, confidence_threshold),
            'box_count_accuracy': self._calc_box_count(target_boxes, pred_boxes, confidence_threshold)
        }

    def _calc_global_metrics(self, confidence_threshold: float = 0.5, giou_threshold: float = 0.5) -> None:
        precision_all = []
        recall_all = []
        f1_all = []
        giou_all = []
        rmse_all = []
        box_count_all = []
        for i in range(1, self.num_classes):
            target_mask, pred_mask = self._get_class_masks(i)
            target_boxes_class = self.target_boxes[target_mask]
            pred_boxes_class = self.pred_boxes[pred_mask]
            class_metrics = self._calc_base_metrics(target_boxes_class, 
                                                    pred_boxes_class, 
                                                    confidence_threshold, 
                                                    giou_threshold)
            precision_all.append(class_metrics['precision'])
            recall_all.append(class_metrics['recall'])
            f1_all.append(class_metrics['f1'])
            giou_all.append(class_metrics['avg_giou'])
            rmse_all.append(class_metrics['rmse_centroid'])
            box_count_all.append(class_metrics['box_count_accuracy'])
        
        self.metrics_dict['global']['call_precision_macro'] = np.nanmean(precision_all)
        self.metrics_dict['global']['call_recall_macro'] = np.nanmean(recall_all)
        self.metrics_dict['global']['call_f1_macro'] = np.nanmean(f1_all)
        self.metrics_dict['global']['call_avg_giou_macro'] = np.nanmean(giou_all)
        self.metrics_dict['global']['call_rmse_centroid_macro'] = np.nanmean(rmse_all)
        self.metrics_dict['global']['call_box_count_accuracy_macro'] = np.nanmean(box_count_all)
                                                    
    def _calc_class_metrics(self) -> None:
        for class_id in range(1, self.num_classes):
            target_mask, pred_mask = self._get_class_masks(class_id)
            target_boxes_class = self.target_boxes[target_mask]
            pred_boxes_class = self.pred_boxes[pred_mask]
            self.metrics_dict['calls'][class_id] = self._calc_base_metrics(
                target_boxes_class, 
                pred_boxes_class
                )
    
    def calc_all_metrics(self) -> Dict:
        """Calculate all metrics at once."""
        self._calc_global_metrics()
        self._calc_class_metrics()
        return self.metrics_dict