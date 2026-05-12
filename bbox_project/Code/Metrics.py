import numpy as np
from sklearn import metrics
from typing import Dict

import torch

from Code.Losses import calculate_giou


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

class MetricsCalculator2Detection:
    """ 
    class for 2d detection metrics calculation.
    calculating: mAP, Object Count, RMSE of centroid, f1 score
    """
    def __init__(self, target_boxes: torch.Tensor, pred_boxes: torch.Tensor, 
                 iou_threshold: float = 0.5, n_classes: int = 2, 
                 confidence_threshold: float = 0.5):
        """
        Initialize the base metrics calculator.
        Args:
                target_boxes: [N, 6] -> (x, y, w, h, class_id, conf)
                pred_boxes: [N, 6] -> (x, y, w, h, class_id, conf)
                iou_threshold: threshold to consider a prediction as true positive
                confidence_threshold: threshold to consider a prediction as valid
                n_classes: number of classes (including background)
        """
        self.pred_boxes = self._class_and_confidence_activation(pred_boxes)
        self.target_boxes = target_boxes # experct the target boxes to already have a confifence score of 1.0 for valid boxes
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.num_classes = n_classes
        self.metrics_dict = {
            'global': {},
            'calls': {}
        }

    @staticmethod
    def _get_class_logits(boxes: torch.Tensor) -> torch.Tensor:
        """Get class logits from the boxes tensor."""
        return boxes[..., 4:-1]
    
    def _get_confidence_logits(self, boxes: torch.Tensor) -> torch.Tensor:
        """Get confidence logits from the boxes tensor."""
        return boxes[..., -1]

    def _class_and_confidence_activation(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Apply activation function to class and confidence predictions.
        expect boxes to be of shape [Batch, Max_Boxes, x, y, w, h, class_1, ..., class_n, conf]
        """
        class_logits = self._get_class_logits(boxes)
        conf_logits = self._get_confidence_logits(boxes)

        class_proba = torch.softmax(class_logits, dim=-1)
        conf_proba = torch.sigmoid(conf_logits)

        # add sigmoid to the prediction x, y, w, h, as this is how they calculated in loss:
        loc_pred = torch.sigmoid(boxes[..., :4])
        return torch.cat([loc_pred, class_proba, conf_proba.unsqueeze(-1)], dim=-1)

    def _get_class_masks(self, class_id: int) -> tuple:
        """Get masks for specific class."""
        target_mask = self._get_class_logits(self.target_boxes)[..., class_id] > 0.5
        pred_mask = self._get_class_logits(self.pred_boxes)[..., class_id] > 0.5
        return target_mask, pred_mask
    
    def _get_box_mask(self, boxes: torch.Tensor) -> torch.Tensor:
        """Get mask for boxes with confidence > threshold."""
        conf = self._get_confidence_logits(boxes)
        return conf > self.confidence_threshold

    def _get_n_boxes(self, boxes: torch.Tensor) -> int:
        """Get the number of boxes in the given array."""
        return torch.sum(boxes[:, -1] > self.iou_threshold).item()
    
    def _calc_box_count(self, target_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> float:
        """Return 1 if the number of predicted boxes matches the number of target boxes, else return 0."""
        correct_count = []
        for i in range(len(target_boxes)):
                # check the number of boxes with confidence > threshold:
                n_target_boxes = self._get_n_boxes(target_boxes[i])
                n_pred_boxes = self._get_n_boxes(pred_boxes[i])
                correct_count.append(n_target_boxes == n_pred_boxes)
        return torch.mean(torch.tensor(correct_count, dtype=torch.float32)).item()
    
    # @staticmethod
    # def calculate_giou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    #     """
    #     pred_boxes/target_boxes: [N, 4] -> (x, y, w, h)
    #     """
    #     # 1. Get standard coordinates (x1, y1, x2, y2)
    #     p_x1, p_y1 = pred_boxes[:, 0] - pred_boxes[:, 2], pred_boxes[:, 1] - pred_boxes[:, 3]
    #     p_x2, p_y2 = pred_boxes[:, 0] + pred_boxes[:, 2], pred_boxes[:, 1] + pred_boxes[:, 3]
    #     t_x1, t_y1 = target_boxes[:, 0] - target_boxes[:, 2], target_boxes[:, 1] - target_boxes[:, 3]
    #     t_x2, t_y2 = target_boxes[:, 0] + target_boxes[:, 2], target_boxes[:, 1] + target_boxes[:, 3]

    #     # 2. Standard Intersection and Union
    #     inter_x1 = torch.max(p_x1, t_x1)
    #     inter_y1 = torch.max(p_y1, t_y1)
    #     inter_x2 = torch.min(p_x2, t_x2)
    #     inter_y2 = torch.min(p_y2, t_y2)
        
    #     inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    #     p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
    #     t_area = (t_x2 - t_x1) * (t_y2 - t_y1)
    #     union_area = p_area + t_area - inter_area + 1e-7
    #     iou = inter_area / union_area

    #     # 3. GIoU: Find the smallest enclosing box (C)
    #     c_x1 = torch.min(p_x1, t_x1)
    #     c_y1 = torch.min(p_y1, t_y1)
    #     c_x2 = torch.max(p_x2, t_x2)
    #     c_y2 = torch.max(p_y2, t_y2)
        
    #     # Area of the enclosing box
    #     c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + 1e-7
        
    #     # GIoU calculation
    #     giou = iou - (c_area - union_area) / c_area
        
    #     return giou # Returns values between -1 and 1
    
    @staticmethod
    def calculate_iou(box1, box2):
        """
        Calculates IoU between two sets of boxes.
        the boxes are in the format (x, y, w, h) where (x, y) is the left-top corner of the box.
        """
        # 1. Convert (x, y, w, h) to (x1, y1, x2, y2)
        # Box 1
        b1_x1, b1_y1 = box1[0], box1[1]
        b1_x2, b1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        # Box 2
        b2_x1, b2_y1 = box2[0], box2[1]
        b2_x2, b2_y2 = box2[0] + box2[2], box2[1] + box2[3]

        # 2. Find the coordinates of the intersection rectangle
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        # 3. Intersection area (width * height)
        # clamp(0) ensures that if there is no overlap, area is 0
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        intersection = inter_w * inter_h

        # 4. Union area
        # Area = width * height
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection

        # 5. IoU
        return intersection / (union + 1e-6)
    
    def _get_iou(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        batch_size = pred_boxes.shape[0]
        iou_scores = []
        
        for i in range(batch_size):
            # Get mask for valid boxes
            pred_mask = self._get_box_mask(pred_boxes[i])
            target_mask = self._get_box_mask(target_boxes[i])
            
            # Get valid boxes
            valid_pred_boxes = pred_boxes[i][pred_mask][:, :4]  # (x, y, w, h)
            valid_target_boxes = target_boxes[i][target_mask][:, :4]  # (x, y, w, h)
            
            if valid_pred_boxes.shape[0] == 0 or valid_target_boxes.shape[0] == 0:
                iou_scores.append(torch.tensor(0.0))  # No valid boxes, IoU is 0
                continue
            
            # Calculate IoU for each pair of valid boxes and take the mean
            iou_sum = 0.0
            count = 0
            for p_box in valid_pred_boxes:
                for t_box in valid_target_boxes:
                    iou_sum += self.calculate_iou(p_box, t_box).item()
                    count += 1
            
            avg_iou = iou_sum / count if count > 0 else 0.0
            iou_scores.append(torch.tensor(avg_iou))
        
        return torch.stack(iou_scores)
    
    def _calc_avg_iou(self, target_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> float:
        """Calculate average IoU across the batch."""
        iou_scores = self._get_iou(pred_boxes, target_boxes)
        return iou_scores.mean().item()
    
    def _calc_rmse_of_centroids(self, target_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> float:
        batch_size = pred_boxes.shape[0]
        rmse_sum = 0.0
        count = 0
        
        for i in range(batch_size):
            # Get valid boxes based on confidence threshold
            pred_mask = self._get_box_mask(pred_boxes[i])
            target_mask = self._get_box_mask(target_boxes[i])
            
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
    
    # def _calc_precision_recall_f1(self, target_boxes: torch.Tensor, pred_boxes: torch.Tensor,
    #                               iou_threshold: float = 0.5) -> Dict:
    #     batch_size = pred_boxes.shape[0]
    #     true_positives = 0
    #     false_positives = 0
    #     false_negatives = 0
        
    #     for i in range(batch_size):
    #         # Get valid boxes based on confidence threshold
    #         pred_mask = self._get_box_mask(pred_boxes[i])
    #         target_mask = self._get_box_mask(target_boxes[i])
            
    #         valid_pred_boxes = pred_boxes[i][pred_mask][:, :4]  # (x, y, w, h)
    #         valid_target_boxes = target_boxes[i][target_mask][:, :4]  # (x, y, w, h)

    #         # false_negatives = 
            
    #         if valid_pred_boxes.shape[0] == 0: # THIS IS A MISTAKE!!!
    #             if len(valid_target_boxes) > 0:
    #                 false_negatives += valid_target_boxes.shape[0]
    #             continue
            
    #         for p_box in valid_pred_boxes:
    #             matched = False
    #             for t_box in valid_target_boxes:
    #                 iou_score = self.calculate_iou(p_box, t_box).item()
    #                 if iou_score >= iou_threshold:
    #                     true_positives += 1
    #                     matched = True
    #                     break
    #             if not matched:
    #                 false_positives += 1
            
    #         if len(valid_target_boxes) > len(valid_pred_boxes):
    #             false_negatives += valid_target_boxes.shape[0] - true_positives
            
    #     precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    #     recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    #     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    #     return {'precision': precision, 'recall': recall, 'f1': f1}

    def _calc_precision_recall_f1(self, target_boxes: torch.Tensor, pred_boxes: torch.Tensor,
                                  iou_threshold: float = 0.5) -> Dict:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        batch_size = pred_boxes.shape[0]
        
        for i in range(batch_size):
            pred_mask = self._get_box_mask(pred_boxes[i])
            target_mask = self._get_box_mask(target_boxes[i])
            
            valid_pred_boxes = pred_boxes[i][pred_mask][:, :4]
            valid_target_boxes = target_boxes[i][target_mask][:, :4]

            # 1. Handle case with no predictions
            if len(valid_pred_boxes) == 0:
                total_fn += len(valid_target_boxes)
                continue
                
            # 2. Handle case with no targets
            if len(valid_target_boxes) == 0:
                total_fp += len(valid_pred_boxes)
                continue

            # Track which targets have been matched
            targets_matched = torch.zeros(len(valid_target_boxes), dtype=torch.bool)
            
            for p_box in valid_pred_boxes:
                best_iou = 0
                best_target_idx = -1
                
                for t_idx, t_box in enumerate(valid_target_boxes):
                    iou_score = self.calculate_iou(p_box, t_box).item()
                    
                    # Find the best matching target for this prediction
                    if iou_score > best_iou:
                        best_iou = iou_score
                        best_target_idx = t_idx
                
                # If the best match is above threshold AND that target isn't already taken
                if best_iou >= iou_threshold and not targets_matched[best_target_idx]:
                    total_tp += 1
                    targets_matched[best_target_idx] = True
                else:
                    # Prediction matched nothing or matched a target already claimed
                    total_fp += 1
            
            # 3. False Negatives are the targets we never matched
            total_fn += (~targets_matched).sum().item()

        # Final Metric Calculations
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}

    def _calc_base_metrics(self, target_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> Dict:
        f1_group = self._calc_precision_recall_f1(target_boxes, pred_boxes)
        return {
            'precision': f1_group['precision'],
            'recall': f1_group['recall'],
            'f1': f1_group['f1'],
            'avg_iou': self._calc_avg_iou(target_boxes, pred_boxes),
            'rmse_centroid': self._calc_rmse_of_centroids(target_boxes, pred_boxes),
            'box_count_accuracy': self._calc_box_count(target_boxes, pred_boxes)
        }

    def _calc_global_metrics(self) -> None:
        precision_all = []
        recall_all = []
        f1_all = []
        iou_all = []
        rmse_all = []
        box_count_all = []
        for i in range(self.num_classes):
            target_mask, pred_mask = self._get_class_masks(i)
            target_boxes_class = (self.target_boxes * target_mask.unsqueeze(-1))
            pred_boxes_class = (self.pred_boxes * pred_mask.unsqueeze(-1))
            class_metrics = self._calc_base_metrics(target_boxes_class, 
                                                    pred_boxes_class)
            precision_all.append(class_metrics['precision'])
            recall_all.append(class_metrics['recall'])
            f1_all.append(class_metrics['f1'])
            iou_all.append(class_metrics['avg_iou'])
            rmse_all.append(class_metrics['rmse_centroid'])
            box_count_all.append(class_metrics['box_count_accuracy'])
        
        self.metrics_dict['global']['call_precision_macro'] = np.nanmean(precision_all)
        self.metrics_dict['global']['call_recall_macro'] = np.nanmean(recall_all)
        self.metrics_dict['global']['call_f1_macro'] = np.nanmean(f1_all)
        self.metrics_dict['global']['call_avg_iou_macro'] = np.nanmean(iou_all)
        self.metrics_dict['global']['call_rmse_centroid_macro'] = np.nanmean(rmse_all)
        self.metrics_dict['global']['call_box_count_accuracy_macro'] = np.nanmean(box_count_all)
                                                    
    def _calc_class_metrics(self) -> None:
        for class_id in range(self.num_classes):
            target_mask, pred_mask = self._get_class_masks(class_id)
            target_boxes_class = (self.target_boxes * target_mask.unsqueeze(-1))
            pred_boxes_class = (self.pred_boxes * pred_mask.unsqueeze(-1))
            self.metrics_dict['calls'][class_id] = self._calc_base_metrics(
                target_boxes_class, 
                pred_boxes_class
                )
    
    def calc_all_metrics(self) -> Dict:
        """Calculate all metrics at once."""
        self._calc_global_metrics()
        self._calc_class_metrics()
        return self.metrics_dict
    
if __name__ == "__main__":
    # test that the MetricsCalculator2Detection works without errors:
    # tensors are of shape [batch_size, n_boxes, x, y, w, h, class_1, ..., class_n, conf]
    n_classes = 3
    pred_boxes = torch.tensor([[[10, 10, 5, 5, 1, 0.9, 0.05, 0.8],
                                  [20, 20, 5, 5, 2, 0.8, 0.1, 0.6],
                                  [30, 30, 5, 5, 3, 0.7, 0.2, 2.0]],
                                 [[15, 15, 5, 5, 1, 0.85, 0.1, 0.05],
                                  [25, 25, 5, 5, 2, 0.75, 0.15, 0.1],
                                  [35, 35, 5, 5, 3, 0.65, 0.25, 0.1]]])
    print(pred_boxes.shape)
    target_boxes = torch.tensor([[[12, 12, 5, 5, 1, 5, 7, 1.0],
                                [22, 22, 5, 5, 2, 0.7, 0.2, 1.0],
                                [32, 32, 5, 5, 3, 0.6, 0.3, 0.1]],
                               [[14, 14, 5, 5, 1, 0.9, 0.05, 0.05],
                                [24, 24, 5, 5, 2, 0.85, 0.1, 0.05],
                                [34, 34, 5, 5, 3, 0.7, 110, 0.1]]])
    print(target_boxes.shape)

    metrics_calculator = MetricsCalculator2Detection(target_boxes, pred_boxes, confidence_threshold=0.5, iou_threshold=0.5)
    metrics = metrics_calculator.calc_all_metrics()
    print(metrics)