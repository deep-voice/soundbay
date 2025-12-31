import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

from dclde_2026 import config


class SoftNegativeYOLOLoss(nn.Module):
    """Wrapper for YOLO v8 loss with soft negative mining.
    
    Reduces cls_loss weight to be more tolerant of potential unlabeled positives.
    Also applies label smoothing to negative targets.
    """
    def __init__(self, base_loss, cls_loss_scale=0.7, neg_label_smooth=0.05):
        super().__init__()
        self.base_loss = base_loss
        self.cls_loss_scale = cls_loss_scale  # Scale down cls loss
        self.neg_label_smooth = neg_label_smooth  # Soft label for negatives (0 -> 0.05)
        if hasattr(base_loss, 'nc'):
            self.nc = base_loss.nc
    
    def forward(self, preds, batch):
        """Compute loss with soft negative mining."""
        loss, loss_items = self.base_loss(preds, batch)
        
        # Scale down cls_loss component (index 1 in loss_items: [box, cls, dfl])
        if isinstance(loss_items, torch.Tensor) and loss_items.numel() >= 3:
            # Reduce cls_loss contribution to total loss
            cls_reduction = loss_items[1] * (1 - self.cls_loss_scale)
            loss = loss - cls_reduction
            loss_items = loss_items.clone()
            loss_items[1] = loss_items[1] * self.cls_loss_scale
        
        return loss, loss_items


def giou_loss(pred_boxes, target_boxes):
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    
    target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
    
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area
    
    iou = inter_area / (union_area + 1e-7)
    convex_x1 = torch.min(pred_x1, target_x1)
    convex_y1 = torch.min(pred_y1, target_y1)
    convex_x2 = torch.max(pred_x2, target_x2)
    convex_y2 = torch.max(pred_y2, target_y2)
    convex_area = (convex_x2 - convex_x1) * (convex_y2 - convex_y1)
    
    giou = iou - (convex_area - union_area) / (convex_area + 1e-7)
    return 1 - giou


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class DetectionLoss(nn.Module):
    def __init__(self, box_loss_weight=1.0, obj_loss_weight=1.0, cls_loss_weight=1.0):
        super().__init__()
        self.box_loss_weight = box_loss_weight
        self.obj_loss_weight = obj_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.focal_loss = FocalLoss()
    
    def hungarian_match(self, pred_boxes, target_boxes):
        if len(target_boxes) == 0:
            return []
        
        cost_matrix = torch.zeros(len(pred_boxes), len(target_boxes))
        for i, pred_box in enumerate(pred_boxes):
            for j, target_box in enumerate(target_boxes):
                cost_matrix[i, j] = torch.sum(torch.abs(pred_box - target_box))
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        return list(zip(row_ind, col_ind))
    
    def forward(self, preds, targets):
        batch_size = preds.shape[0]
        total_box_loss = total_obj_loss = total_cls_loss = 0
        
        for b in range(batch_size):
            pred = preds[b]
            target = targets[b]
            
            if len(target) == 0:
                total_obj_loss += F.binary_cross_entropy(pred[:, 4:5], torch.zeros_like(pred[:, 4:5]))
                continue
            
            pred_boxes, pred_obj, pred_cls = pred[:, :4], pred[:, 4:5], pred[:, 5:]
            target_cls, target_boxes = target[:, 0].long(), target[:, 1:5]
            
            matches = self.hungarian_match(pred_boxes, target_boxes)
            matched_pred_indices = set()
            
            matched_pred_boxes, matched_target_boxes = [], []
            matched_pred_obj, matched_target_obj = [], []
            matched_pred_cls, matched_target_cls = [], []
            
            for pred_idx, target_idx in matches:
                matched_pred_indices.add(pred_idx)
                matched_pred_boxes.append(pred_boxes[pred_idx])
                matched_target_boxes.append(target_boxes[target_idx])
                matched_pred_obj.append(pred_obj[pred_idx])
                matched_target_obj.append(torch.ones(1, device=pred_obj.device))
                matched_pred_cls.append(pred_cls[pred_idx])
                cls_onehot = torch.zeros(config.NUM_CLASSES, device=pred_cls.device)
                cls_onehot[target_cls[target_idx]] = 1.0
                matched_target_cls.append(cls_onehot)
            
            if len(matched_pred_boxes) > 0:
                matched_pred_boxes = torch.stack(matched_pred_boxes)
                matched_target_boxes = torch.stack(matched_target_boxes)
                matched_pred_obj = torch.cat(matched_pred_obj)
                matched_target_obj = torch.cat(matched_target_obj)
                matched_pred_cls = torch.stack(matched_pred_cls)
                matched_target_cls = torch.stack(matched_target_cls)
                
                total_box_loss += giou_loss(matched_pred_boxes, matched_target_boxes).mean()
                total_obj_loss += F.binary_cross_entropy(matched_pred_obj, matched_target_obj)
                total_cls_loss += self.focal_loss(matched_pred_cls, matched_target_cls)
            
            unmatched_indices = set(range(len(pred_boxes))) - matched_pred_indices
            if len(unmatched_indices) > 0:
                unmatched_obj = pred_obj[list(unmatched_indices)]
                total_obj_loss += F.binary_cross_entropy(unmatched_obj, torch.zeros_like(unmatched_obj))
        
        box_loss = total_box_loss / batch_size
        obj_loss = total_obj_loss / batch_size
        cls_loss = total_cls_loss / batch_size
        
        total_loss = self.box_loss_weight * box_loss + self.obj_loss_weight * obj_loss + self.cls_loss_weight * cls_loss
        
        return total_loss, {
            'box_loss': box_loss.item(),
            'obj_loss': obj_loss.item(),
            'cls_loss': cls_loss.item(),
            'total_loss': total_loss.item()
        }
