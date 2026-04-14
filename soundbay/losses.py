import torch
import torch.nn as nn
import torch.nn.functional as F

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

class DetectionLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, targets):
        # predictions/targets: [batch, 3, 5] (x, y, w, h, conf)
        obj_mask = targets[:, :, 4] == 1
        noobj_mask = targets[:, :, 4] == 0

        # --- 1. GIoU Loss (Coordinates) ---
        iou_loss = torch.tensor(0.0, device=predictions.device)
        if obj_mask.any():
            # Only calculate GIoU for slots where an object exists
            pos_preds = predictions[obj_mask][:, :4]
            pos_targets = targets[obj_mask][:, :4]
            
            # Use the calculate_giou function from earlier
            giou_values = calculate_giou(torch.sigmoid(pos_preds), pos_targets)
            iou_loss = (1 - giou_values).mean()

        # --- 2. Balanced Confidence Loss ---
        raw_conf_loss = self.bce(predictions[:, :, 4], targets[:, :, 4])
        
        # Apply different weights to 'object' vs 'no-object' slots
        loss_obj = (raw_conf_loss * obj_mask).sum() * self.lambda_obj
        loss_noobj = (raw_conf_loss * noobj_mask).sum() * self.lambda_noobj
        
        # --- Total ---
        total_loss = (self.lambda_coord * iou_loss) + loss_obj + loss_noobj
        
        return total_loss / predictions.size(0)