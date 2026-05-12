import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import box_iou, distance_box_iou_loss

# def calculate_iou(pred_boxes, target_boxes):
#     # Standard x1, y1, x2, y2 conversion
#     p_x1, p_y1 = pred_boxes[:, 0], pred_boxes[:, 1]
#     p_x2, p_y2 = pred_boxes[:, 0] + pred_boxes[:, 2], pred_boxes[:, 1] + pred_boxes[:, 3]

#     t_x1, t_y1 = target_boxes[:, 0], target_boxes[:, 1]
#     t_x2, t_y2 = target_boxes[:, 0] + target_boxes[:, 2], target_boxes[:, 1] + target_boxes[:, 3]

#     inter_x1 = torch.max(p_x1, t_x1)
#     inter_y1 = torch.max(p_y1, t_y1)
#     inter_x2 = torch.min(p_x2, t_x2)
#     inter_y2 = torch.min(p_y2, t_y2)
    
#     inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
#     union_area = (p_x2 - p_x1)*(p_y2 - p_y1) + (t_x2 - t_x1)*(t_y2 - t_y1) - inter_area + 1e-7
    
#     return inter_area / union_area
def calculate_diou(pred_boxes, target_boxes):
    # 1. Get corners
    p_x1, p_y1 = pred_boxes[:, 0], pred_boxes[:, 1]
    p_x2, p_y2 = p_x1 + pred_boxes[:, 2], p_y1 + pred_boxes[:, 3]
    t_x1, t_y1 = target_boxes[:, 0], target_boxes[:, 1]
    t_x2, t_y2 = t_x1 + target_boxes[:, 2], t_y1 + target_boxes[:, 3]

    # 2. Intersection
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    union_area = (p_x2 - p_x1)*(p_y2 - p_y1) + (t_x2 - t_x1)*(t_y2 - t_y1) - inter_area + 1e-7
    iou = inter_area / union_area

    # 3. DIoU Term: Center Distance
    p_cx, p_cy = (p_x1 + p_x2) / 2, (p_y1 + p_y2) / 2
    t_cx, t_cy = (t_x1 + t_x2) / 2, (t_y1 + t_y2) / 2
    center_distance = (p_cx - t_cx)**2 + (p_cy - t_cy)**2

    # 4. Enclosing box (C)
    c_x1 = torch.min(p_x1, t_x1)
    c_y1 = torch.min(p_y1, t_y1)
    c_x2 = torch.max(p_x2, t_x2)
    c_y2 = torch.max(p_y2, t_y2)
    diag_distance = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2 + 1e-7

    return iou - (center_distance / diag_distance)

def calculate_giou(pred_boxes, target_boxes):
    """
    pred_boxes/target_boxes: [N, 4] -> (x, y, w, h)
    """

    # Assuming [x, y, w, h] where x,y is Top-Left
    p_x1, p_y1 = pred_boxes[:, 0], pred_boxes[:, 1]
    p_x2, p_y2 = pred_boxes[:, 0] + pred_boxes[:, 2], pred_boxes[:, 1] + pred_boxes[:, 3]

    t_x1, t_y1 = target_boxes[:, 0], target_boxes[:, 1]
    t_x2, t_y2 = target_boxes[:, 0] + target_boxes[:, 2], target_boxes[:, 1] + target_boxes[:, 3]

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
    def __init__(self, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=5.0, lambda_class=1.0):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.bce = nn.BCEWithLogitsLoss(reduction='none') # for confidence loss
        self.ce = nn.CrossEntropyLoss(reduction='none') # for class loss

    def forward(self, predictions, targets):
        # predictions/targets: [batch, 3, 5 + n_classes] 
        # structure: [x, y, w, h, class_0...class_n, conf]
        
        obj_mask = targets[:, :, -1] == 1
        noobj_mask = targets[:, :, -1] == 0

        # --- 1. GIoU Loss (Coordinates) ---
        iou_loss = torch.tensor(0.0, device=predictions.device)
        if obj_mask.any():
            # ACTIVATION: Use Sigmoid to bring raw coords to [0, 1] range
            pos_preds = torch.sigmoid(predictions[obj_mask][:, :4])

            pos_targets = targets[obj_mask][:, :4]

            # print(f"pos_preds: {pos_preds[0].cpu().detach().numpy()}, pos_targets: {pos_targets[0].cpu().detach().numpy()}")
            
            # FOR TEST CHANGE TO MSE LOSS TO CHECK IF GIOU LOSS IS WORKING PROPERLY
            iou_values = calculate_diou(pos_preds, pos_targets)
            iou_loss = (1 - iou_values).mean()
            # mse_loss = F.mse_loss(pos_preds, pos_targets, reduction='mean')

            # iou_loss = (1.0 * iou_loss) + (0.0 * mse_loss)

        # --- 2. Classification Loss ---
        class_loss = torch.tensor(0.0, device=predictions.device)
        if obj_mask.any():
            # NO ACTIVATION: nn.CrossEntropyLoss expects raw logits 
            # and handles Softmax internally.
            pred_classes = predictions[obj_mask][:, 4:-1] 
            true_classes = targets[obj_mask][:, 4:-1]
            
            target_indices = torch.argmax(true_classes, dim=-1)
            class_loss = self.ce(pred_classes, target_indices).mean()

        # --- 3. Confidence Loss (Objectness) ---
        # NO ACTIVATION: nn.BCEWithLogitsLoss expects raw logits 
        # and handles Sigmoid internally.
        pred_conf_logits = predictions[:, :, -1]
        true_conf = targets[:, :, -1]
        
        raw_conf_loss = self.bce(pred_conf_logits, true_conf)
        loss_obj = (raw_conf_loss * obj_mask).sum() * self.lambda_obj
        loss_noobj = (raw_conf_loss * noobj_mask).sum() * self.lambda_noobj
        
        # --- Total ---
        total_loss = (self.lambda_coord * iou_loss) + \
                     (self.lambda_class * class_loss) + \
                     loss_obj + loss_noobj
        
        return total_loss / predictions.size(0)