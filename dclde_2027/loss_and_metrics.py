import torch


# Clamp logits to avoid overflow in BCE and focal weight (keeps loss finite)
LOGIT_CLAMP = 15.0


def _focal_weight(outputs, targets, gamma, max_weight=10.0):
    """Per-element focal weight: (1-p)^gamma for positives, p^gamma for negatives. Capped for stability."""
    p = torch.sigmoid(outputs).clamp(1e-7, 1.0 - 1e-7)
    w = targets * (1 - p).pow(gamma) + (1 - targets) * p.pow(gamma)
    return w.clamp(max=max_weight)


def compute_distance_to_annotations(positive_mask):
    """
    Compute distance (in frames) from each frame to the nearest annotated frame.
    
    Args:
        positive_mask: (batch, frames) boolean mask of annotated frames
    
    Returns:
        distances: (batch, frames) distance to nearest annotation (0 for annotated frames)
    """
    batch_size, num_frames = positive_mask.shape
    device = positive_mask.device
    
    # Frame indices
    frame_idx = torch.arange(num_frames, device=device).unsqueeze(0).expand(batch_size, -1)
    
    distances = torch.full((batch_size, num_frames), float('inf'), device=device)
    
    for i in range(batch_size):
        annotated_indices = torch.where(positive_mask[i])[0]
        if len(annotated_indices) > 0:
            # Distance from each frame to each annotated frame, take minimum
            # Shape: (num_frames, num_annotated)
            dists = torch.abs(frame_idx[i].unsqueeze(1) - annotated_indices.unsqueeze(0).float())
            distances[i] = dists.min(dim=1).values
    
    return distances


def compute_distance_weighted_loss(
    outputs,
    targets,
    decay_rate=0.0,
    min_weight=0.1,
    class_weights=None,
    focal_gamma=0.0,
):
    """
    BCE loss on all frames with distance-based frame weights, optional per-class
    weights, and optional focal weighting. Full multi-label: every (frame, class)
    is included with targets 0 or 1 (no masking of negatives).
    
    - Annotated frames: frame weight = 1.0
    - Unannotated frames: weight = max(min_weight, exp(-decay_rate * distance))
      (decay_rate=0 gives all unannotated frames full weight)
    
    Args:
        outputs: (batch, frames, classes)
        targets: (batch, frames, classes) — 0s and 1s for all entries
        decay_rate: distance decay (0 = no decay, all frames weighted)
        min_weight: minimum weight for distant frames
        class_weights: optional (num_classes,) or list; applied per class
        focal_gamma: if > 0, apply focal weighting (default 0 = plain BCE)
    
    Returns:
        batch_loss, per_sample_loss, positive_mask
    """
    positive_mask = targets.sum(dim=-1) > 0  # (batch, frames)

    # Clamp logits for numerical stability (avoids inf/nan in BCE and focal)
    outputs_safe = outputs.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)

    distances = compute_distance_to_annotations(positive_mask)
    weights = torch.exp(-decay_rate * distances)
    weights = torch.clamp(weights, min=min_weight)
    weights[positive_mask] = 1.0
    weights_expanded = weights.unsqueeze(-1).expand_as(outputs)

    # Per-element BCE on all (frame, class) — full multi-label, no masking of 0s
    per_element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs_safe, targets, reduction='none'
    )

    if focal_gamma > 0:
        per_element_loss = per_element_loss * _focal_weight(outputs_safe, targets, focal_gamma)

    if class_weights is not None:
        device = outputs.device
        cw = class_weights if isinstance(class_weights, torch.Tensor) else torch.tensor(class_weights, dtype=outputs.dtype, device=device)
        cw = cw.view(1, 1, -1).expand_as(outputs)
        per_element_loss = per_element_loss * cw

    weighted_loss = per_element_loss * weights_expanded
    batch_loss = weighted_loss.sum() / (weights_expanded.sum() + 1e-8)
    per_sample_loss = weighted_loss.sum(dim=(1, 2)) / (weights_expanded.sum(dim=(1, 2)) + 1e-8)
    return batch_loss, per_sample_loss, positive_mask


def compute_positive_only_loss(outputs, targets, class_weights=None, focal_gamma=0.0):
    """
    BCE loss only on frames that have annotations. On those frames we use full
    multi-label: all classes are included (targets 0 and 1), no masking of negatives.
    
    Args:
        outputs: (batch, frames, classes)
        targets: (batch, frames, classes)
        class_weights: optional (num_classes,) or list
        focal_gamma: if > 0, apply focal weighting
    
    Returns:
        batch_loss, per_sample_loss, positive_mask
    """
    positive_mask = targets.sum(dim=-1) > 0  # (batch, frames)
    mask_expanded = positive_mask.unsqueeze(-1).expand_as(outputs).float()
    outputs_safe = outputs.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)

    per_element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs_safe, targets, reduction='none'
    )
    if focal_gamma > 0:
        per_element_loss = per_element_loss * _focal_weight(outputs_safe, targets, focal_gamma)
    if class_weights is not None:
        device = outputs.device
        cw = class_weights if isinstance(class_weights, torch.Tensor) else torch.tensor(class_weights, dtype=outputs.dtype, device=device)
        cw = cw.view(1, 1, -1).expand_as(outputs)
        per_element_loss = per_element_loss * cw

    per_element_loss = per_element_loss * mask_expanded
    n = mask_expanded.sum()
    if n > 0:
        batch_loss = per_element_loss.sum() / n
        per_sample_loss = per_element_loss.sum(dim=(1, 2)) / (mask_expanded.sum(dim=(1, 2)) + 1e-8)
    else:
        batch_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs_safe, targets)
        per_sample_loss = torch.zeros(outputs.shape[0], device=outputs.device)
    return batch_loss, per_sample_loss, positive_mask


class RunningMetrics:
    """Accumulates TP/FP/FN statistics across batches for memory-efficient metric computation.
    
    Metrics are computed only on annotated frames (positive-unlabeled setting).
    """
    
    def __init__(self, num_classes, device='cpu'):
        self.num_classes = num_classes
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all running statistics."""
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)
        self.binary_tp = 0
        self.binary_fp = 0
        self.binary_fn = 0
    
    def update(self, outputs, targets, threshold, frame_mask=None):
        """
        Update running statistics from a batch.
        
        Args:
            outputs: Model outputs (batch, frames, classes)
            targets: Target labels (batch, frames, classes)
            threshold: Classification threshold
            frame_mask: Optional (batch, frames) mask for which frames to count.
                       If None, uses all frames with any annotation.
        """
        preds_bool = torch.sigmoid(outputs) > threshold
        targets_bool = targets > 0.5
        
        # Default mask: only frames with annotations
        if frame_mask is None:
            frame_mask = targets.sum(dim=-1) > 0
        
        # Apply mask: only count TP/FP/FN on masked frames
        masked_preds = preds_bool & frame_mask.unsqueeze(-1)
        masked_targets = targets_bool & frame_mask.unsqueeze(-1)
        
        self.tp += (masked_preds & masked_targets).sum(dim=(0, 1)).cpu()
        self.fp += (masked_preds & ~masked_targets).sum(dim=(0, 1)).cpu()
        self.fn += (~masked_preds & masked_targets).sum(dim=(0, 1)).cpu()
        
        # Binary metrics (only on masked frames)
        binary_preds = preds_bool.any(dim=-1) & frame_mask
        binary_targets = targets_bool.any(dim=-1)
        self.binary_tp += (binary_preds & binary_targets).sum().item()
        self.binary_fp += (binary_preds & ~binary_targets).sum().item()
        self.binary_fn += (~binary_preds & binary_targets).sum().item()
    
    def compute(self):
        """Compute precision, recall, F1 from accumulated statistics."""
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        binary_precision = self.binary_tp / (self.binary_tp + self.binary_fp + 1e-8)
        binary_recall = self.binary_tp / (self.binary_tp + self.binary_fn + 1e-8)
        binary_f1 = 2 * binary_precision * binary_recall / (binary_precision + binary_recall + 1e-8)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_f1': f1.mean().item(),
            'binary_precision': binary_precision,
            'binary_recall': binary_recall,
            'binary_f1': binary_f1,
        }

