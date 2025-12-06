"""
Main training script for the audio object detection pipeline.
Supports both BEATS transformer + detection head and YOLO models.
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

from dclde_2026 import config, dataset, models, losses, utils

if config.USE_WANDB:
    import wandb


class Trainer:
    def __init__(self, model, optimizer, criterion, device, model_type='beats'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_type = model_type.lower()
        
    def _prepare_batch(self, inputs, labels):
        inputs = inputs.to(self.device)
        
        if self.model_type == 'beats':
            return inputs, labels
            
        # YOLO batch preparation
        batch_idx = []
        cls = []
        bboxes = []
        
        for i, label_tensor in enumerate(labels):
            if label_tensor is None or len(label_tensor) == 0:
                continue
            
            # label_tensor is [N, 5] (class, x, y, w, h)
            batch_idx.append(torch.full((len(label_tensor),), i, device=self.device, dtype=torch.float32))
            cls.append(label_tensor[:, 0].to(self.device))
            bboxes.append(label_tensor[:, 1:5].to(self.device))
        
        if len(batch_idx) > 0:
            batch = {
                'batch_idx': torch.cat(batch_idx),
                'cls': torch.cat(cls).unsqueeze(1),
                'bboxes': torch.cat(bboxes)
            }
            # For v8DetectionLoss, it expects a tensor of shape [N, 6] (batch_idx, cls, bbox...) or similar
            # But wait, v8DetectionLoss in Ultralytics expects a "batch" dict containing specific keys, 
            # OR a tensor depending on how it's called.
            # Looking at source code of v8DetectionLoss.__call__:
            # preds = preds[1] if isinstance(preds, tuple) else preds
            # loss = torch.zeros(3, device=self.device)  # box, cls, dfl
            # feats = preds[1] if isinstance(preds, tuple) else preds
            # ...
            # targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
            # 
            # So we should construct that tensor here.
            # The error `split_with_sizes` suggests it's trying to split a tensor `targets` that has size 1 in dim 2.
            # If we pass a dict, maybe it's treated differently?
            # Let's try converting to the tensor format Ultralytics expects: [batch_idx, class_idx, x, y, w, h]
            
            # Reconstruct to expected tensor format
            # Ultralytics v8DetectionLoss expects `targets` to be a tensor if it's not a dict with specific structure?
            # Actually, the error `gt_labels, gt_bboxes = targets.split((1, 4), 2)` implies targets is 3D? [B, N, 5]?
            # No, `targets.split((1, 4), 2)` means split dim 2.
            # If targets is passed as the second argument to criterion(preds, targets), 
            # v8DetectionLoss.__call__(self, preds, batch)
            # So `batch` is `targets`.
            
            # If we look at the error:
            # File "/usr/local/lib/python3.10/dist-packages/ultralytics/utils/loss.py", line 262, in __call__
            # gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            #
            # It seems `targets` is treated as a tensor.
            # If we pass a dict, does it work? 
            # Maybe the issue is how we call it in train loop: `criterion(preds, targets)`
            # If targets is a dict, v8DetectionLoss might use it differently.
            
            # The error implies `targets` has a 3rd dimension (dim=2) of size 1, but we request split of 1+4=5.
            # This suggests `targets` is [B, N, 1] instead of [B, N, 5] or [N, 6].
            
            # Let's conform to what `v8DetectionLoss` likely expects when used standalone.
            # It seems to expect `batch` dict containing 'batch_idx', 'cls', 'bboxes'.
            # BUT, the `__call__` implementation might vary.
            
            # Let's try to provide the tensor it constructs internally:
            # targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
            # and pass THAT as targets.
            
            # batch['batch_idx']: [N]
            # batch['cls']: [N, 1]
            # batch['bboxes']: [N, 4]
            # Result: [N, 6]
            
            return inputs, batch
        else:
            batch = {
                'batch_idx': torch.empty((0), device=self.device),
                'cls': torch.empty((0), device=self.device),
                'bboxes': torch.empty((0, 4), device=self.device)
            }
            return inputs, batch

    def _forward(self, inputs):
        # Simply call the model. The model wrapper handles internal logic for YOLO.
        return self.model(inputs)

    def _unpack_loss(self, loss, details):
        if self.model_type == 'beats':
            return {
                'total_loss': loss.item(),
                'box_loss': details['box_loss'],
                'obj_loss': details['obj_loss'],
                'cls_loss': details['cls_loss']
            }
        
        # YOLO loss unpacking
        # details is loss_items tensor [box, cls, dfl]
        loss_dict = {'total_loss': loss.item(), 'box_loss': 0, 'cls_loss': 0, 'dfl_loss': 0}
        if isinstance(details, torch.Tensor):
            details = details.detach()
            if details.numel() >= 3:
                loss_dict['box_loss'] = details[0].item()
                loss_dict['cls_loss'] = details[1].item()
                loss_dict['dfl_loss'] = details[2].item()
        return loss_dict

    def run_epoch(self, loader, epoch=None, is_train=True):
        if is_train:
            self.model.train()
            desc = f"Epoch {epoch+1} Train"
        else:
            self.model.eval()
            desc = "Validation"

        metrics = {'total_loss': 0, 'box_loss': 0, 'obj_loss': 0, 'cls_loss': 0, 'dfl_loss': 0}
        n_batches = 0
        
        pbar = tqdm(loader, desc=desc)
        for batch_idx, (inputs, labels) in enumerate(pbar):
            if inputs.nelement() == 0:
                continue
            
            inputs, targets = self._prepare_batch(inputs, labels)
            
            if is_train:
                self.optimizer.zero_grad()
                
            with torch.set_grad_enabled(is_train):
                preds = self._forward(inputs)
                if is_train and batch_idx == 0:
                     # Debug gradients
                     print(f"Preds type: {type(preds)}")
                     if isinstance(preds, (list, tuple)):
                         print(f"Preds[0] grad_fn: {preds[0].grad_fn if isinstance(preds[0], torch.Tensor) else 'Not Tensor'}")
                     elif isinstance(preds, torch.Tensor):
                         print(f"Preds grad_fn: {preds.grad_fn}")
                     
                     # Check model params
                     params_require_grad = any(p.requires_grad for p in self.model.parameters())
                     print(f"Model has params with requires_grad: {params_require_grad}")
                     if not params_require_grad:
                         print("WARNING: No parameters require gradients!")
                        # Check internal YOLO model if wrapper
                         if hasattr(self.model, 'core_model'):
                             print(f"Core model params: {any(p.requires_grad for p in self.model.core_model.parameters())}")
                
                loss, details = self.criterion(preds, targets)
            
            # Debug/Fix for non-scalar loss
            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.mean()
                if isinstance(details, torch.Tensor) and details.ndim > 1:
                    details = details.mean(dim=0)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss at batch {batch_idx}, skipping...")
                continue
                
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0 if self.model_type == 'yolo' else 1.0)
                self.optimizer.step()
            
            batch_metrics = self._unpack_loss(loss, details)
            for k, v in batch_metrics.items():
                if k in metrics:
                    metrics[k] += v
            n_batches += 1
            
            # Update progress bar
            postfix = {k.split('_')[0]: f"{v / n_batches:.4f}" for k, v in metrics.items() if v > 0}
            pbar.set_postfix(postfix)
            
            if config.DEBUG and batch_idx >= 2:
                break
        
        # Average metrics
        n_batches = max(1, n_batches)
        return {k: v / n_batches for k, v in metrics.items()}


def log_spectrogram_samples(model, val_loader, device, num_samples=4, model_type='beats'):
    """Log spectrogram samples with predictions to wandb"""
    if not config.USE_WANDB: return
    
    model.eval()
    samples_logged = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            if inputs.nelement() == 0 or samples_logged >= num_samples: break
            
            inputs = inputs.to(device)
            # inputs is raw audio [B, 1, T] or [B, T]
            
            # Generate predictions
            preds = model(inputs) 
            
            # Only support YOLO spectrogram logging for now
            if model_type != 'yolo':
                continue
                
            # We need to manually generate spectrogram for visualization
            # since model output is detection results.
            # Re-use the transform from the model if available
            if hasattr(model, 'spec_transform'):
                if inputs.dim() == 2: inputs = inputs.unsqueeze(1)
                
                spec = model.spec_transform(inputs)
                spec = 10 * torch.log10(spec + 1e-10)
                spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
                
                # Resize if needed (same logic as in model.forward)
                if spec.shape[2] != config.TARGET_FREQ_BINS:
                     spec = torch.nn.functional.interpolate(spec, size=(config.TARGET_FREQ_BINS, spec.shape[3]), mode='bilinear', align_corners=False)
                
                # Flip
                spec = torch.flip(spec, [2])
                
                # Spec is [B, 1, F, T]
                # Convert to numpy for plotting
                specs_np = spec.cpu().numpy()
            else:
                continue

            batch_size = inputs.shape[0]
            for i in range(min(batch_size, num_samples - samples_logged)):
                # Get the i-th spectrogram
                spec_img = specs_np[i, 0] # [F, T]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                # origin='upper' (default) puts index 0 at top. 
                # Since we flipped spectrogram, index 0 is MaxHz (Top). 0Hz is Bottom.
                # This matches standard visual intuition.
                ax.imshow(spec_img, aspect='auto', cmap='viridis')
                ax.set_title(f'Sample {samples_logged + 1}')
                
                # Ground truth
                if len(labels) > i and labels[i] is not None:
                    for box in labels[i].cpu().numpy():
                        if len(box) >= 5:
                            x_c, y_c, w, h = box[1:5]
                            rect = plt.Rectangle(((x_c-w/2)*spec_img.shape[1], (y_c-h/2)*spec_img.shape[0]), 
                                               w*spec_img.shape[1], h*spec_img.shape[0], 
                                               linewidth=2, edgecolor='green', facecolor='none')
                            ax.add_patch(rect)
                
                # Predictions
                # preds[i] might be a tensor [N, 6] or similar depending on Ultralytics output format in eval
                # Ultralytics results object is a list of Results objects
                if isinstance(preds, list):
                    res = preds[i]
                    # res.boxes.xywhn or xywh
                    # The boxes should be normalized if we trained with normalized labels?
                    # Ultralytics usually outputs absolute coordinates for inference.
                    # But wait, our custom training loop might affect this?
                    # No, we called self.yolo_model(x) which returns standard Results objects.
                    
                    boxes = res.boxes
                    for box in boxes:
                        # box.xywhn gives normalized x,y,w,h
                        xywhn = box.xywhn[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        
                        if conf > 0.3:
                            x_c, y_c, w, h = xywhn
                            rect = plt.Rectangle(((x_c-w/2)*spec_img.shape[1], (y_c-h/2)*spec_img.shape[0]), 
                                               w*spec_img.shape[1], h*spec_img.shape[0], 
                                               linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
                            ax.add_patch(rect)

                wandb.log({f'spectrograms/sample_{samples_logged + 1}': wandb.Image(fig)})
                plt.close(fig)
                samples_logged += 1


def main():
    parser = argparse.ArgumentParser(description='Train audio object detection model')
    parser.add_argument('--model', type=str, default=None, choices=['beats', 'yolo'])
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    parser.add_argument('--debug', action='store_true')
    # Removed --use-gcs and --offline as we assume online GCS loading now
    
    args = parser.parse_args()
    
    if args.debug:
        config.DEBUG = True
        config.EPOCHS = config.DEBUG_EPOCHS
        print("DEBUG MODE ENABLED")

    # Setup config
    model_type = args.model or config.MODEL_TYPE.lower()
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.EPOCHS = args.epochs
    config.MODEL_TYPE = model_type
    
    # Adjust frequencies
    if model_type == 'beats':
        config.MAX_FREQ_HZ = min(config.MAX_FREQ_HZ, 8000.0)
    elif model_type == 'yolo':
        config.MAX_FREQ_HZ = min(config.MAX_FREQ_HZ, config.SAMPLE_RATE / 2.0)
        
    print(f"Training {model_type.upper()} on fold {args.fold}")
    
    # Data loading
    print("Loading data...")
    df_ann = dataset.create_full_annotation_df()
    df_chips = dataset.create_chip_list(df_ann)
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        df_ann, df_chips, fold_id=args.fold, 
        use_beats=(model_type=='beats'), model_type=model_type
    )

    # WandB
    if config.USE_WANDB and not config.DEBUG:
        # Filter config to remove non-serializable objects like __builtins__ (which contains Ellipsis) and modules
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('__') and not isinstance(v, type(os))}
        wandb.init(project=config.WANDB_PROJECT, name=f"{model_type}_fold{args.fold}", config=config_dict)

    # Model Setup
    print("Initializing model...")
    model = models.get_model(model_type=model_type).to(config.DEVICE)
    
    if model_type == 'beats':
        criterion = losses.DetectionLoss()
    else:
        criterion = model.get_loss_criterion()
        
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Trainer
    trainer = Trainer(model, optimizer, criterion, config.DEVICE, model_type)

    # Loop
    for epoch in range(start_epoch, config.EPOCHS):
        train_metrics = trainer.run_epoch(train_loader, epoch, is_train=True)
        val_metrics = trainer.run_epoch(val_loader, epoch, is_train=False)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss {train_metrics['total_loss']:.4f} | Val Loss {val_metrics['total_loss']:.4f}")
        
        if config.USE_WANDB and not config.DEBUG:
            wandb.log({
                'epoch': epoch + 1,
                'lr': scheduler.get_last_lr()[0],
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()}
            })
            if (epoch + 1) % config.LOG_SPECTROGRAMS_EVERY_N_EPOCHS == 0:
                log_spectrogram_samples(model, val_loader, config.DEVICE, model_type=model_type)

        # Save
        is_best = val_metrics['total_loss'] < best_val_loss
        if is_best: best_val_loss = val_metrics['total_loss']
        
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(state, os.path.join(args.output_dir, 'latest.pth'))
        if is_best: torch.save(state, os.path.join(args.output_dir, 'best.pth'))
        
    if config.USE_WANDB and not config.DEBUG:
        wandb.finish()

if __name__ == "__main__":
    main()
