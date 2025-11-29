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
            batch_idx.append(torch.full((len(label_tensor),), i, device=self.device))
            cls.append(label_tensor[:, 0].to(self.device))
            bboxes.append(label_tensor[:, 1:5].to(self.device))
        
        if len(batch_idx) > 0:
            batch = {
                'batch_idx': torch.cat(batch_idx),
                'cls': torch.cat(cls).unsqueeze(1),
                'bboxes': torch.cat(bboxes)
            }
        else:
            batch = {
                'batch_idx': torch.tensor([], device=self.device),
                'cls': torch.tensor([], device=self.device),
                'bboxes': torch.tensor([], device=self.device)
            }
            
        return inputs, batch

    def _forward(self, inputs):
        if self.model_type == 'yolo':
            # For YOLO training, we need raw model outputs, not inference results
            return self.model.yolo_model.model(inputs)
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
                loss, details = self.criterion(preds, targets)
            
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
            preds = model(inputs) # Inference mode
            
            # Only support YOLO spectrogram logging for now
            if model_type != 'yolo' or len(inputs.shape) != 4 or inputs.shape[1] != 3:
                continue
                
            batch_size = inputs.shape[0]
            for i in range(min(batch_size, num_samples - samples_logged)):
                spec = inputs[i].cpu().numpy()
                spec_img = np.transpose(spec, (1, 2, 0))
                spec_img = spec_img.mean(axis=2) if spec_img.shape[2] > 1 else spec_img.squeeze()
                
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
                pred = preds[i].cpu().numpy()
                for box in pred[:5]:
                    if len(box) >= 5 and box[4] > 0.3: # obj_conf > 0.3
                        x_c, y_c, w, h = box[:4]
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
        wandb.init(project=config.WANDB_PROJECT, name=f"{model_type}_fold{args.fold}", config=vars(config))

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
