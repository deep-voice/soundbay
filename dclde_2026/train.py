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


def train_epoch_beats(model, train_loader, optimizer, criterion, device, epoch):
    """Train BEATS model for one epoch"""
    model.train()
    total_loss = 0
    loss_details = {'box_loss': 0, 'obj_loss': 0, 'cls_loss': 0}
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        if inputs.nelement() == 0:
            continue
        
        inputs = inputs.to(device)
        
        # Debug mode: print input shapes
        if config.DEBUG and batch_idx == 0:
            print(f"\nDEBUG: Batch {batch_idx}")
            print(f"  Input shape: {inputs.shape}")
            print(f"  Labels type: {type(labels)}, num samples: {len(labels) if isinstance(labels, list) else 'N/A'}")
            if isinstance(labels, list) and len(labels) > 0:
                print(f"  First label shape: {labels[0].shape if hasattr(labels[0], 'shape') else 'N/A'}")
        
        optimizer.zero_grad()
        preds = model(inputs)
        
        # Debug mode: print prediction shapes
        if config.DEBUG and batch_idx == 0:
            print(f"  Predictions shape: {preds.shape}")
        
        loss, details = criterion(preds, labels)
        
        if torch.isnan(loss):
            print(f"Warning: NaN loss at batch {batch_idx}, skipping...")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        loss_details['box_loss'] += details['box_loss']
        loss_details['obj_loss'] += details['obj_loss']
        loss_details['cls_loss'] += details['cls_loss']
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'box': f"{details['box_loss']:.4f}",
            'obj': f"{details['obj_loss']:.4f}",
            'cls': f"{details['cls_loss']:.4f}"
        })
        
        # Debug mode: only process a few batches
        if config.DEBUG and batch_idx >= 2:
            break
    
    n_batches = batch_idx + 1 if config.DEBUG else len(train_loader)
    return {
        'total_loss': total_loss / n_batches,
        'box_loss': loss_details['box_loss'] / n_batches,
        'obj_loss': loss_details['obj_loss'] / n_batches,
        'cls_loss': loss_details['cls_loss'] / n_batches
    }


def validate_beats(model, val_loader, criterion, device):
    """Validate BEATS model"""
    model.eval()
    total_loss = 0
    loss_details = {'box_loss': 0, 'obj_loss': 0, 'cls_loss': 0}
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader, desc="Validation")):
            if inputs.nelement() == 0:
                continue
            
            inputs = inputs.to(device)
            preds = model(inputs)
            
            loss, details = criterion(preds, labels)
            
            total_loss += loss.item()
            loss_details['box_loss'] += details['box_loss']
            loss_details['obj_loss'] += details['obj_loss']
            loss_details['cls_loss'] += details['cls_loss']
            batch_count += 1
            
            # Debug mode: only process a few batches
            if config.DEBUG and batch_idx >= 2:
                break
    
    n_batches = batch_count if batch_count > 0 else 1
    return {
        'total_loss': total_loss / n_batches,
        'box_loss': loss_details['box_loss'] / n_batches,
        'obj_loss': loss_details['obj_loss'] / n_batches,
        'cls_loss': loss_details['cls_loss'] / n_batches
    }


def log_spectrogram_samples(model, val_loader, device, num_samples=4, model_type='beats'):
    """Log spectrogram samples with predictions to wandb"""
    if not config.USE_WANDB:
        return
    
    model.eval()
    samples_logged = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            if inputs.nelement() == 0 or samples_logged >= num_samples:
                break
            
            inputs = inputs.to(device)
            preds = model(inputs)
            
            # For YOLO, inputs are spectrograms [B, 3, H, W]
            # For BEATS, inputs are raw audio, skip spectrogram logging
            batch_size = inputs.shape[0]
            
            for i in range(min(batch_size, num_samples - samples_logged)):
                # Check if inputs are spectrograms (YOLO) or audio (BEATS)
                if model_type == 'yolo' and len(inputs.shape) == 4 and inputs.shape[1] == 3:
                    # YOLO: inputs are spectrograms [B, 3, H, W]
                    spec = inputs[i].cpu().numpy()
                    # Convert [3, H, W] to [H, W] by averaging channels
                    spec_img = np.transpose(spec, (1, 2, 0))
                    spec_img = spec_img.mean(axis=2) if spec_img.shape[2] > 1 else spec_img.squeeze()
                else:
                    # BEATS: inputs are raw audio, skip spectrogram logging
                    continue
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(spec_img, aspect='auto', origin='lower', cmap='viridis')
                ax.set_xlabel('Time')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Sample {samples_logged + 1}')
                
                # Add ground truth boxes if available
                if len(labels) > i and labels[i] is not None and len(labels[i]) > 0:
                    gt_boxes = labels[i].cpu().numpy()
                    for box in gt_boxes:
                        if len(box) >= 5:
                            class_id, x_center, y_center, width, height = box[:5]
                            x1 = (x_center - width/2) * spec_img.shape[1]
                            x2 = (x_center + width/2) * spec_img.shape[1]
                            y1 = (y_center - height/2) * spec_img.shape[0]
                            y2 = (y_center + height/2) * spec_img.shape[0]
                            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                linewidth=2, edgecolor='green', facecolor='none')
                            ax.add_patch(rect)
                
                # Add predictions
                pred = preds[i].cpu().numpy()
                for box in pred[:5]:  # Show top 5 predictions
                    if len(box) >= 5:
                        x_center, y_center, width, height, obj_score = box[:5]
                        if obj_score > 0.3:  # Only show confident predictions
                            x1 = (x_center - width/2) * spec_img.shape[1]
                            x2 = (x_center + width/2) * spec_img.shape[1]
                            y1 = (y_center - height/2) * spec_img.shape[0]
                            y2 = (y_center + height/2) * spec_img.shape[0]
                            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                linewidth=2, edgecolor='red', facecolor='none', 
                                                linestyle='--', alpha=0.7)
                            ax.add_patch(rect)
                
                plt.tight_layout()
                wandb.log({f'spectrograms/sample_{samples_logged + 1}': wandb.Image(fig)})
                plt.close(fig)
                
                samples_logged += 1
                if samples_logged >= num_samples:
                    break


def train_yolo(model, data_yaml, epochs, imgsz, batch_size, **kwargs):
    """
    Train YOLO using its native training API.
    This is the recommended way to train YOLO models.
    """
    print("Training YOLO using native ultralytics API...")
    print(f"Data config: {data_yaml}")
    print(f"Epochs: {epochs}, Image size: {imgsz}, Batch size: {batch_size}")
    
    results = model.train_yolo(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        **kwargs
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train audio object detection model')
    parser.add_argument('--model', type=str, default=None, choices=['beats', 'yolo'],
                       help='Model type: beats or yolo (overrides config.MODEL_TYPE)')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=config.WEIGHT_DECAY, help='Weight decay')
    parser.add_argument('--fold', type=int, default=0, help='Fold ID for cross-validation')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default='./checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--data-yaml', type=str, default='dclde_2026/data.yaml',
                       help='Path to data.yaml for YOLO training')
    parser.add_argument('--use-gcs', action='store_true',
                       help='Load YOLO data from GCS on-the-fly instead of preprocessed files')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (smaller dataset, verbose logging, fewer epochs)')
    
    args = parser.parse_args()
    
    # Enable debug mode if requested
    if args.debug:
        config.DEBUG = True
        config.EPOCHS = config.DEBUG_EPOCHS
        print("\n" + "=" * 60)
        print("DEBUG MODE ENABLED")
        print(f"  Max samples per split: {config.DEBUG_MAX_SAMPLES}")
        print(f"  Epochs: {config.DEBUG_EPOCHS}")
        print("=" * 60 + "\n")
    
    # Determine model type
    model_type = args.model or config.MODEL_TYPE.lower()
    if model_type not in ['beats', 'yolo']:
        raise ValueError(f"Invalid model type: {model_type}. Choose 'beats' or 'yolo'.")
    
    # Update config
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.WEIGHT_DECAY = args.weight_decay
    config.EPOCHS = args.epochs
    config.MODEL_TYPE = model_type
    
    # Adjust MAX_FREQ_HZ based on model capabilities
    if model_type == 'beats':
        BEATS_MAX_FREQ_HZ = 8000.0  # BEATS resamples to 16kHz, Nyquist = 8kHz
        if config.MAX_FREQ_HZ > BEATS_MAX_FREQ_HZ:
            print(f"\nWARNING: MAX_FREQ_HZ ({config.MAX_FREQ_HZ}Hz) exceeds BEATS capability ({BEATS_MAX_FREQ_HZ}Hz)")
            print(f"Adjusting MAX_FREQ_HZ to {BEATS_MAX_FREQ_HZ}Hz for BEATS model")
            config.MAX_FREQ_HZ = BEATS_MAX_FREQ_HZ
    elif model_type == 'yolo':
        YOLO_MAX_FREQ_HZ = config.SAMPLE_RATE / 2.0  # YOLO can use full Nyquist frequency
        if config.MAX_FREQ_HZ > YOLO_MAX_FREQ_HZ:
            print(f"\nWARNING: MAX_FREQ_HZ ({config.MAX_FREQ_HZ}Hz) exceeds Nyquist ({YOLO_MAX_FREQ_HZ}Hz)")
            print(f"Adjusting MAX_FREQ_HZ to {YOLO_MAX_FREQ_HZ}Hz")
            config.MAX_FREQ_HZ = YOLO_MAX_FREQ_HZ
    
    print("=" * 60)
    print("Audio Object Detection Training")
    print("=" * 60)
    print(f"Model: {model_type.upper()}")
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Fold: {args.fold}")
    print(f"MAX_FREQ_HZ: {config.MAX_FREQ_HZ}Hz (adjusted for {model_type.upper()})")
    print(f"SAMPLE_RATE: {config.SAMPLE_RATE}Hz")
    print(f"WINDOW_SEC: {config.WINDOW_SEC}s")
    if config.DEBUG:
        print(f"DEBUG MODE: Using {config.DEBUG_MAX_SAMPLES} samples per split")
    print("=" * 60)
    
    # YOLO training
    if model_type == 'yolo':
        # Check if we should load from GCS or use preprocessed data
        use_gcs = args.use_gcs or not os.path.exists(args.data_yaml)
        
        if use_gcs:
            # Load from GCS on-the-fly
            print("\nLoading YOLO data from GCS on-the-fly...")
            df_ann = dataset.create_full_annotation_df()
            df_chips = dataset.create_chip_list(df_ann)
            
            train_loader, val_loader, test_loader = dataset.get_dataloaders(
                df_ann, df_chips,
                fold_id=args.fold,
                use_beats=False,  # YOLO uses spectrograms
                model_type='yolo'  # Enable YOLO-specific processing
            )
            
            print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
            
            model = models.get_model(model_type='yolo').to(config.DEVICE)
            criterion = losses.DetectionLoss(
                box_loss_weight=1.0,
                obj_loss_weight=1.0,
                cls_loss_weight=1.0
            )
            
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
            
            scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
            
            print("\nStarting training...")
            for epoch in range(config.EPOCHS):
                train_metrics = train_epoch_beats(model, train_loader, optimizer, criterion, config.DEVICE, epoch)
                val_metrics = validate_beats(model, val_loader, criterion, config.DEVICE)
                scheduler.step()
                
                print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
                print(f"Train - Loss: {train_metrics['total_loss']:.4f}")
                print(f"Val   - Loss: {val_metrics['total_loss']:.4f}")
            
            print("\nYOLO training complete!")
            return
        else:
            # Use preprocessed data from disk with YOLO's native API
            if not os.path.exists(args.data_yaml):
                print(f"\nERROR: {args.data_yaml} not found!")
                print("Please run preprocessing first:")
                print("  python -m dclde_2026.preprocess_for_yolo")
                print("\nOr use --use-gcs to load from GCS on-the-fly")
                return
            
            model = models.get_model(model_type='yolo')
            train_yolo(
                model=model,
                data_yaml=args.data_yaml,
                epochs=config.EPOCHS,
                imgsz=config.YOLO_IMG_SIZE,
                batch_size=config.BATCH_SIZE
            )
            print("\nYOLO training complete!")
            return
    
    # BEATS training
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\nLoading data...")
    df_ann = dataset.create_full_annotation_df()
    df_chips = dataset.create_chip_list(df_ann)
    
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        df_ann, df_chips,
        fold_id=args.fold,
        use_beats=True,  # BEATS uses raw audio
        model_type=model_type
    )
    
    # Store test chips for later inference
    gkf = GroupKFold(n_splits=config.N_SPLITS)
    splits = list(gkf.split(df_chips, groups=df_chips['group']))
    test_idx = splits[(args.fold + 1) % config.N_SPLITS][1]
    df_test = df_chips.iloc[test_idx].reset_index(drop=True)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize wandb (skip in debug mode to avoid unnecessary logging)
    if config.USE_WANDB and not config.DEBUG:
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=f"{model_type}_fold{args.fold}",
            config={
                'model_type': model_type,
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
                'weight_decay': config.WEIGHT_DECAY,
                'epochs': config.EPOCHS,
                'fold': args.fold,
                'sample_rate': config.SAMPLE_RATE,
                'window_sec': config.WINDOW_SEC,
                'num_classes': config.NUM_CLASSES,
            }
        )
    elif config.DEBUG:
        print("DEBUG MODE: Skipping wandb initialization")
    
    print("\nInitializing model...")
    model = models.get_model(model_type=model_type).to(config.DEVICE)
    
    criterion = losses.DetectionLoss(
        box_loss_weight=1.0,
        obj_loss_weight=1.0,
        cls_loss_weight=1.0
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    
    print("\nStarting training...")
    for epoch in range(start_epoch, config.EPOCHS):
        train_metrics = train_epoch_beats(model, train_loader, optimizer, criterion, config.DEVICE, epoch)
        val_metrics = validate_beats(model, val_loader, criterion, config.DEVICE)
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
              f"Box: {train_metrics['box_loss']:.4f}, "
              f"Obj: {train_metrics['obj_loss']:.4f}, "
              f"Cls: {train_metrics['cls_loss']:.4f}")
        print(f"Val   - Loss: {val_metrics['total_loss']:.4f}, "
              f"Box: {val_metrics['box_loss']:.4f}, "
              f"Obj: {val_metrics['obj_loss']:.4f}, "
              f"Cls: {val_metrics['cls_loss']:.4f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Log metrics to wandb (skip in debug mode)
        if config.USE_WANDB and not config.DEBUG:
            log_dict = {
                'epoch': epoch + 1,
                'train/total_loss': train_metrics['total_loss'],
                'train/box_loss': train_metrics['box_loss'],
                'train/obj_loss': train_metrics['obj_loss'],
                'train/cls_loss': train_metrics['cls_loss'],
                'val/total_loss': val_metrics['total_loss'],
                'val/box_loss': val_metrics['box_loss'],
                'val/obj_loss': val_metrics['obj_loss'],
                'val/cls_loss': val_metrics['cls_loss'],
                'learning_rate': scheduler.get_last_lr()[0]
            }
            wandb.log(log_dict)
            
            # Log spectrogram samples every N epochs
            if (epoch + 1) % config.LOG_SPECTROGRAMS_EVERY_N_EPOCHS == 0:
                log_spectrogram_samples(model, val_loader, config.DEVICE, num_samples=config.NUM_SPECTROGRAM_SAMPLES, model_type=model_type)
        
        is_best = val_metrics['total_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['total_loss']
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_type': 'beats'
        }
        
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest.pth'))
        
        if is_best:
            torch.save(checkpoint, os.path.join(args.output_dir, 'best.pth'))
            print(f"✓ Saved best model (val loss: {best_val_loss:.4f})")
        
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(args.output_dir, f'epoch_{epoch+1}.pth'))
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 60)
    
    if config.USE_WANDB and not config.DEBUG:
        wandb.finish()
    
    # Run test inference and save to Raven format (skip in debug mode)
    if not config.DEBUG:
        print("\nRunning test inference...")
        model.eval()
        test_predictions = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Test inference"):
                if inputs.nelement() == 0:
                    continue
                
                inputs = inputs.to(config.DEVICE)
                preds = model(inputs)
                
                # Convert to numpy and store
                for i in range(preds.shape[0]):
                    test_predictions.append(preds[i].cpu().numpy())
        
        # Save test results to Raven format (full version)
        if test_predictions:
            output_path = os.path.join(args.output_dir, 'test_predictions_raven.txt')
            raven_df = utils.convert_predictions_to_raven(
                test_predictions,
                df_test,
                df_ann,
                confidence_threshold=0.5,
                output_path=output_path,
                simplified=False
            )
            print(f"\nTest inference complete!")
            print(f"Total predictions (full): {len(raven_df)}")
            print(f"Results saved to: {output_path}")
            
            # Save simplified version (time only, full frequency range)
            output_path_simplified = os.path.join(args.output_dir, 'test_predictions_raven_simplified.txt')
            raven_df_simplified = utils.convert_predictions_to_raven(
                test_predictions,
                df_test,
                df_ann,
                confidence_threshold=0.5,
                output_path=output_path_simplified,
                simplified=True
            )
            print(f"Total predictions (simplified): {len(raven_df_simplified)}")
            print(f"Simplified results saved to: {output_path_simplified}")


if __name__ == "__main__":
    main()
