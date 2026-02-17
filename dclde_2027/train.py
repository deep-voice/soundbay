import random
import torch
import click
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from pathlib import Path

from config import Config
from model import BioacousticDetector, BioacousticDetectorBEATS, load_state_dict_compat
from dataset import get_dataloaders, get_debug_dataloaders
from local_dataset import get_local_dataloaders, download_dataset
from callbacks import WandbCallback, SampleCollector
from loss_and_metrics import compute_positive_only_loss, compute_distance_weighted_loss, RunningMetrics
from augmentations import MixupAugmentation, get_time_masking


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        config,
        callback=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.callback = callback
        
        self.class_names = list(config.label_map.keys())
        self.best_f1 = 0
        self.global_step = 0
        self.output_dir = None  # Will be set by callback
        
        # Mixup augmentation (simulates overlapping calls)
        self.mixup = MixupAugmentation(
            prob=config.aug_mixup_prob,
            alpha=config.aug_mixup_alpha
        ) if config.use_augmentations else None
        
        # Time masking augmentation (masks time segments and excludes from loss)
        self.time_masking = get_time_masking(config) if config.use_augmentations else None
        
        # Sample collectors for visualizations (separate for train and val)
        # Class-balanced sampling ensures we see examples from all species
        self.train_sample_collector = SampleCollector(
            num_random=config.num_random_samples,
            num_worst=config.num_worst_samples,
            class_names=self.class_names
        )
        self.val_sample_collector = SampleCollector(
            num_random=config.num_random_samples,
            num_worst=config.num_worst_samples,
            class_names=self.class_names
        )
        
        # Log loss type
        if config.loss_type == "distance_weighted":
            print(f"Using distance-weighted loss (decay={config.loss_decay_rate}, min_weight={config.loss_min_weight})")
        else:
            print("Using positive-only loss")
        if getattr(config, 'class_weights', None) is not None:
            print(f"Class weights: {config.class_weights}")
        if getattr(config, 'focal_gamma', 0.0) > 0:
            print(f"Focal loss gamma: {config.focal_gamma}")
        
        # Log time masking status
        if self.time_masking is not None:
            print(f"Time masking: ENABLED (prob={config.time_mask_prob}, max_frames={config.time_mask_max_frames})")
            print("  -> Only masks unannotated regions (preserves all labeled data)")
    
    def compute_loss(self, outputs, targets):
        """Compute loss based on config.loss_type."""
        kwargs = dict(
            class_weights=getattr(self.config, 'class_weights', None),
            focal_gamma=getattr(self.config, 'focal_gamma', 0.0),
        )
        if self.config.loss_type == "distance_weighted":
            return compute_distance_weighted_loss(
                outputs, targets,
                decay_rate=self.config.loss_decay_rate,
                min_weight=self.config.loss_min_weight,
                **kwargs
            )
        else:
            return compute_positive_only_loss(outputs, targets, **kwargs)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        step_loss = 0
        step_count = 0
        
        running_metrics = RunningMetrics(self.config.num_classes)
        self.train_sample_collector.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}", mininterval=30.0)
        for batch in pbar:
            audio = batch['audio'].to(self.device)
            targets = batch['target_frames'].to(self.device)
            
            # Apply Mixup augmentation (blends pairs of samples to simulate overlapping calls)
            # Note: Mixup has its own probability, but respects general aug_prob
            if self.mixup is not None and random.random() < self.config.aug_prob:
                self.mixup.train()
                audio, targets = self.mixup(audio, targets)
            
            # Apply time masking augmentation (only to unannotated regions)
            # This zeros out small segments in background areas, preserving all labeled data
            if self.time_masking is not None and random.random() < self.config.aug_prob:
                self.time_masking.train()
                audio = self.time_masking(audio, targets, self.config.num_output_frames)
            
            self.optimizer.zero_grad()
            outputs = self.model(audio)
            
            # Compute loss (based on config.loss_type)
            loss, per_sample_loss, positive_mask = self.compute_loss(outputs, targets)

            # Skip backward/step on NaN/Inf to avoid corrupting parameters
            if not (torch.isfinite(loss).item()):
                if getattr(self, "_nan_skip_count", 0) < 5:
                    print(f"\n  [WARN] Skipping batch at step {self.global_step}: loss={loss.item()}")
                    self._nan_skip_count = getattr(self, "_nan_skip_count", 0) + 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)
            self.optimizer.step()

            with torch.no_grad():
                running_metrics.update(outputs, targets, self.config.threshold, positive_mask)

            total_loss += loss.item()
            step_loss += loss.item()
            num_batches += 1
            step_count += 1
            self.global_step += 1
            
            self.train_sample_collector.add_batch(batch, outputs.detach(), per_sample_loss.detach())
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=self.global_step)
            
            # Log every N steps
            if self.global_step % self.config.log_every_n_steps == 0:
                avg_step_loss = step_loss / step_count
                step_metrics = running_metrics.compute()
                
                if self.callback:
                    self.callback.log_step_metrics({
                        'train/loss_step': avg_step_loss,
                        'train/macro_f1_step': step_metrics['macro_f1'],
                    }, step=self.global_step)
                    
                    random_samples = self.train_sample_collector.get_random_samples()
                    worst_samples = self.train_sample_collector.get_worst_samples()
                    if random_samples:
                        self.callback.log_spectrograms(random_samples, self.global_step, prefix="train_step/random", use_step=True)
                    if worst_samples:
                        self.callback.log_spectrograms(worst_samples, self.global_step, prefix="train_step/worst", use_step=True)
                    
                    self.train_sample_collector.reset()
                
                print(f"\n  Step {self.global_step}: Loss={avg_step_loss:.4f}, F1={step_metrics['macro_f1']:.4f}")
                step_loss = 0
                step_count = 0
            
            # Mid-epoch validation and best checkpoint saving
            if self.global_step % self.config.save_best_every_n_steps == 0:
                print(f"\n  [Step {self.global_step}] Running mid-epoch validation...")
                val_metrics = self.validate_quick()
                print(f"  [Step {self.global_step}] Val F1: {val_metrics['macro_f1']:.4f}")
                
                if val_metrics['macro_f1'] > self.best_f1:
                    self.best_f1 = val_metrics['macro_f1']
                    self.save_checkpoint(epoch, total_loss / num_batches, val_metrics, 'best.pt')
                    print(f"  [Step {self.global_step}] New best model saved! F1: {self.best_f1:.4f}")
                
                self.model.train()
        
        # Compute epoch metrics
        metrics = running_metrics.compute()
        metrics['loss'] = total_loss / num_batches
        
        # Log train spectrograms
        if self.callback:
            random_samples = self.train_sample_collector.get_random_samples()
            worst_samples = self.train_sample_collector.get_worst_samples()
            if random_samples:
                self.callback.log_spectrograms(random_samples, epoch, prefix="train/random")
            if worst_samples:
                self.callback.log_spectrograms(worst_samples, epoch, prefix="train/worst")
        
        return metrics
    
    @torch.no_grad()
    def _run_validation(self, collect_samples=False, show_progress=False):
        """Core validation loop, shared by validate() and validate_quick()."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        running_metrics = RunningMetrics(self.config.num_classes)
        
        if collect_samples:
            self.val_sample_collector.reset()
        
        loader = tqdm(self.val_loader, desc="Validating", mininterval=30.0) if show_progress else self.val_loader
        
        for batch in loader:
            audio = batch['audio'].to(self.device)
            targets = batch['target_frames'].to(self.device)
            
            outputs = self.model(audio)
            loss, per_sample_loss, positive_mask = self.compute_loss(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            running_metrics.update(outputs, targets, self.config.threshold, positive_mask)
            
            if collect_samples:
                self.val_sample_collector.add_batch(batch, outputs, per_sample_loss)
        
        if num_batches == 0:
            raise RuntimeError(
                "Validation dataloader is empty (0 batches). "
                "Check that your validation dataset has samples and batch_size <= dataset size."
            )
        
        metrics = running_metrics.compute()
        metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    def validate_quick(self):
        """Quick validation without spectrogram logging (for mid-epoch checkpointing)."""
        return self._run_validation(collect_samples=False, show_progress=False)
    
    def validate(self, epoch):
        """Full validation with spectrogram logging."""
        metrics = self._run_validation(collect_samples=True, show_progress=True)
        
        # Log val spectrograms
        if self.callback:
            random_samples = self.val_sample_collector.get_random_samples()
            worst_samples = self.val_sample_collector.get_worst_samples()
            
            if random_samples:
                self.callback.log_spectrograms(random_samples, epoch, prefix="val/random")
            if worst_samples:
                self.callback.log_spectrograms(worst_samples, epoch, prefix="val/worst")
        
        return metrics
    
    def save_checkpoint(self, epoch, train_loss, val_metrics, filename):
        if self.output_dir is None:
            return
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_metrics': val_metrics,
            'best_f1': self.best_f1,
        }
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        # Sync checkpoint to AWS (callback handles S3 upload)
        if self.callback:
            self.callback.log_checkpoint(checkpoint_path, filename)
    
    def fit(self):
        epochs = self.config.debug_epochs if self.config.debug else self.config.epochs
        
        # Start W&B run and get output directory
        if self.callback:
            self.output_dir = self.callback.on_train_start()
        else:
            self.output_dir = Path(self.config.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for epoch in range(1, epochs + 1):
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}/{epochs}")
                print(f"{'='*60}")
                
                train_metrics = self.train_epoch(epoch)
                print(f"Train Loss: {train_metrics['loss']:.4f} | Macro F1: {train_metrics['macro_f1']:.4f} | Binary F1: {train_metrics['binary_f1']:.4f}")
                for i, name in enumerate(self.class_names):
                    print(f"  {name}: P={train_metrics['precision'][i]:.3f} R={train_metrics['recall'][i]:.3f} F1={train_metrics['f1'][i]:.3f}")
                
                val_metrics = self.validate(epoch)
                print(f"Val Loss: {val_metrics['loss']:.4f} | Macro F1: {val_metrics['macro_f1']:.4f} | Binary F1: {val_metrics['binary_f1']:.4f}")
                for i, name in enumerate(self.class_names):
                    print(f"  {name}: P={val_metrics['precision'][i]:.3f} R={val_metrics['recall'][i]:.3f} F1={val_metrics['f1'][i]:.3f}")
                
                self.scheduler.step()
                
                # Log epoch metrics
                if self.callback:
                    self.callback.log_epoch_metrics(train_metrics, val_metrics, epoch)
                
                # Save last checkpoint
                self.save_checkpoint(epoch, train_metrics['loss'], val_metrics, 'last.pt')
                
                # Save best checkpoint
                if val_metrics['macro_f1'] > self.best_f1:
                    self.best_f1 = val_metrics['macro_f1']
                    self.save_checkpoint(epoch, train_metrics['loss'], val_metrics, 'best.pt')
                    print(f"  New best model saved! F1: {self.best_f1:.4f}")
                
                # Debug mode: check for overfitting
                if self.config.debug:
                    print(f"\n  [DEBUG] Train Loss: {train_metrics['loss']:.4f}")
                    if train_metrics['loss'] < 0.1:
                        print(f"  [DEBUG] ✓ Successfully overfitting! Train loss < 0.1")
                    if train_metrics['macro_f1'] > 0.9:
                        print(f"  [DEBUG] ✓ High F1 achieved on small dataset!")
            
            print(f"\nTraining complete. Best macro F1: {self.best_f1:.4f}")
        
        finally:
            if self.callback:
                self.callback.on_train_end()


@click.command()
@click.option('--debug', is_flag=True, help='Run in debug mode (overfit on small dataset)')
@click.option('--local', is_flag=True, help='Use local pre-extracted samples (faster, auto-downloads if needed)')
def main(debug, local):
    config = Config()
    
    # Override config with CLI arguments
    if debug:
        config.debug = True
    if local:
        config.use_local_data = True
    
    device = torch.device(config.device)
    
    if config.debug:
        print("\n" + "="*60)
        print("DEBUG MODE: Overfitting sanity check")
        print("="*60)
        print(f"Using {config.debug_samples} samples for {config.debug_epochs} epochs")
        print("Expected: train loss should approach 0, F1 should approach 1")
        print("="*60 + "\n")
    
    print("Loading data...")
    if config.debug:
        train_loader, val_loader = get_debug_dataloaders(config)
    elif config.use_local_data:
        # Download data from S3 if not already present
        download_dataset()
        train_loader, val_loader = get_local_dataloaders(config)
    else:
        train_loader, val_loader = get_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    print("Building model...")
    if config.model_type == "beats":
        print(f"  Using BEATS model from checkpoint: {config.beats_checkpoint_path}")
        model = BioacousticDetectorBEATS(config).to(device)
        # Freeze BEATS encoder (pretrained, we only train attention + classifier)
        for param in model.encoder.beats_model.parameters():
            param.requires_grad = False
        print(f"  BEATS encoder frozen (embed_dim={model.encoder.embed_dim})")
    elif config.model_type == "perch":
        print(f"  Using Perch encoder")
        model = BioacousticDetector(config).to(device)
        # Freeze Perch encoder (pretrained, we only train attention + classifier)
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}. Choose 'perch' or 'beats'.")

    # Optional: load weights from checkpoint (e.g. old single-classifier or same architecture)
    if getattr(config, "resume_checkpoint_path", None):
        ckpt_path = config.resume_checkpoint_path
        print(f"  Loading weights from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict, strict = load_state_dict_compat(checkpoint["model_state_dict"], model)
        model.load_state_dict(state_dict, strict=strict)
        print("  Checkpoint loaded (old classifier format remapped if needed).")

    if config.use_augmentations:
        aug_list = ["noise", "gain", "mixup"]
        if config.use_time_masking:
            aug_list.append("time_mask")
        print(f"  Audio augmentations: ENABLED ({' + '.join(aug_list)})")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Note: Loss type is configured via config.loss_type (positive_only or distance_weighted)
    # This handles unannotated frames which might contain unlabeled calls (PU learning)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    epochs = config.debug_epochs if config.debug else config.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.lr * 0.01)
    
    # Setup W&B callback
    callback = WandbCallback(config, project_name=config.wandb_project)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        callback=callback,
    )
    
    trainer.fit()


if __name__ == '__main__':
    main()

