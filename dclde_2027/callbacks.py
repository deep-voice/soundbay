"""
Callbacks for training: W&B logging, spectrogram visualization, S3 backup.
"""

import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from torchaudio.transforms import Spectrogram, AmplitudeToDB
from dataclasses import asdict
import wandb


# S3 backup location for checkpoints
S3_CHECKPOINT_BUCKET = "deepvoice-datasets"
S3_CHECKPOINT_PREFIX = "dclde_2027_samples/checkpoints"


# Colors for each class
CLASS_COLORS = {
    'KW': '#FF6B6B',   # Red
    'HW': '#4ECDC4',   # Teal
    'AB': '#FFE66D',   # Yellow
    'UndBio': '#95E1D3' # Mint
}


class SpectrogramVisualizer:
    """Visualize spectrograms with predictions and ground truth."""
    
    def __init__(self, config):
        self.config = config
        self.class_names = list(config.label_map.keys())
        self.spec_transform = torch.nn.Sequential(
            Spectrogram(n_fft=2048, hop_length=512),
            AmplitudeToDB(top_db=80)
        )
    
    def plot_spectrogram_with_predictions(
        self,
        waveform,
        target_frames,
        pred_frames,
        window_start,
        window_end,
        gcs_url,
        loss_value=None,
    ):
        """
        Plot spectrogram with prediction and ground truth overlays.
        
        Returns:
            fig: matplotlib figure
        """
        # Ensure mono
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Compute spectrogram
        spec = self.spec_transform(waveform.cpu())
        spec_db = spec.squeeze().numpy()
        
        # Create figure with GridSpec for proper alignment
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, width_ratios=[50, 1], height_ratios=[4, 1, 1], 
                              hspace=0.15, wspace=0.02)
        
        ax_spec = fig.add_subplot(gs[0, 0])
        ax_cbar = fig.add_subplot(gs[0, 1])
        ax_gt = fig.add_subplot(gs[1, 0], sharex=ax_spec)
        ax_pred = fig.add_subplot(gs[2, 0], sharex=ax_spec)
        
        # Hide the empty subplot spaces for label rows
        fig.add_subplot(gs[1, 1]).axis('off')
        fig.add_subplot(gs[2, 1]).axis('off')
        
        # Plot spectrogram
        time_axis = np.linspace(0, self.config.window_sec, spec_db.shape[1])
        freq_axis = np.linspace(0, self.config.target_sr // 2, spec_db.shape[0])
        
        im = ax_spec.pcolormesh(time_axis, freq_axis, spec_db, shading='auto', cmap='magma')
        ax_spec.set_ylabel('Frequency (Hz)', fontsize=11)
        ax_spec.set_ylim(0, self.config.target_sr // 2)
        ax_spec.set_xlim(0, self.config.window_sec)
        
        # Extract filename from URL
        filename = gcs_url.split('/')[-1] if gcs_url else 'Unknown'
        title = f'{filename}\nWindow: {window_start:.2f}s - {window_end:.2f}s'
        if loss_value is not None:
            title += f' | Loss: {loss_value:.4f}'
        ax_spec.set_title(title, fontsize=12, fontweight='bold')
        
        # Colorbar in dedicated axis
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label('dB', fontsize=10)
        
        # Legend
        patches = [mpatches.Patch(color=CLASS_COLORS[c], label=c) for c in self.class_names]
        ax_spec.legend(handles=patches, loc='upper right', fontsize=9)
        
        # Plot ground truth timeline
        target_np = target_frames.cpu().numpy() if torch.is_tensor(target_frames) else target_frames
        self._plot_label_timeline(ax_gt, target_np, 'Ground Truth')
        
        # Plot predictions timeline
        pred_np = pred_frames.cpu().numpy() if torch.is_tensor(pred_frames) else pred_frames
        self._plot_label_timeline(ax_pred, pred_np, 'Predictions', is_prediction=True)
        
        ax_pred.set_xlabel('Time (seconds)', fontsize=11)
        
        # Hide x-axis labels for top plots
        plt.setp(ax_spec.get_xticklabels(), visible=False)
        plt.setp(ax_gt.get_xticklabels(), visible=False)
        
        return fig
    
    def _plot_label_timeline(self, ax, frames, title, is_prediction=False):
        """Plot label timeline for GT or predictions."""
        num_frames = frames.shape[0]
        frame_times = np.linspace(0, self.config.window_sec, num_frames + 1)
        
        # Convert logits to probabilities if this is a prediction
        if is_prediction:
            # Apply sigmoid to convert logits to probabilities, then clamp to [0, 1]
            if torch.is_tensor(frames):
                frames = torch.sigmoid(frames).cpu().numpy()
            elif isinstance(frames, np.ndarray):
                # Check if values look like logits (outside [0, 1] range)
                if frames.min() < 0 or frames.max() > 1:
                    frames = torch.sigmoid(torch.from_numpy(frames)).numpy()
                # If already in [0, 1], assume they're probabilities
        
        for class_idx, class_name in enumerate(self.class_names):
            class_labels = frames[:, class_idx]
            color = CLASS_COLORS.get(class_name, '#888888')
            
            y_pos = len(self.class_names) - class_idx - 1
            for i, val in enumerate(class_labels):
                # For predictions, use probability as alpha; for GT, use binary
                if is_prediction:
                    # Clamp alpha to [0, 1] to prevent matplotlib errors
                    alpha_val = np.clip(float(val), 0.0, 1.0)
                    if alpha_val > 0.1:  # Show predictions above 0.1 threshold
                        ax.barh(
                            y_pos,
                            frame_times[i+1] - frame_times[i],
                            left=frame_times[i],
                            height=0.8,
                            color=color,
                            alpha=alpha_val
                        )
                else:
                    if val > 0.5:
                        ax.barh(
                            y_pos,
                            frame_times[i+1] - frame_times[i],
                            left=frame_times[i],
                            height=0.8,
                            color=color,
                            alpha=0.9
                        )
        
        ax.set_yticks(range(len(self.class_names)))
        ax.set_yticklabels(self.class_names[::-1], fontsize=10)
        ax.set_xlim(0, self.config.window_sec)
        ax.set_ylim(-0.5, len(self.class_names) - 0.5)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='x', alpha=0.3)


class WandbCallback:
    """Callback for Weights & Biases logging during training."""
    
    def __init__(self, config, project_name="dclde_2026"):
        self.config = config
        self.project_name = project_name
        self.visualizer = SpectrogramVisualizer(config)
        self.run_id = None
        self.artifact_dir = None
        
    def on_train_start(self):
        """Called at the start of training."""
        # Serialize config properly (convert tuples to lists for JSON compatibility)
        config_dict = asdict(self.config)
        for key, value in config_dict.items():
            if isinstance(value, tuple):
                config_dict[key] = list(value)
        
        # Initialize wandb run with robust settings
        run = wandb.init(
            project=self.project_name,
            config=config_dict,
            settings=wandb.Settings(
                console="wrap",  # Redirect mode is more stable than "wrap" with high-frequency output (tqdm)
                _disable_stats=True,  # Disable system stats collection
                start_method="thread",  # Use threading to avoid multiprocessing conflicts with DataLoader
            )
        )
        
        self.run_id = run.id
        
        # Create artifact directory with run_id
        self.artifact_dir = Path(self.config.output_dir) / self.run_id
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"W&B run started: {self.run_id}")
        print(f"Checkpoints will be saved to: {self.artifact_dir}")
        
        return self.artifact_dir
    
    def on_train_end(self):
        """Called at the end of training."""
        # Final S3 sync of all checkpoints
        if self.artifact_dir and self.artifact_dir.exists():
            s3_path = f"s3://{S3_CHECKPOINT_BUCKET}/{S3_CHECKPOINT_PREFIX}/{self.run_id}/"
            try:
                subprocess.run(
                    ["aws", "s3", "sync", str(self.artifact_dir), s3_path, "--quiet"],
                    check=True,
                    capture_output=True,
                )
                print(f"Checkpoints synced to {s3_path}")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to sync checkpoints to S3: {e}")
        
        wandb.finish(quiet=True)  # quiet=True prevents blocking on slow uploads
        print(f"W&B run ended: {self.run_id}")
    
    def log_step_metrics(self, metrics, step):
        """Log metrics at a specific step."""
        wandb.log(metrics, step=step)
    
    def log_epoch_metrics(self, train_metrics, val_metrics, epoch):
        """Log metrics at the end of an epoch."""
        class_names = list(self.config.label_map.keys())
        
        metrics = {'epoch': epoch}
        
        # Train metrics - per class
        metrics['train/loss'] = train_metrics['loss']
        metrics['train/macro_f1'] = train_metrics['macro_f1']
        for i, name in enumerate(class_names):
            metrics[f'train/precision_{name}'] = train_metrics['precision'][i].item()
            metrics[f'train/recall_{name}'] = train_metrics['recall'][i].item()
            metrics[f'train/f1_{name}'] = train_metrics['f1'][i].item()
        
        # Train metrics - binary "call" detection (any class = call)
        metrics['train/binary_precision'] = train_metrics['binary_precision']
        metrics['train/binary_recall'] = train_metrics['binary_recall']
        metrics['train/binary_f1'] = train_metrics['binary_f1']
        
        # Val metrics - per class
        metrics['val/loss'] = val_metrics['loss']
        metrics['val/macro_f1'] = val_metrics['macro_f1']
        for i, name in enumerate(class_names):
            metrics[f'val/precision_{name}'] = val_metrics['precision'][i].item()
            metrics[f'val/recall_{name}'] = val_metrics['recall'][i].item()
            metrics[f'val/f1_{name}'] = val_metrics['f1'][i].item()
        
        # Val metrics - binary "call" detection (any class = call)
        metrics['val/binary_precision'] = val_metrics['binary_precision']
        metrics['val/binary_recall'] = val_metrics['binary_recall']
        metrics['val/binary_f1'] = val_metrics['binary_f1']
        
        wandb.log(metrics)
    
    def log_spectrograms(self, samples, step_or_epoch, prefix="val/random", use_step=False):
        """
        Log spectrogram visualizations to W&B.
        
        Args:
            samples: list of dicts with 'audio', 'target_frames', 'pred_frames', 
                    'window_start', 'window_end', 'gcs_url', 'loss'
            step_or_epoch: current step or epoch number
            prefix: e.g. 'train/random', 'train/worst', 'val/random', 'val/worst'
            use_step: if True, log with step= parameter for proper x-axis alignment
        """
        images = []
        for i, sample in enumerate(samples):
            fig = self.visualizer.plot_spectrogram_with_predictions(
                waveform=sample['audio'],
                target_frames=sample['target_frames'],
                pred_frames=sample['pred_frames'],
                window_start=sample['window_start'],
                window_end=sample['window_end'],
                gcs_url=sample['gcs_url'],
                loss_value=sample.get('loss'),
            )
            
            images.append(wandb.Image(fig, caption=f"{prefix}_{i} (loss: {sample.get('loss', 0):.4f})"))
            plt.close(fig)
        
        if images:
            if use_step:
                wandb.log({f"spectrograms/{prefix}": images}, step=step_or_epoch)
            else:
                wandb.log({f"spectrograms/{prefix}": images, "epoch": step_or_epoch})
    
    def log_checkpoint(self, checkpoint_path, artifact_name):
        """Log a model checkpoint as artifact and backup to S3."""
        # Log to W&B
        artifact = wandb.Artifact(
            name=f"model-{self.run_id}",
            type="model",
            description=f"Model checkpoint: {artifact_name}"
        )
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)
        
        # Backup to S3
        s3_path = f"s3://{S3_CHECKPOINT_BUCKET}/{S3_CHECKPOINT_PREFIX}/{self.run_id}/{artifact_name}"
        try:
            subprocess.run(
                ["aws", "s3", "cp", str(checkpoint_path), s3_path, "--quiet"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to backup checkpoint to S3: {e}")


class SampleCollector:
    """Collect random and worst samples during validation for visualization.
    
    Uses reservoir sampling for random samples (class-balanced) and a bounded heap 
    for worst samples to avoid storing all samples in memory.
    """
    
    def __init__(self, num_random=4, num_worst=4, class_names=None):
        self.num_random = num_random
        self.num_worst = num_worst
        self.class_names = class_names or ['KW', 'HW', 'AB', 'UndBio']
        self.reset()
    
    def reset(self):
        """Reset collectors for new epoch."""
        # Separate reservoir per class for balanced sampling
        self.random_by_class = {cls: [] for cls in self.class_names}
        self.samples_seen_by_class = {cls: 0 for cls in self.class_names}
        self.worst_samples = []   # Min-heap of (loss, sample) for worst samples
        self.total_samples_seen = 0
    
    def _get_classes_present(self, target_frames):
        """Get list of classes that have annotations in this sample."""
        # Check which classes have any positive frames
        class_sums = target_frames.sum(dim=0)  # (num_classes,)
        classes = []
        for idx, count in enumerate(class_sums):
            if count > 0:
                classes.append(self.class_names[idx])
        return classes
    
    def _make_sample(self, batch, preds, losses, i):
        """Create a sample dict from batch index."""
        return {
            'audio': batch['audio'][i].cpu(),
            'target_frames': batch['target_frames'][i].cpu(),
            'pred_frames': preds[i],
            'window_start': batch['window_start'][i].item() if torch.is_tensor(batch['window_start'][i]) else batch['window_start'][i],
            'window_end': batch['window_end'][i].item() if torch.is_tensor(batch['window_end'][i]) else batch['window_end'][i],
            'gcs_url': batch['gcs_url'][i],
            'loss': losses[i].item() if torch.is_tensor(losses[i]) else losses[i],
        }
    
    def add_batch(self, batch, outputs, losses):
        """
        Add batch samples to collector using class-balanced reservoir sampling.
        
        Args:
            batch: dict with 'audio', 'target_frames', 'window_start', 'window_end', 'gcs_url'
            outputs: model outputs (logits)
            losses: per-sample losses
        """
        import heapq
        import random
        
        batch_size = outputs.shape[0]
        preds = torch.sigmoid(outputs).detach().cpu()
        
        # Samples per class for balanced random sampling
        samples_per_class = max(1, self.num_random // len(self.class_names))
        
        for i in range(batch_size):
            loss_val = losses[i].item() if torch.is_tensor(losses[i]) else losses[i]
            self.total_samples_seen += 1
            target_frames = batch['target_frames'][i]
            
            # Class-balanced reservoir sampling: add to reservoir for EACH class present
            classes_present = self._get_classes_present(target_frames)
            for cls in classes_present:
                self.samples_seen_by_class[cls] += 1
                class_seen = self.samples_seen_by_class[cls]
                class_reservoir = self.random_by_class[cls]
                
                if len(class_reservoir) < samples_per_class:
                    class_reservoir.append(self._make_sample(batch, preds, losses, i))
                else:
                    # Reservoir sampling: replace with probability samples_per_class/class_seen
                    j = random.randint(0, class_seen - 1)
                    if j < samples_per_class:
                        class_reservoir[j] = self._make_sample(batch, preds, losses, i)
            
            # Min-heap for worst (highest loss) samples (not class-balanced)
            if len(self.worst_samples) < self.num_worst:
                heapq.heappush(self.worst_samples, (loss_val, self.total_samples_seen, self._make_sample(batch, preds, losses, i)))
            elif loss_val > self.worst_samples[0][0]:
                heapq.heapreplace(self.worst_samples, (loss_val, self.total_samples_seen, self._make_sample(batch, preds, losses, i)))
    
    def get_random_samples(self):
        """Get class-balanced random samples for visualization."""
        # Collect samples from each class reservoir
        samples = []
        for cls in self.class_names:
            samples.extend(self.random_by_class[cls])
        return samples[:self.num_random]  # Cap at num_random
    
    def get_worst_samples(self):
        """Get worst (highest loss) samples for visualization."""
        # Extract samples from heap, sorted by loss descending
        return [item[2] for item in sorted(self.worst_samples, key=lambda x: -x[0])]
