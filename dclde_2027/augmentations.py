"""
Audio augmentations for bioacoustic detection training.

Augmentations are applied to raw waveforms before the encoder.
"""

import torch
import torch.nn as nn
import random


class AudioAugmentations(nn.Module):
    """
    Waveform augmentations for bioacoustic detection.
    
    Recommended augmentations for whale/marine mammal detection:
    - Gaussian noise: Simulates varying ocean noise conditions
    - Gain: Robustness to recording level variations  
    - Time shift: Already handled by random window positioning in dataset
    
    Note: SpecAugment (time/freq masking) could be added post-spectrogram
    if using a spectrogram-based model, but Perch handles raw waveforms.
    """
    
    def __init__(
        self,
        aug_prob: float = 1.0,  # General probability of attempting any augmentation
        noise_prob: float = 0.5,
        noise_snr_range: tuple = (10, 30),  # SNR in dB
        gain_prob: float = 0.5,
        gain_range: tuple = (0.5, 1.5),  # Multiplicative gain
        time_shift_prob: float = 0.0,  # Disabled by default (dataset handles this)
        time_shift_max: float = 0.2,  # Max shift as fraction of length
    ):
        super().__init__()
        self.aug_prob = aug_prob
        self.noise_prob = noise_prob
        self.noise_snr_range = noise_snr_range
        self.gain_prob = gain_prob
        self.gain_range = gain_range
        self.time_shift_prob = time_shift_prob
        self.time_shift_max = time_shift_max
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to waveform.
        
        Args:
            waveform: (batch, channels, samples) or (batch, samples)
        
        Returns:
            Augmented waveform with same shape
        """
        if not self.training:
            return waveform
        
        # General augmentation probability - skip all augmentations with (1 - aug_prob)
        if random.random() > self.aug_prob:
            return waveform
        
        # Ensure 3D: (batch, channels, samples)
        squeeze_channel = False
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
            squeeze_channel = True
        
        # Apply augmentations (each with its own probability)
        if random.random() < self.gain_prob:
            waveform = self._apply_gain(waveform)
        
        if random.random() < self.noise_prob:
            waveform = self._add_noise(waveform)
        
        if random.random() < self.time_shift_prob:
            waveform = self._time_shift(waveform)
        
        if squeeze_channel:
            waveform = waveform.squeeze(1)
        
        return waveform
    
    def _apply_gain(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random gain (volume change)."""
        gain = random.uniform(*self.gain_range)
        return waveform * gain
    
    def _add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise at random SNR."""
        snr_db = random.uniform(*self.noise_snr_range)
        
        # Compute signal power
        signal_power = waveform.pow(2).mean(dim=-1, keepdim=True)
        
        # Compute noise power for target SNR
        # SNR = 10 * log10(signal_power / noise_power)
        # noise_power = signal_power / 10^(SNR/10)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate noise
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        
        return waveform + noise
    
    def _time_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply circular time shift."""
        num_samples = waveform.shape[-1]
        max_shift = int(num_samples * self.time_shift_max)
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(waveform, shifts=shift, dims=-1)


class MixupAugmentation(nn.Module):
    """
    Mixup augmentation for audio: blends pairs of samples.
    
    This simulates overlapping calls which are common in marine environments
    (multiple animals vocalizing, calls overlapping with boat noise, etc.)
    
    For frame-level multi-label detection, we blend both audio and labels:
        mixed_audio = λ * audio_A + (1-λ) * audio_B
        mixed_labels = max(labels_A, labels_B)  # Union of active frames
    
    Note: We use max (union) instead of weighted average for labels because
    with frame-level detection, if either source has a call at frame t,
    the mixed audio should be labeled as having a call at frame t.
    """
    
    def __init__(self, prob: float = 0.3, alpha: float = 0.4):
        """
        Args:
            prob: Probability of applying mixup to a batch
            alpha: Beta distribution parameter (lower = more extreme mixing)
                   alpha=0.4 gives λ mostly in [0.6, 1.0] range
                   alpha=1.0 gives uniform λ in [0, 1]
        """
        super().__init__()
        self.prob = prob
        self.alpha = alpha
    
    def forward(self, audio: torch.Tensor, targets: torch.Tensor):
        """
        Apply mixup to audio and targets.
        
        Args:
            audio: (batch, samples) or (batch, channels, samples)
            targets: (batch, frames, classes)
        
        Returns:
            mixed_audio, mixed_targets
        """
        if not self.training or random.random() > self.prob:
            return audio, targets
        
        batch_size = audio.shape[0]
        if batch_size < 2:
            return audio, targets
        
        # Sample mixing coefficient from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 so original dominates
        
        # Random permutation for pairing
        indices = torch.randperm(batch_size, device=audio.device)
        
        # Mix audio
        mixed_audio = lam * audio + (1 - lam) * audio[indices]
        
        # Mix targets: use element-wise max (union of labels)
        # This ensures if either source has a call, the mixed sample is labeled
        mixed_targets = torch.max(targets, targets[indices])
        
        return mixed_audio, mixed_targets


class SpecAugment(nn.Module):
    """
    SpecAugment: time and frequency masking for spectrograms.
    
    Apply this AFTER computing spectrogram if you want spectrogram-level augmentation.
    Note: This is separate from AudioAugmentations since Perch takes raw waveforms.
    
    Reference: Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR"
    """
    
    def __init__(
        self,
        freq_mask_prob: float = 0.5,
        freq_mask_max: int = 20,  # Max frequency bins to mask
        time_mask_prob: float = 0.5,
        time_mask_max: int = 50,  # Max time bins to mask
        num_masks: int = 2,  # Number of masks to apply
    ):
        super().__init__()
        self.freq_mask_prob = freq_mask_prob
        self.freq_mask_max = freq_mask_max
        self.time_mask_prob = time_mask_prob
        self.time_mask_max = time_mask_max
        self.num_masks = num_masks
    
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: (batch, freq_bins, time_bins) or (batch, channels, freq_bins, time_bins)
        
        Returns:
            Augmented spectrogram
        """
        if not self.training:
            return spectrogram
        
        spec = spectrogram.clone()
        
        # Handle different input shapes
        if spec.dim() == 3:
            _, freq_bins, time_bins = spec.shape
        else:
            _, _, freq_bins, time_bins = spec.shape
        
        # Frequency masking
        if random.random() < self.freq_mask_prob:
            for _ in range(self.num_masks):
                f = random.randint(1, min(self.freq_mask_max, freq_bins - 1))
                f0 = random.randint(0, freq_bins - f)
                if spec.dim() == 3:
                    spec[:, f0:f0+f, :] = 0
                else:
                    spec[:, :, f0:f0+f, :] = 0
        
        # Time masking
        if random.random() < self.time_mask_prob:
            for _ in range(self.num_masks):
                t = random.randint(1, min(self.time_mask_max, time_bins - 1))
                t0 = random.randint(0, time_bins - t)
                if spec.dim() == 3:
                    spec[:, :, t0:t0+t] = 0
                else:
                    spec[:, :, :, t0:t0+t] = 0
        
        return spec


class WaveformTimeMasking(nn.Module):
    """
    Annotation-aware time masking augmentation for waveforms.
    
    Zeros out random time segments of the waveform, but ONLY in regions
    that do not contain annotations. This preserves all labeled data while
    adding regularization to background/negative regions.
    
    NOTE: This creates unnatural silence gaps in the audio. Consider using
    other regularizations (dropout, weight decay) instead, or modify this
    to replace masked regions with noise rather than silence for more
    natural augmentation.
    
    This simulates varying noise conditions and teaches the model to be
    robust to gaps in the signal, without losing any annotated events.
    """
    
    def __init__(
        self,
        prob: float = 0.5,
        max_mask_frames: int = 5,  # Max frames to mask per mask (small periods)
        num_masks: int = 3,  # Number of masks to attempt
    ):
        super().__init__()
        self.prob = prob
        self.max_mask_frames = max_mask_frames
        self.num_masks = num_masks
    
    def forward(self, waveform: torch.Tensor, targets: torch.Tensor, num_output_frames: int):
        """
        Apply time masking to waveform, only in unannotated regions.
        
        Args:
            waveform: (batch, samples) or (batch, channels, samples)
            targets: (batch, frames, classes) - annotation targets
            num_output_frames: Number of output frames (e.g., 64 for Perch).
        
        Returns:
            masked_waveform: Same shape as input, with masked regions zeroed
        """
        if not self.training or random.random() > self.prob:
            return waveform
        
        batch_size = waveform.shape[0]
        num_samples = waveform.shape[-1]
        samples_per_frame = num_samples / num_output_frames
        
        # Find unannotated frames: frames with no positive labels
        # targets shape: (batch, frames, classes)
        has_annotation = targets.sum(dim=-1) > 0  # (batch, frames)
        
        waveform = waveform.clone()
        
        for b in range(batch_size):
            # Get indices of unannotated frames for this sample
            unannotated_frames = torch.where(~has_annotation[b])[0]
            
            if len(unannotated_frames) == 0:
                continue  # All frames have annotations, skip masking
            
            for _ in range(self.num_masks):
                # Random mask length (in frames), small periods
                mask_len_frames = random.randint(1, min(self.max_mask_frames, len(unannotated_frames)))
                
                # Find a contiguous unannotated region to mask
                # Pick a random starting frame from unannotated frames
                start_idx = random.randint(0, len(unannotated_frames) - 1)
                frame_start = unannotated_frames[start_idx].item()
                
                # Extend mask only through contiguous unannotated frames
                frame_end = frame_start
                for i in range(mask_len_frames):
                    next_frame = frame_start + i
                    if next_frame >= num_output_frames:
                        break
                    if has_annotation[b, next_frame]:
                        break  # Stop at annotated frame
                    frame_end = next_frame + 1
                
                if frame_end <= frame_start:
                    continue  # No valid region to mask
                
                # Convert frame indices to sample indices
                sample_start = int(frame_start * samples_per_frame)
                sample_end = int(frame_end * samples_per_frame)
                sample_end = min(sample_end, num_samples)
                
                # Zero out the waveform segment
                if waveform.dim() == 2:
                    waveform[b, sample_start:sample_end] = 0
                else:
                    waveform[b, :, sample_start:sample_end] = 0
        
        return waveform


def get_augmentations(config) -> AudioAugmentations:
    """
    Factory function to create augmentations based on config.
    
    Args:
        config: Config object (can add augmentation settings to it)
    
    Returns:
        AudioAugmentations module
    """
    # Default augmentation settings - can be moved to config
    return AudioAugmentations(
        noise_prob=0.5,
        noise_snr_range=(10, 30),
        gain_prob=0.5,
        gain_range=(0.7, 1.3),
        time_shift_prob=0.0,  # Disabled since dataset handles window positioning
    )


def get_time_masking(config) -> WaveformTimeMasking:
    """
    Factory function to create time masking augmentation based on config.
    
    Args:
        config: Config object with time masking settings
    
    Returns:
        WaveformTimeMasking module or None if disabled
    """
    if not getattr(config, 'use_time_masking', False):
        return None
    
    return WaveformTimeMasking(
        prob=getattr(config, 'time_mask_prob', 0.5),
        max_mask_frames=getattr(config, 'time_mask_max_frames', 5),
        num_masks=getattr(config, 'time_mask_num_masks', 3),
    )

