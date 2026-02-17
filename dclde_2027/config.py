from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Paths
    onnx_path: str = "/home/ubuntu/soundbay/dclde_2027/perch_v2.onnx"
    csv_path: str = "gs://noaa-passive-bioacoustic/dclde/2027/dclde_2026_killer_whales/Annotations.csv"
    
    # Audio
    target_sr: int = 32000
    window_sec: float = 5.0
    num_output_frames: int = 64  # Matches Perch encoder output (64 frames / 5s = 12.8 fps, ~78ms per frame)
    
    # Model
    model_type: str = "perch"  # "perch" or "beats"
    device: str = "cuda"
    num_classes: int = 4
    embed_dim: int = 1536  # For Perch; BEATS uses its own embed_dim from pretrained model
    num_heads: int = 8
    hidden_dim: int = 512
    dropout: float = 0.5
    
    # BEATS model configuration (only used when model_type="beats")
    beats_model_name: str = "facebook/beats-base"  # HuggingFace model name
    
    # Training
    batch_size: int = 32
    num_workers: int = 2  # Reduced to save memory
    prefetch_factor: int = 2  # Prefetch batches per worker
    epochs: int = 50
    lr: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    threshold: float = 0.5
    output_dir: str = "/home/ubuntu/soundbay/dclde_2027/checkpoints"
    resume_checkpoint_path: Optional[str] = None  # If set, load this checkpoint into model (supports old classifier format)

    # Loss function
    loss_type: str = "distance_weighted"  # "positive_only" or "distance_weighted"
    loss_decay_rate: float = 0.02  # How fast weight decays with distance (for distance_weighted)
    loss_min_weight: float = 0.1  # Minimum weight for distant frames (for distance_weighted)
    
    # Augmentations
    use_augmentations: bool = True  # Enable audio augmentations during training
    aug_prob: float = 0.3  # General probability of attempting any augmentation on a sample
    aug_noise_prob: float = 0.5  # Probability of adding Gaussian noise (given aug_prob)
    aug_noise_snr_min: float = 10.0  # Min SNR in dB (lower = more noise)
    aug_noise_snr_max: float = 30.0  # Max SNR in dB
    aug_gain_prob: float = 0.5  # Probability of applying gain variation
    aug_gain_min: float = 0.7  # Min gain multiplier
    aug_gain_max: float = 1.3  # Max gain multiplier
    aug_mixup_prob: float = 0.3  # Probability of applying mixup (blends pairs of samples)
    aug_mixup_alpha: float = 0.4  # Beta distribution parameter (lower = less aggressive mixing)
    
    # Time masking augmentation (zeros out time segments ONLY in unannotated regions)
    use_time_masking: bool = False  # Enable time masking augmentation
    time_mask_prob: float = 0.5  # Probability of applying time masking
    time_mask_max_frames: int = 5  # Max frames to mask per mask (small periods, ~390ms at 64 frames/5s)
    time_mask_num_masks: int = 3  # Number of masks to attempt per sample
    
    # Logging
    log_every_n_steps: int = 500  # Log metrics every N steps
    save_best_every_n_steps: int = 2000  # Run validation and save best checkpoint every N steps
    num_random_samples: int = 4
    num_worst_samples: int = 4
    wandb_project: str = "dclde_2026"
    
    # Debug mode (for overfitting sanity check)
    debug: bool = False
    debug_samples: int = 32  # Number of samples to use in debug mode
    debug_epochs: int = 100  # More epochs for overfitting check
    
    # Local data mode (faster training with pre-extracted samples on NVMe)
    use_local_data: bool = True  # If True, use local pre-extracted samples instead of streaming from GCS
    
    # Sites
    # Option 4: Diverse val with more HW samples while preserving UndBio in train
    train_sites: tuple = ('WVanIsl', 'NorthBc', 'Field_HTI', 'HaroStraitSouth', 'StraitofGeorgia', 'HaroStraitNorth')
    val_sites: tuple = ('BarkleyCanyon', 'BoundaryPass', 'port_townsend', 'Quin_Can', 'SwanChan', 'LimeKiln', 'CarmanahPt', 'Tekteksen')
    test_sites: tuple = ('Field_SondTrap',)  # Remaining site for final test
    
    # Labels
    label_map: dict = None
    
    def __post_init__(self):
        if self.label_map is None:
            self.label_map = {'KW': 0, 'HW': 1, 'AB': 2, 'UndBio': 3}
    
    @property
    def frames_per_sec(self):
        """Derived frame rate from Perch encoder output."""
        return self.num_output_frames / self.window_sec
