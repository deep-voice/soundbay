"""
Local dataset for training with pre-extracted samples.

Usage:
    # Download data once before training
    python local_dataset.py download
    
    # In training, use get_local_dataloaders instead of get_dataloaders
    from local_dataset import get_local_dataloaders
    train_loader, val_loader = get_local_dataloaders(config)
"""

import subprocess
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from config import Config
from augmentations import AudioAugmentations


S3_URI = "s3://deepvoice-datasets/nature_dclde_2026_samples"
LOCAL_DIR = Path("/opt/dlami/nvme/dclde_samples")


def download_dataset(force: bool = False):
    """Download samples from S3 to local NVMe."""
    if LOCAL_DIR.exists() and not force:
        print(f"Dataset exists at {LOCAL_DIR}. Use force=True to re-download.")
        return LOCAL_DIR
    
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {S3_URI} to {LOCAL_DIR}...")
    subprocess.run(["aws", "s3", "sync", S3_URI, str(LOCAL_DIR)], check=True)
    print("Done!")
    return LOCAL_DIR


class DCLDELocalDataset(Dataset):
    """Dataset reading pre-extracted samples from local storage."""
    
    def __init__(self, df: pd.DataFrame, config: Config, all_annotations: pd.DataFrame = None, is_train: bool = False):
        """
        Args:
            df: DataFrame with samples (must have sample_id, window_start, window_end, gcs_url)
            all_annotations: Full annotations DataFrame for looking up overlapping labels
            is_train: Whether this is a training dataset (augmentations only applied during training)
        """
        self.df = df.copy()
        self.config = config
        self.num_frames = config.num_output_frames
        self.frames_per_sec = self.num_frames / config.window_sec
        self.label_map = config.label_map
        self.window_sec = config.window_sec
        self.target_sr = config.target_sr
        self.is_train = is_train
        
        # Build sample paths
        self.df['sample_path'] = self.df['sample_id'].apply(
            lambda sid: LOCAL_DIR / "samples" / f"{sid}.pt"
        )
        
        # Filter to existing files (skip failed exports)
        exists = self.df['sample_path'].apply(lambda p: p.exists())
        if not exists.all():
            print(f"Warning: {(~exists).sum()} samples not found locally")
            self.df = self.df[exists].reset_index(drop=True)
        
        # Use full annotations for looking up overlapping labels (not just train/val subset)
        # This fixes the bug where annotations from other splits were missed
        ann_df = all_annotations if all_annotations is not None else self.df
        self.file_groups = ann_df.groupby('gcs_url')
        
        # Audio augmentations (only for training datasets)
        self.augmentations = AudioAugmentations(
            aug_prob=config.aug_prob,
            noise_prob=config.aug_noise_prob,
            noise_snr_range=(config.aug_noise_snr_min, config.aug_noise_snr_max),
            gain_prob=config.aug_gain_prob,
            gain_range=(config.aug_gain_min, config.aug_gain_max),
        ) if (is_train and config.use_augmentations) else None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio
        data = torch.load(row['sample_path'], weights_only=True)
        audio = data['audio'].float()
        
        # Apply augmentations (only during training)
        if self.augmentations is not None:
            # AudioAugmentations expects to be in training mode
            self.augmentations.train()
            audio = self.augmentations(audio)
        
        w_start, w_end = row['window_start'], row['window_end']
        
        # Find overlapping annotations
        try:
            file_anns = self.file_groups.get_group(row['gcs_url'])
            overlapping = file_anns[
                (file_anns['FileBeginSec'] < w_end) & (file_anns['FileEndSec'] > w_start)
            ]
        except KeyError:
            overlapping = pd.DataFrame()
        
        # Build frame-level labels
        target = torch.zeros(self.num_frames, len(self.label_map))
        annotations = []
        
        for _, ann in overlapping.iterrows():
            if ann['ClassSpecies'] in self.label_map:
                rel_start = max(0, ann['FileBeginSec'] - w_start)
                rel_end = min(self.window_sec, ann['FileEndSec'] - w_start)
                start_frame = int(rel_start * self.frames_per_sec)
                end_frame = min(int(rel_end * self.frames_per_sec), self.num_frames)
                target[start_frame:end_frame, self.label_map[ann['ClassSpecies']]] = 1.0
                annotations.append({
                    'class': ann['ClassSpecies'],
                    'rel_start': rel_start,
                    'rel_end': rel_end,
                    'file_start': ann['FileBeginSec'],
                    'file_end': ann['FileEndSec'],
                })
        
        return {
            'audio': audio,
            'target_frames': target,
            'window_start': w_start,
            'window_end': w_end,
            'gcs_url': row['gcs_url'],
            'annotations': annotations,
        }


def collate_fn(batch):
    return {
        'audio': torch.stack([b['audio'] for b in batch]),
        'target_frames': torch.stack([b['target_frames'] for b in batch]),
        'window_start': [b['window_start'] for b in batch],
        'window_end': [b['window_end'] for b in batch],
        'gcs_url': [b['gcs_url'] for b in batch],
        'annotations': [b['annotations'] for b in batch],
    }


def get_local_dataloaders(config: Config):
    """Create train/val dataloaders from local pre-extracted samples."""
    # Load samples.csv which has the mapping from annotations to windows
    df = pd.read_csv(LOCAL_DIR / "samples.csv")
    initial_count = len(df)
    
    # Filter out failed exports (if error column exists)
    if 'error' in df.columns:
        n_errors = df['error'].notna().sum()
        df = df[df['error'].isna()].reset_index(drop=True)
    else:
        n_errors = 0
    
    # === Data Cleaning ===
    # 1. Remove duplicate annotations
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['gcs_url', 'FileBeginSec', 'FileEndSec', 'ClassSpecies'])
    n_duplicates = before_dedup - len(df)
    
    # 2. Filter very short annotations (<0.1s) - likely click artifacts or annotation errors
    df['_ann_duration'] = df['FileEndSec'] - df['FileBeginSec']
    before_short = len(df)
    df = df[df['_ann_duration'] >= 0.1]
    n_short = before_short - len(df)
    df = df.drop(columns=['_ann_duration'])
    
    df = df.reset_index(drop=True)
    
    print(f"Data cleaning: {initial_count} -> {len(df)} samples")
    print(f"  - Filtered {n_errors} export errors")
    print(f"  - Removed {n_duplicates} duplicates")
    print(f"  - Removed {n_short} short annotations (<0.1s)")
    
    # Show available sites for debugging
    available_sites = set(df['Dataset'].unique())
    print(f"Available sites: {sorted(available_sites)}")
    print(f"Available sites: {sorted(available_sites)}")
    
    train_df = df[df['Dataset'].isin(config.train_sites)].reset_index(drop=True)
    val_df = df[df['Dataset'].isin(config.val_sites)].reset_index(drop=True)
    
    # Check for site mismatches
    missing_train_sites = set(config.train_sites) - available_sites
    missing_val_sites = set(config.val_sites) - available_sites
    if missing_train_sites:
        print(f"Warning: Train sites not found in data: {missing_train_sites}")
    if missing_val_sites:
        print(f"Warning: Val sites not found in data: {missing_val_sites}")
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    if len(train_df) == 0:
        raise ValueError(
            f"No training samples found! Train sites: {config.train_sites}. "
            f"Available sites: {sorted(available_sites)}"
        )
    if len(val_df) == 0:
        raise ValueError(
            f"No validation samples found! Val sites: {config.val_sites}. "
            f"Available sites: {sorted(available_sites)}"
        )
    
    # Class balancing
    weights = 1.0 / train_df['ClassSpecies'].value_counts()
    sampler = WeightedRandomSampler(
        train_df['ClassSpecies'].map(weights).values, len(train_df)
    )
    
    # Use persistent workers only if we have enough samples
    use_persistent = len(train_df) >= config.batch_size * config.num_workers
    
    # Pass full annotations to dataset for proper label lookup
    # (fixes bug where annotations from other splits were not found)
    train_loader = DataLoader(
        DCLDELocalDataset(train_df, config, all_annotations=df, is_train=True),
        batch_size=config.batch_size, sampler=sampler,
        num_workers=config.num_workers, collate_fn=collate_fn,
        prefetch_factor=config.prefetch_factor, persistent_workers=use_persistent, pin_memory=True,
    )
    val_loader = DataLoader(
        DCLDELocalDataset(val_df, config, all_annotations=df, is_train=False),
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, collate_fn=collate_fn,
        prefetch_factor=config.prefetch_factor, persistent_workers=use_persistent, pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download_dataset(force="--force" in sys.argv)
    else:
        print("Usage: python local_dataset.py download [--force]")
