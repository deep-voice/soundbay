import torch
import random
import time
import gcsfs
import re
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchaudio.io import StreamReader

from augmentations import AudioAugmentations


FOLDER_MAPPINGS = {
    'quin_can': 'qc',
    'straitofgeorgia': 'straitofgeorgia_globus-robertsbank',
}

def map_to_gcs(local_path):
    path = local_path.replace("\\", "/")
    match = re.search(r'DCLDE/(.*)', path, re.IGNORECASE)
    if not match:
        return None
    relative_path = match.group(1)
    parts = [p for p in relative_path.split("/") if p]  # Filter out empty strings from double slashes
    # Normalize: lowercase, spaces to hyphens, apply folder mappings
    normalized = []
    for p in parts[:-1]:
        p = p.lower().replace(" ", "-")
        p = FOLDER_MAPPINGS.get(p, p)
        normalized.append(p)
    directory = "/".join(normalized)
    filename = parts[-1]
    base_bucket = "gs://noaa-passive-bioacoustic/dclde/2027/dclde_2026_killer_whales"
    return f"{base_bucket}/{directory}/{filename}"


class DCLDEMultiLabelDataset(Dataset):
    def __init__(self, df, config, is_train: bool = False):
        self.df = df
        self.window_sec = config.window_sec
        self.target_sr = config.target_sr
        self.label_map = config.label_map
        self.num_frames = config.num_output_frames  # Matches Perch encoder output
        self.frames_per_sec = self.num_frames / self.window_sec  # Derived: 12.8 fps
        self._fs = None  # Lazy init for fork-safety with multiprocessing
        self.file_groups = df.groupby('gcs_url')
        self.is_train = is_train
        
        # Audio augmentations (only for training datasets)
        self.augmentations = AudioAugmentations(
            aug_prob=config.aug_prob,
            noise_prob=config.aug_noise_prob,
            noise_snr_range=(config.aug_noise_snr_min, config.aug_noise_snr_max),
            gain_prob=config.aug_gain_prob,
            gain_range=(config.aug_gain_min, config.aug_gain_max),
        ) if (is_train and config.use_augmentations) else None
    
    @property
    def fs(self):
        """Lazy-initialize gcsfs to avoid fork-safety issues with DataLoader workers."""
        if self._fs is None:
            self._fs = gcsfs.GCSFileSystem(token='anon')
        return self._fs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gcs_url = row['gcs_url']
        ann_start, ann_end = row['FileBeginSec'], row['FileEndSec']
        
        try:
            # Load audio and get actual window bounds (clamped to file duration)
            waveform, window_start, window_end = self._load_with_retries(
                gcs_url, ann_start, ann_end
            )
            
            # Apply augmentations (only during training)
            if self.augmentations is not None:
                # AudioAugmentations expects to be in training mode
                self.augmentations.train()
                waveform = self.augmentations(waveform)

            # Find all annotations overlapping this window
            other_anns = self.file_groups.get_group(gcs_url)
            overlapping = other_anns[
                (other_anns['FileBeginSec'] < window_end) &
                (other_anns['FileEndSec'] > window_start)
            ]
        except Exception as exc:
            return self._fallback_sample(gcs_url, ann_start, exc)

        # Frame-level labels: (num_frames, num_classes)
        target_frames = torch.zeros(self.num_frames, len(self.label_map))
        
        # Store raw annotation info for debugging
        annotations_in_window = []
        
        for _, ann in overlapping.iterrows():
            if ann['ClassSpecies'] in self.label_map:
                # Annotation times relative to window start
                rel_start = max(0, ann['FileBeginSec'] - window_start)
                rel_end = min(self.window_sec, ann['FileEndSec'] - window_start)
                
                # Convert to frame indices
                start_frame = int(rel_start * self.frames_per_sec)
                end_frame = min(int(rel_end * self.frames_per_sec), self.num_frames)
                
                class_idx = self.label_map[ann['ClassSpecies']]
                target_frames[start_frame:end_frame, class_idx] = 1.0
                
                # Store annotation info
                annotations_in_window.append({
                    'class': ann['ClassSpecies'],
                    'rel_start': rel_start,
                    'rel_end': rel_end,
                    'low_freq': ann.get('LowFreqHz', 0),
                    'high_freq': ann.get('HighFreqHz', self.target_sr // 2),
                    'file_start': ann['FileBeginSec'],
                    'file_end': ann['FileEndSec'],
                })
        
        return {
            'audio': waveform,
            'target_frames': target_frames,  # Shape: (num_frames, num_classes)
            'window_start': window_start,
            'window_end': window_end,
            'gcs_url': gcs_url,
            'annotations': annotations_in_window,  # Raw annotation boxes
        }

    def _fallback_sample(self, gcs_url, ann_start, exc):
        """Return a safe fallback sample when data loading fails."""
        expected_samples = int(self.window_sec * self.target_sr)
        waveform = torch.zeros(1, expected_samples)
        window_start = max(0.0, float(ann_start))
        window_end = window_start + self.window_sec
        target_frames = torch.zeros(self.num_frames, len(self.label_map))

        print(
            f"[DATA] Failed to load audio for {gcs_url} at {ann_start}: {exc}",
            flush=True,
        )

        return {
            'audio': waveform,
            'target_frames': target_frames,
            'window_start': window_start,
            'window_end': window_end,
            'gcs_url': gcs_url,
            'annotations': [],
        }

    def _load_with_retries(self, url, ann_start, ann_end, max_attempts=3):
        """Retry GCS streaming to avoid transient network errors."""
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self._load_from_stream(url, ann_start, ann_end)
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 5))
        raise last_exc

    def _load_from_stream(self, url, ann_start, ann_end):
        """Load audio window, clamped to file duration. Returns (waveform, window_start, window_end)."""
        with self.fs.open(url, 'rb') as f:
            streamer = StreamReader(f)
            
            # Get file duration from stream metadata
            info = streamer.get_src_stream_info(0)
            if info.num_frames > 0 and info.sample_rate > 0:
                file_duration = info.num_frames / info.sample_rate
            elif info.bit_rate > 0:
                # Compute duration from file size and bit rate (for PCM streams)
                file_size = f.details['size']
                file_duration = file_size * 8 / info.bit_rate
            else:
                # Fallback: use annotation end time + buffer as minimum duration estimate
                file_duration = ann_end + self.window_sec
            
            # Random window placement centered around annotation, clamped to file bounds
            min_window_start = max(0, ann_end - self.window_sec)
            max_window_start = min(ann_start, file_duration - self.window_sec)
            
            if min_window_start > max_window_start:
                window_start = max(0, file_duration - self.window_sec)
            else:
                window_start = random.uniform(min_window_start, max_window_start)
            
            window_end = window_start + self.window_sec
            
            streamer.add_basic_audio_stream(
                frames_per_chunk=int(self.window_sec * self.target_sr),
                sample_rate=self.target_sr
            )
            streamer.seek(window_start)
            
            expected_samples = int(self.window_sec * self.target_sr)
            for (chunk,) in streamer.stream():
                waveform = chunk.T
                # Convert to mono by averaging channels if stereo
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                # Ensure consistent length (pad or trim)
                if waveform.shape[1] < expected_samples:
                    pad_size = expected_samples - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, pad_size))
                elif waveform.shape[1] > expected_samples:
                    waveform = waveform[:, :expected_samples]
                return waveform, window_start, window_end
            
            # No chunks returned from stream (seek past EOF or empty file)
            # Return zero-padded waveform to allow training to continue
            waveform = torch.zeros(1, expected_samples)
            return waveform, window_start, window_end


def collate_fn(batch):
    """Custom collate function to handle variable-length annotations."""
    return {
        'audio': torch.stack([item['audio'] for item in batch]),
        'target_frames': torch.stack([item['target_frames'] for item in batch]),
        'window_start': [item['window_start'] for item in batch],
        'window_end': [item['window_end'] for item in batch],
        'gcs_url': [item['gcs_url'] for item in batch],
        'annotations': [item['annotations'] for item in batch],  # Keep as list of lists
    }


def get_dataloaders(config):
    df = pd.read_csv(config.csv_path)
    df['gcs_url'] = df['FilePath'].apply(map_to_gcs)
    df = df[df['gcs_url'].notna()].reset_index(drop=True)

    train_df = df[df['Dataset'].isin(config.train_sites)].reset_index(drop=True)
    val_df = df[df['Dataset'].isin(config.val_sites)].reset_index(drop=True)
    
    # Class balancing
    counts = train_df['ClassSpecies'].value_counts()
    weights = 1.0 / counts
    sample_weights = train_df['ClassSpecies'].map(weights).values
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        DCLDEMultiLabelDataset(train_df, config, is_train=True), 
        batch_size=config.batch_size, sampler=sampler, num_workers=config.num_workers,
        collate_fn=collate_fn,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        DCLDEMultiLabelDataset(val_df, config, is_train=False), 
        batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
        collate_fn=collate_fn,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def get_debug_dataloaders(config):
    """Get small dataloaders for debug/overfitting mode."""
    df = pd.read_csv(config.csv_path)
    df['gcs_url'] = df['FilePath'].apply(map_to_gcs)
    df = df[df['gcs_url'].notna()].reset_index(drop=True)
    
    # Use only a small subset of training data
    train_df = df[df['Dataset'].isin(config.train_sites)].reset_index(drop=True)
    train_df = train_df.head(config.debug_samples)
    
    # Use same data for validation in debug mode (to check overfitting)
    val_df = train_df.copy()
    
    train_loader = DataLoader(
        DCLDEMultiLabelDataset(train_df, config, is_train=True),
        batch_size=min(config.batch_size, config.debug_samples),
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        DCLDEMultiLabelDataset(val_df, config, is_train=False),
        batch_size=min(config.batch_size, config.debug_samples),
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True,
        pin_memory=True,
    )
    
    return train_loader, val_loader
