import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from google.cloud import storage
from gcsfs import GCSFileSystem
from sklearn.model_selection import GroupKFold
from skimage.transform import resize

from dclde_2026 import config, utils

try:
    from audiosample import AudioSample
    AUDIOSAMPLE_AVAILABLE = True
except ImportError:
    AUDIOSAMPLE_AVAILABLE = False
    print("Warning: AudioSample not available. Install audiosample for streaming audio loading.")


class GCSAudioLoader:
    def __init__(self, bucket_name, cache_dir=None, enable_cache=False):
        self.bucket_name = bucket_name
        self.target_sr = config.SAMPLE_RATE
        if enable_cache and cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(bucket_name)
        else:
            self.storage_client = None
            self.bucket = None

    def _get_gcs_url(self, gcs_path):
        if gcs_path.startswith('http'):
            return gcs_path
        if gcs_path.startswith('gs://'):
            gcs_path = gcs_path.replace('gs://', '')
        return f"https://storage.googleapis.com/{self.bucket_name}/{gcs_path}"

    def _get_sample_rate(self, audio_sample, orig_sr_hint):
        """Get sample rate from AudioSample or use hint"""
        for attr in ['sample_rate', 'sr', 'get_sample_rate']:
            if hasattr(audio_sample, attr):
                try:
                    sr = getattr(audio_sample, attr)
                    return sr() if callable(sr) else sr
                except:
                    pass
        return orig_sr_hint or config.SAMPLE_RATE
    
    def _normalize_audio_shape(self, audio):
        """Normalize audio tensor to 1D mono"""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() > 2:
            audio = audio.squeeze()
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        return audio.squeeze(0)
    
    def _try_audiosample_resample(self, segment, target_sr):
        """Try AudioSample's resampling methods"""
        if hasattr(segment, 'resample'):
            try:
                return segment.resample(target_sr).as_tensor()
            except:
                pass
        if hasattr(segment, 'as_tensor'):
            try:
                return segment.as_tensor(sample_rate=target_sr)
            except TypeError:
                pass
        return None
    
    def load_audio_segment(self, gcs_path, offset_sec, duration_sec, orig_sr_hint=None):
        """Load audio segment using AudioSample with built-in resampling"""
        if not AUDIOSAMPLE_AVAILABLE:
            raise ImportError("AudioSample not available. Please install audiosample package.")
        
        try:
            gcs_url = self._get_gcs_url(gcs_path)
            
            # Try constructor with target SR first
            try:
                audio = AudioSample(gcs_url, sample_rate=self.target_sr)[offset_sec:offset_sec + duration_sec].as_tensor()
            except (TypeError, ValueError):
                # Fallback: load then resample
                audio_sample = AudioSample(gcs_url)
                orig_sr = self._get_sample_rate(audio_sample, orig_sr_hint)
                segment = audio_sample[offset_sec:offset_sec + duration_sec]
                
                # Try AudioSample resampling
                audio = self._try_audiosample_resample(segment, self.target_sr) if orig_sr != self.target_sr else None
                
                # Load normally if resampling failed or not needed
                if audio is None:
                    audio = segment.as_tensor()
                    if orig_sr != self.target_sr:
                        audio = self._resample_with_torchaudio(audio, orig_sr, self.target_sr)
            
            audio = self._normalize_audio_shape(audio)
            return (audio.numpy() if isinstance(audio, torch.Tensor) else audio, self.target_sr)
        except Exception as e:
            print(f"ERROR: Failed to load audio segment from {gcs_path}. {e}")
            return None, 0
    
    def _resample_with_torchaudio(self, audio, orig_sr, target_sr):
        """Fallback resampling using torchaudio if AudioSample resampling fails"""
        if orig_sr == target_sr:
            return audio
        
        # Warn about upsampling - it doesn't add new frequency information
        if orig_sr < target_sr:
            max_freq_available = orig_sr / 2.0  # Nyquist frequency
            if max_freq_available < config.MAX_FREQ_HZ:
                print(f"WARNING: Upsampling from {orig_sr}Hz to {target_sr}Hz. "
                      f"Max frequency available: {max_freq_available:.0f}Hz (Nyquist), "
                      f"but target is {config.MAX_FREQ_HZ:.0f}Hz. "
                      f"Upsampling won't create frequencies above {max_freq_available:.0f}Hz.")
        
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio) if isinstance(audio, np.ndarray) else torch.tensor(audio)
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)
        
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        resampled = resampler(audio)
        
        if resampled.dim() == 3:
            resampled = resampled.squeeze(0)
        if resampled.dim() == 2 and resampled.shape[0] == 1:
            resampled = resampled.squeeze(0)
        
        return resampled


class AudioAugmentations:
    """Audio-level augmentations (applied before spectrogram)"""
    def __init__(self, annotations_df=None):
        self.annotations_df = annotations_df
        self.noise_segments = None
        if config.AUGMENT_AUDIO_NOISE and annotations_df is not None:
            # Extract noise segments (annotations marked as noise/background)
            noise_mask = annotations_df.get('ClassSpeciesKW', pd.Series()).str.contains('noise|background|Noise|Background', case=False, na=False)
            if noise_mask.any():
                self.noise_segments = annotations_df[noise_mask]
    
    def add_noise(self, audio, gcs_path):
        """Add noise from annotations if available"""
        if not config.AUGMENT_AUDIO_NOISE or self.noise_segments is None or np.random.rand() > config.AUGMENT_AUDIO_NOISE_P:
            return audio
        
        noise_candidates = self.noise_segments[self.noise_segments['gcs_path'] == gcs_path]
        if len(noise_candidates) == 0:
            noise_candidates = self.noise_segments
        
        if len(noise_candidates) > 0:
            # Load noise segment (simplified - would need audio_loader in practice)
            # For now, add Gaussian noise as placeholder
            noise_level = np.random.uniform(0.01, 0.05)
            audio = audio + np.random.normal(0, noise_level, audio.shape)
        
        return audio
    
    def apply_gain(self, audio):
        """Apply random gain/volume change"""
        if not config.AUGMENT_AUDIO_GAIN or np.random.rand() > config.AUGMENT_AUDIO_GAIN_P:
            return audio
        gain_db = np.random.uniform(config.AUGMENT_AUDIO_GAIN_MIN_DB, config.AUGMENT_AUDIO_GAIN_MAX_DB)
        gain_linear = 10 ** (gain_db / 20.0)
        return audio * gain_linear


class SpectrogramAugmentations:
    """Spectrogram-level augmentations (SpecAugment, Mixup, CutMix)"""
    
    @staticmethod
    def specaugment(spec, labels):
        """Apply SpecAugment: mask frequency and time bands"""
        if not config.AUGMENT_SPEC_SPECAUGMENT or np.random.rand() > config.AUGMENT_SPEC_SPECAUGMENT_P:
            return spec, labels
        
        # Frequency masking
        if config.AUGMENT_SPEC_FREQ_MASK:
            freq_mask_size = np.random.randint(0, config.AUGMENT_SPEC_FREQ_MASK_MAX + 1)
            if freq_mask_size > 0:
                freq_start = np.random.randint(0, max(1, spec.shape[1] - freq_mask_size))
                spec[:, freq_start:freq_start + freq_mask_size, :] = 0
        
        # Time masking
        if config.AUGMENT_SPEC_TIME_MASK:
            time_mask_size = np.random.randint(0, config.AUGMENT_SPEC_TIME_MASK_MAX + 1)
            if time_mask_size > 0:
                time_start = np.random.randint(0, max(1, spec.shape[2] - time_mask_size))
                spec[:, :, time_start:time_start + time_mask_size] = 0
        
        return spec, labels
    
    @staticmethod
    def mixup(spec1, labels1, spec2, labels2):
        """Mixup: blend two spectrograms and their labels"""
        if not config.AUGMENT_MIXUP or np.random.rand() > config.AUGMENT_MIXUP_P:
            return spec1, labels1
        
        alpha = config.AUGMENT_MIXUP_ALPHA
        lam = np.random.beta(alpha, alpha)
        
        mixed_spec = lam * spec1 + (1 - lam) * spec2
        # Combine labels (for detection, we concatenate boxes)
        if len(labels1) > 0 and len(labels2) > 0:
            mixed_labels = torch.cat([labels1, labels2], dim=0)
        else:
            mixed_labels = labels1 if len(labels1) > 0 else labels2
        
        return mixed_spec, mixed_labels
    
    @staticmethod
    def cutmix(spec1, labels1, spec2, labels2):
        """CutMix: replace a region of spec1 with spec2"""
        if not config.AUGMENT_CUTMIX or np.random.rand() > config.AUGMENT_CUTMIX_P:
            return spec1, labels1
        
        alpha = config.AUGMENT_CUTMIX_ALPHA
        lam = np.random.beta(alpha, alpha)
        
        # Get bounding box for cut region
        h, w = spec1.shape[1], spec1.shape[2]
        cut_h = int(h * np.sqrt(1 - lam))
        cut_w = int(w * np.sqrt(1 - lam))
        y = np.random.randint(0, max(1, h - cut_h))
        x = np.random.randint(0, max(1, w - cut_w))
        
        # Apply cutmix
        mixed_spec = spec1.clone()
        mixed_spec[:, y:y+cut_h, x:x+cut_w] = spec2[:, y:y+cut_h, x:x+cut_w]
        
        # Adjust labels (boxes in cut region from spec2, others from spec1)
        # Simplified: concatenate all labels
        if len(labels1) > 0 and len(labels2) > 0:
            mixed_labels = torch.cat([labels1, labels2], dim=0)
        else:
            mixed_labels = labels1 if len(labels1) > 0 else labels2
        
        return mixed_spec, mixed_labels


class AudioObjectDataset(Dataset):
    def __init__(self, df, annotations_df, audio_loader, is_train=True, use_beats=False, model_type=None):
        self.df_chips = df
        self.annotations_df = annotations_df
        self.audio_loader = audio_loader
        self.is_train = is_train
        self.use_beats = use_beats
        self.model_type = model_type or config.MODEL_TYPE.lower()
        self.audio_aug = AudioAugmentations(annotations_df) if is_train else None
        self.spec_aug = SpectrogramAugmentations() if is_train else None
        
        # Create spectrogram transforms (one per n_fft)
        self.spec_transforms = {}
        for n_fft in config.N_FFT_LIST:
            self.spec_transforms[n_fft] = torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                hop_length=config.HOP_LENGTH,
                win_length=n_fft,
                window_fn=torch.hann_window,
                power=2.0
            )

    def __len__(self):
        return len(self.df_chips)

    def generate_spectrograms(self, y):
        """Generate 3-channel linear spectrograms using torchaudio.transforms.Spectrogram"""
        specs = []
        y_torch = torch.from_numpy(y).float()
        
        for n_fft in config.N_FFT_LIST:
            # Use torchaudio.transforms.Spectrogram (much simpler!)
            spec = self.spec_transforms[n_fft](y_torch.unsqueeze(0))  # [1, freq_bins, time_frames]
            spec = spec.squeeze(0)  # [freq_bins, time_frames]
            
            # Convert to dB
            S_db = 10 * torch.log10(spec + 1e-10)
            S_db = S_db - S_db.max()  # Normalize to max
            
            S_db_np = S_db.numpy()
            
            # Resize to common height if needed
            if S_db_np.shape[0] != config.TARGET_FREQ_BINS:
                S_db_np = resize(S_db_np, (config.TARGET_FREQ_BINS, S_db_np.shape[1]), 
                                preserve_range=True, anti_aliasing=True)
            specs.append(S_db_np)
        
        specs_stacked = np.stack(specs, axis=0)
        specs_norm = (specs_stacked - specs_stacked.min()) / (specs_stacked.max() - specs_stacked.min() + 1e-6)
        return specs_norm

    def __getitem__(self, idx):
        chip_info = self.df_chips.iloc[idx]
        gcs_path, chip_start_sec = chip_info['gcs_path'], chip_info['chip_start_sec']
        chip_end_sec = chip_start_sec + config.WINDOW_SEC

        orig_sr_hint = None
        if 'SampleRate' in self.annotations_df.columns:
            file_anns = self.annotations_df[self.annotations_df['gcs_path'] == gcs_path]
            if len(file_anns) > 0:
                orig_sr_hint = file_anns.iloc[0].get('SampleRate', None)
        
        y, sr = self.audio_loader.load_audio_segment(gcs_path, chip_start_sec, config.WINDOW_SEC, orig_sr_hint)
        if y is None:
            if config.DEBUG:
                print(f"DEBUG: Failed to load audio for idx {idx}, gcs_path: {gcs_path}")
            return None, None
        
        # Debug mode: print audio info
        if config.DEBUG and idx < 3:
            print(f"DEBUG: Sample {idx}")
            print(f"  gcs_path: {gcs_path}")
            print(f"  chip_start_sec: {chip_start_sec}")
            print(f"  Audio shape: {y.shape}, sample_rate: {sr}")

        # Audio-level augmentations
        if self.is_train and self.audio_aug:
            y = self.audio_aug.add_noise(y, gcs_path)
            y = self.audio_aug.apply_gain(y)

        if len(y) < config.CHIP_SAMPLES:
            y = np.pad(y, (0, config.CHIP_SAMPLES - len(y)))
        else:
            y = y[:config.CHIP_SAMPLES]
        
        chip_labels = self.annotations_df[
            (self.annotations_df['gcs_path'] == gcs_path) &
            (self.annotations_df['FileBeginSec'] < chip_end_sec) &
            (self.annotations_df['FileEndSec'] > chip_start_sec)
        ]
        
        yolo_labels = utils.convert_labels_to_yolo(chip_labels, chip_start_sec, config.WINDOW_SEC)
        labels_tensor = torch.tensor(yolo_labels, dtype=torch.float32)
        
        # Debug mode: print label info
        if config.DEBUG and idx < 3:
            print(f"  Num labels: {len(yolo_labels)}")
            if len(yolo_labels) > 0:
                print(f"  Label shape: {labels_tensor.shape}")
                print(f"  Sample labels: {yolo_labels[:min(2, len(yolo_labels))]}")
        
        if self.use_beats:
            output = torch.from_numpy(y).float(), labels_tensor
        else:
            specs_norm = self.generate_spectrograms(y)
            image_tensor = torch.tensor(specs_norm, dtype=torch.float32)
            
            # Debug mode: print spectrogram info
            if config.DEBUG and idx < 3:
                print(f"  Spectrogram shape: {image_tensor.shape}")
            
            # Spectrogram-level augmentations
            if self.is_train and self.spec_aug:
                image_tensor, labels_tensor = self.spec_aug.specaugment(image_tensor, labels_tensor)
                
                # Mixup/CutMix: randomly select another sample
                if config.AUGMENT_MIXUP or config.AUGMENT_CUTMIX:
                    other_idx = np.random.randint(0, len(self.df_chips))
                    other_spec, other_labels = self._get_spec_and_labels(other_idx)
                    if other_spec is not None:
                        if config.AUGMENT_MIXUP and np.random.rand() < config.AUGMENT_MIXUP_P:
                            image_tensor, labels_tensor = self.spec_aug.mixup(image_tensor, labels_tensor, other_spec, other_labels)
                        elif config.AUGMENT_CUTMIX and np.random.rand() < config.AUGMENT_CUTMIX_P:
                            image_tensor, labels_tensor = self.spec_aug.cutmix(image_tensor, labels_tensor, other_spec, other_labels)
            
            # YOLO uses natural spectrogram dimensions (no resizing)
            # Labels are already normalized [0, 1] so they work with any image size
            output = image_tensor, labels_tensor
        
        # Debug mode: print final output info
        if config.DEBUG and idx < 3:
            if self.use_beats:
                print(f"  Output audio shape: {output[0].shape}, labels shape: {output[1].shape}")
            else:
                print(f"  Output image shape: {output[0].shape}, labels shape: {output[1].shape}")
            print()
        
        return output
    
    def _get_spec_and_labels(self, idx):
        """Helper to get spectrogram and labels for Mixup/CutMix"""
        chip_info = self.df_chips.iloc[idx]
        gcs_path, chip_start_sec = chip_info['gcs_path'], chip_info['chip_start_sec']
        chip_end_sec = chip_start_sec + config.WINDOW_SEC
        
        y, sr = self.audio_loader.load_audio_segment(gcs_path, chip_start_sec, config.WINDOW_SEC)
        if y is None:
            return None, None
        
        if len(y) < config.CHIP_SAMPLES:
            y = np.pad(y, (0, config.CHIP_SAMPLES - len(y)))
        else:
            y = y[:config.CHIP_SAMPLES]
        
        chip_labels = self.annotations_df[
            (self.annotations_df['gcs_path'] == gcs_path) &
            (self.annotations_df['FileBeginSec'] < chip_end_sec) &
            (self.annotations_df['FileEndSec'] > chip_start_sec)
        ]
        
        yolo_labels = utils.convert_labels_to_yolo(chip_labels, chip_start_sec, config.WINDOW_SEC)
        labels_tensor = torch.tensor(yolo_labels, dtype=torch.float32)
        specs_norm = self.generate_spectrograms(y)
        return torch.tensor(specs_norm, dtype=torch.float32), labels_tensor


def create_full_annotation_df():
    print("Loading annotations...")
    fs = GCSFileSystem()
    with fs.open(config.GCS_ANNOTATION_PATH) as f:
        df_ann = pd.read_csv(f)

    print("Filtering annotations...")
    df_ann = df_ann[~df_ann['Provider'].isin(config.PROVIDERS_TO_EXCLUDE)]
    df_ann = df_ann[
        (df_ann['LowFreqHz'] < config.MAX_FREQ_HZ) &
        (df_ann['HighFreqHz'] > config.MIN_FREQ_HZ) &
        ((df_ann['HighFreqHz'] - df_ann['LowFreqHz']) > 1)
    ]
    
    # Filter by minimum required sample rate if specified
    if config.MIN_REQUIRED_SAMPLE_RATE is not None and 'SampleRate' in df_ann.columns:
        before = len(df_ann)
        df_ann = df_ann[df_ann['SampleRate'] >= config.MIN_REQUIRED_SAMPLE_RATE]
        print(f"Filtered {before - len(df_ann)} files with sample rate < {config.MIN_REQUIRED_SAMPLE_RATE}Hz")
    
    df_ann['gcs_path'] = df_ann.apply(utils.map_filepath_to_gcs, axis=1)
    df_ann = df_ann.dropna(subset=['gcs_path'])
    return df_ann


def create_chip_list(df_ann):
    print("Creating chips...")
    file_durations = df_ann.groupby('gcs_path')['FileEndSec'].max()
    
    chip_list = []
    for gcs_path, max_time in file_durations.items():
        group = df_ann[df_ann['gcs_path'] == gcs_path].iloc[0][config.SPLIT_GROUP_COLUMN]
        for start_sec in np.arange(0, max_time - config.WINDOW_SEC, config.WINDOW_SEC / 2):
            chip_list.append({'gcs_path': gcs_path, 'chip_start_sec': start_sec, 'group': group})
    return pd.DataFrame(chip_list)


def get_dataloaders(df_ann, df_chips, fold_id=0, use_beats=False, model_type=None):
    print("Creating train/val splits...")
    gkf = GroupKFold(n_splits=config.N_SPLITS)
    splits = list(gkf.split(df_chips, groups=df_chips['group']))
    
    val_idx = splits[fold_id][1]
    test_idx = splits[(fold_id + 1) % config.N_SPLITS][1]
    train_idx = list(set(range(len(df_chips))) - set(val_idx) - set(test_idx))
    
    df_train = df_chips.iloc[train_idx].reset_index(drop=True)
    df_val = df_chips.iloc[val_idx].reset_index(drop=True)
    df_test = df_chips.iloc[test_idx].reset_index(drop=True)
    
    # Debug mode: limit dataset size
    if config.DEBUG:
        print(f"DEBUG MODE: Limiting to {config.DEBUG_MAX_SAMPLES} samples per split")
        df_train = df_train.head(config.DEBUG_MAX_SAMPLES).reset_index(drop=True)
        df_val = df_val.head(config.DEBUG_MAX_SAMPLES).reset_index(drop=True)
        df_test = df_test.head(config.DEBUG_MAX_SAMPLES).reset_index(drop=True)

    print(f"Train chips: {len(df_train)}, Val chips: {len(df_val)}, Test chips: {len(df_test)}")
    print(f"Train files: {df_train['gcs_path'].nunique()}, Val files: {df_val['gcs_path'].nunique()}, Test files: {df_test['gcs_path'].nunique()}")
    
    # Verify splits have different files and locations
    train_files = set(df_train['gcs_path'].unique())
    val_files = set(df_val['gcs_path'].unique())
    test_files = set(df_test['gcs_path'].unique())
    
    train_groups = set(df_train['group'].unique())
    val_groups = set(df_val['group'].unique())
    test_groups = set(df_test['group'].unique())
    
    train_val_overlap = train_files & val_files
    train_test_overlap = train_files & test_files
    val_test_overlap = val_files & test_files
    
    train_val_group_overlap = train_groups & val_groups
    train_test_group_overlap = train_groups & test_groups
    val_test_group_overlap = val_groups & test_groups
    
    print(f"\nSplit verification:")
    print(f"  Train-Val file overlap: {len(train_val_overlap)} files")
    print(f"  Train-Test file overlap: {len(train_test_overlap)} files")
    print(f"  Val-Test file overlap: {len(val_test_overlap)} files")
    print(f"  Train-Val location overlap: {len(train_val_group_overlap)} locations")
    print(f"  Train-Test location overlap: {len(train_test_group_overlap)} locations")
    print(f"  Val-Test location overlap: {len(val_test_group_overlap)} locations")
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print(f"  WARNING: File overlap detected! This may cause data leakage.")
    if train_val_group_overlap or train_test_group_overlap or val_test_group_overlap:
        print(f"  WARNING: Location overlap detected! Test set may not be from different locations.")
    else:
        print(f"  ✓ All splits have different locations (groups)")

    audio_loader = GCSAudioLoader(
        bucket_name=config.GCS_AUDIO_BUCKET_NAME,
        cache_dir=config.LOCAL_AUDIO_CACHE_DIR if config.ENABLE_AUDIO_CACHE else None,
        enable_cache=config.ENABLE_AUDIO_CACHE
    )

    train_ds = AudioObjectDataset(df_train, df_ann, audio_loader, is_train=True, use_beats=use_beats, model_type=model_type)
    val_ds = AudioObjectDataset(df_val, df_ann, audio_loader, is_train=False, use_beats=use_beats, model_type=model_type)
    test_ds = AudioObjectDataset(df_test, df_ann, audio_loader, is_train=False, use_beats=use_beats, model_type=model_type)

    def collate_fn(batch):
        batch = [b for b in batch if b[0] is not None]
        if not batch:
            return torch.tensor([]), []
        
        if use_beats:
            audios = [item[0] for item in batch]
            max_len = max([a.shape[0] for a in audios])
            return torch.stack([F.pad(a, (0, max_len - a.shape[0]), "constant", 0) for a in audios]), [item[1] for item in batch]
        else:
            images = [item[0] for item in batch]
            max_time = max([img.shape[2] for img in images])
            return torch.stack([F.pad(img, (0, max_time - img.shape[2]), "constant", 0) for img in images]), [item[1] for item in batch]

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                             num_workers=config.NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                           num_workers=config.NUM_WORKERS, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
