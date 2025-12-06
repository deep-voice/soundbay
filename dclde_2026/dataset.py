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


GCS_FILE_CACHE_PATH = "dclde_2026/gcs_files_cache.pkl"


def get_existing_gcs_files(cache_path=GCS_FILE_CACHE_PATH, force_refresh=False):
    """
    List all .wav files that actually exist in the GCS bucket under the DCLDE prefix.
    Results are cached to disk to avoid re-listing on every run.
    
    Returns:
        set: A set of full gs:// paths that exist in GCS
    """
    import pickle
    
    if os.path.exists(cache_path) and not force_refresh:
        print(f"Loading cached GCS file list from {cache_path}")
        with open(cache_path, 'rb') as f:
            existing_files = pickle.load(f)
        print(f"Loaded {len(existing_files)} existing files from cache")
        return existing_files
    
    print("Listing files in GCS bucket (this may take a few minutes for ~200k files)...")
    
    fs = GCSFileSystem()
    prefix = f"{config.GCS_AUDIO_BUCKET_NAME}/dclde/2026/dclde_2026_killer_whales/"
    
    # List all files recursively
    try:
        all_files = fs.glob(f"{prefix}**/*.wav")
    except Exception as e:
        print(f"Error listing with glob, trying ls: {e}")
        # Fallback to recursive ls
        all_files = []
        def list_recursive(path):
            try:
                items = fs.ls(path, detail=True)
                for item in items:
                    if item['type'] == 'directory':
                        list_recursive(item['name'])
                    elif item['name'].endswith('.wav'):
                        all_files.append(item['name'])
            except Exception as e2:
                print(f"  Error listing {path}: {e2}")
        list_recursive(prefix)
    
    # Convert to gs:// format for consistency
    existing_files = set()
    for f in all_files:
        if not f.startswith('gs://'):
            f = f'gs://{f}'
        existing_files.add(f)
    
    print(f"Found {len(existing_files)} .wav files in GCS")
    
    # Cache results
    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(existing_files, f)
    print(f"Cached file list to {cache_path}")
    
    return existing_files


def filter_annotations_by_existing_files(df, existing_files, gcs_path_column='gcs_path'):
    """
    Filter annotations DataFrame to only include rows where the audio file exists in GCS.
    
    Args:
        df: Annotations DataFrame with gcs_path column
        existing_files: Set of existing gs:// paths from get_existing_gcs_files()
        gcs_path_column: Name of the column containing GCS paths
    
    Returns:
        Filtered DataFrame
    """
    before_count = len(df)
    unique_paths_before = df[gcs_path_column].nunique()
    
    # Check which paths exist
    df_filtered = df[df[gcs_path_column].isin(existing_files)]
    
    after_count = len(df_filtered)
    unique_paths_after = df_filtered[gcs_path_column].nunique()
    
    removed_annotations = before_count - after_count
    removed_files = unique_paths_before - unique_paths_after
    
    print(f"Filtered annotations: {before_count} -> {after_count} ({removed_annotations} removed)")
    print(f"Unique files: {unique_paths_before} -> {unique_paths_after} ({removed_files} missing files removed)")
    
    return df_filtered


class GCSAudioLoader:
    def __init__(self, bucket_name, cache_dir=None, enable_cache=False):
        self.bucket_name = bucket_name
        self.target_sr = config.SAMPLE_RATE
        if enable_cache and cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(bucket_name)

    def _get_gcs_url(self, gcs_path):
        if gcs_path.startswith('http'): return gcs_path
        if gcs_path.startswith('gs://'): gcs_path = gcs_path.replace('gs://', '')
        if gcs_path.startswith(f"{self.bucket_name}/"):
            gcs_path = gcs_path[len(f"{self.bucket_name}/"):]
        return f"https://storage.googleapis.com/{self.bucket_name}/{gcs_path}"

    def load_audio_segment(self, gcs_path, offset_sec, duration_sec, orig_sr_hint=None):
        if not AUDIOSAMPLE_AVAILABLE: raise ImportError("AudioSample not available.")
        try:
            gcs_url = self._get_gcs_url(gcs_path)
            audio_sample = AudioSample(gcs_url)
            orig_sr = orig_sr_hint or (audio_sample.sample_rate if hasattr(audio_sample, 'sample_rate') else config.SAMPLE_RATE)
            
            segment = audio_sample[offset_sec:offset_sec + duration_sec]
            
            # Attempt native resampling
            if hasattr(segment, 'resample'):
                try: segment = segment.resample(self.target_sr)
                except: pass
            
            audio = segment.as_tensor()
            # Fallback resampling
            if hasattr(segment, 'sample_rate') and segment.sample_rate != self.target_sr:
                 audio = torchaudio.functional.resample(audio, segment.sample_rate, self.target_sr)
            elif orig_sr != self.target_sr and not hasattr(segment, 'resample'):
                 # Best effort if metadata missing
                 pass

            if audio.dim() > 1: audio = audio.mean(dim=0, keepdim=True)
            else: audio = audio.unsqueeze(0)
            
            return audio.numpy(), self.target_sr
        except Exception as e:
            print(f"ERROR: Failed to load {gcs_path}: {e}")
            return None, 0


class AudioAugmentations:
    def __init__(self, annotations_df=None):
        self.noise_segments = None
        if config.AUGMENT_AUDIO_NOISE and annotations_df is not None:
             noise_mask = annotations_df.get(config.ANNOTATION_CLASS_COLUMN, pd.Series()).astype(str).str.contains(config.NOISE_CLASS_STRING, case=False, na=False)
             if noise_mask.any(): self.noise_segments = annotations_df[noise_mask]
    
    def add_noise(self, audio, gcs_path):
        if not config.AUGMENT_AUDIO_NOISE or np.random.rand() > config.AUGMENT_AUDIO_NOISE_P: return audio
        return audio + np.random.normal(0, 0.01, audio.shape)
    
    def apply_gain(self, audio):
        if not config.AUGMENT_AUDIO_GAIN or np.random.rand() > config.AUGMENT_AUDIO_GAIN_P: return audio
        gain = 10 ** (np.random.uniform(-6, 6) / 20.0)
        return audio * gain


class AudioObjectDataset(Dataset):
    def __init__(self, df, annotations_df, audio_loader, is_train=True, use_beats=False, model_type=None):
        self.df_chips = df
        self.annotations_df = annotations_df
        self.audio_loader = audio_loader
        self.is_train = is_train
        self.use_beats = use_beats
        self.model_type = model_type or config.MODEL_TYPE.lower()
        self.class_column = config.ANNOTATION_CLASS_COLUMN
        self.audio_aug = AudioAugmentations(annotations_df) if is_train else None
        
        # NOTE: Spectrogram generation moved to GPU (in models.py) to save CPU/bandwidth
        # self.spec_aug removed from here as we don't have specs yet

    def __len__(self): return len(self.df_chips)

    def __getitem__(self, idx):
        row = self.df_chips.iloc[idx]
        y, sr = self.audio_loader.load_audio_segment(row['gcs_path'], row['chip_start_sec'], config.WINDOW_SEC)
        if y is None: return None, None
        
        if self.is_train and self.audio_aug:
            y = self.audio_aug.apply_gain(y)

        # Pad/Crop
        if len(y[0]) < config.CHIP_SAMPLES:
             y = np.pad(y, ((0,0), (0, config.CHIP_SAMPLES - len(y[0]))))
        else:
             y = y[:, :config.CHIP_SAMPLES]

        # Labels
        chip_labels = self.annotations_df[
            (self.annotations_df['gcs_path'] == row['gcs_path']) &
            (self.annotations_df['FileBeginSec'] < row['chip_start_sec'] + config.WINDOW_SEC) &
            (self.annotations_df['FileEndSec'] > row['chip_start_sec'])
        ]
        yolo_labels = utils.convert_labels_to_yolo(chip_labels, row['chip_start_sec'], config.WINDOW_SEC, class_column=self.class_column)
        labels = torch.tensor(yolo_labels, dtype=torch.float32)

        # Return raw audio as tensor [C, T] (C=1 usually)
        return torch.from_numpy(y).float(), labels


class CustomCollate:
    def __init__(self, use_beats=False):
        self.use_beats = use_beats

    def __call__(self, batch):
        batch = [b for b in batch if b[0] is not None]
        if not batch: return torch.tensor([]), []
        
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        # Audio: [1, T]
        # Check dimensions
        # If all samples are same length (ensured by __getitem__ padding), we can just stack
        # data is list of tensors [1, T]
        
        try:
            data = torch.stack(data)
        except RuntimeError:
             # Fallback if lengths differ for some reason
             max_len = max([d.shape[-1] for d in data])
             data = torch.stack([F.pad(d, (0, max_len - d.shape[-1])) for d in data])
             
        # Result: [B, 1, T]
        if self.use_beats:
             # BEATS expects [B, T] (no channel dim if mono?) or [B, C, T]
             # If [B, 1, T], let's squeeze channel if BEATS expects so.
             # BEATS model usually takes [B, T].
             data = data.squeeze(1)
             
        return data, labels


def create_full_annotation_df():
    print("Loading annotations...")
    fs = GCSFileSystem()
    with fs.open(config.GCS_ANNOTATION_PATH) as f:
        df = pd.read_csv(f)
    
    print(f"Loaded annotations with columns: {df.columns.tolist()}")
    
    # Auto-detect class column if missing
    if config.ANNOTATION_CLASS_COLUMN not in df.columns:
        possible_cols = [c for c in df.columns if 'species' in c.lower() or 'class' in c.lower() or 'label' in c.lower() or 'call' in c.lower()]
        if possible_cols:
            print(f"WARNING: '{config.ANNOTATION_CLASS_COLUMN}' not found. Using '{possible_cols[0]}' instead.")
            config.ANNOTATION_CLASS_COLUMN = possible_cols[0]
        else:
            print(f"ERROR: '{config.ANNOTATION_CLASS_COLUMN}' not found and no obvious alternative detected. Available columns: {df.columns.tolist()}")
    
    df = df[~df['Provider'].isin(config.PROVIDERS_TO_EXCLUDE)]
    df = df[(df['LowFreqHz'] < config.MAX_FREQ_HZ) & 
            (df['HighFreqHz'] > config.MIN_FREQ_HZ) & 
            ((df['HighFreqHz'] - df['LowFreqHz']) > 1)]
            
    print("Mapping paths...")
    gcs_base = f"gs://{config.GCS_AUDIO_BUCKET_NAME}/dclde/2026/dclde_2026_killer_whales"
    
    def map_path(row):
        path = str(row['FilePath'])
        
        def fix_casing(p):
            p = p.replace('\\', '/')
            parts = p.split('/')
            # Lowercase directories, preserve filename case
            if len(parts) > 1:
                return '/'.join([part.lower() for part in parts[:-1]] + [parts[-1]])
            return p

        if 'DFO_CRP' in path:
            return (gcs_base + '/dfo_crp/' + fix_casing(path.split('DFO_CRP/')[-1]))
        if 'UAF_NGOS' in path:
            return (gcs_base + '/uaf/' + fix_casing(path.split('UAF/')[-1]))
        
        prov = str(row['Provider']).lower()
        if prov == 'nan': return None
        
        # Mapping for provider names that don't match directory names exactly
        prov_map = {
            'jasco_vfpa': 'vfpa',
            'smruconsulting': 'smru',
            'jasco_vfpa_onc': 'vfpa', 
            'uaf_ngos': 'uaf',
        }
        prov = prov_map.get(prov, prov)

        # Fix double slashes and ensure clean path joining
        suffix = fix_casing(path.split('Audio/')[-1])
        suffix = suffix.lstrip('/')
        
        while '//' in suffix:
            suffix = suffix.replace('//', '/')

        # Apply directory/filename mappings
        suffix = suffix.replace('straitofgeorgia/', 'straitofgeorgia_globus-robertsbank/')
        suffix = suffix.replace('lime kiln', 'lime-kiln')
            
        return (gcs_base + '/' + prov + '/audio/' + suffix)

    df['gcs_path'] = df.apply(map_path, axis=1)
    df = df.dropna(subset=['gcs_path'])
    print(f"Mapped {len(df)} annotations to GCS paths")
    
    # Filter out annotations for files that don't exist in GCS
    existing_files = get_existing_gcs_files()
    df = filter_annotations_by_existing_files(df, existing_files)
    
    print(f"Final annotation count: {len(df)}")
    return df


def refresh_gcs_file_cache():
    """Force refresh the GCS file cache. Call this if files have been added/removed."""
    return get_existing_gcs_files(force_refresh=True)


def create_chip_list(df_ann, cache_path='chip_list_cache.pkl', force_refresh=False):
    if os.path.exists(cache_path) and not force_refresh:
        print(f"Loading cached chip list from {cache_path}")
        cached = pd.read_pickle(cache_path)
        # Validate cache against current annotation files
        cached_files = set(cached['gcs_path'].unique())
        current_files = set(df_ann['gcs_path'].unique())
        if cached_files == current_files:
            return cached
        else:
            print(f"Cache outdated (files changed). Regenerating...")
            # Fall through to regenerate

    print("Creating chips...")
    file_durations = df_ann.groupby('gcs_path')['FileEndSec'].max()
    chip_list = []
    for gcs_path, max_time in file_durations.items():
        group = df_ann[df_ann['gcs_path'] == gcs_path].iloc[0][config.SPLIT_GROUP_COLUMN]
        # Use simple range
        starts = np.arange(0, max_time - config.WINDOW_SEC, config.WINDOW_SEC / 2)
        for s in starts:
            chip_list.append({'gcs_path': gcs_path, 'chip_start_sec': s, 'group': group})
    
    df_chips = pd.DataFrame(chip_list)
    print(f"Saving chip list to {cache_path}")
    df_chips.to_pickle(cache_path)
    return df_chips


def get_dataloaders(df_ann, df_chips, fold_id=0, use_beats=False, model_type=None):
    gkf = GroupKFold(n_splits=config.N_SPLITS)
    splits = list(gkf.split(df_chips, groups=df_chips['group']))
    
    val_idx = splits[fold_id][1]
    test_idx = splits[(fold_id + 1) % config.N_SPLITS][1]
    train_idx = list(set(range(len(df_chips))) - set(val_idx) - set(test_idx))
    
    if config.DEBUG:
         train_idx = train_idx[:config.DEBUG_MAX_SAMPLES]
         val_idx = val_idx[:config.DEBUG_MAX_SAMPLES]
         test_idx = test_idx[:config.DEBUG_MAX_SAMPLES]
    
    loader = GCSAudioLoader(config.GCS_AUDIO_BUCKET_NAME)
    
    datasets = {
        'train': AudioObjectDataset(df_chips.iloc[train_idx], df_ann, loader, True, use_beats, model_type),
        'val': AudioObjectDataset(df_chips.iloc[val_idx], df_ann, loader, False, use_beats, model_type),
        'test': AudioObjectDataset(df_chips.iloc[test_idx], df_ann, loader, False, use_beats, model_type)
    }
    
    collate = CustomCollate(use_beats)
    
    return (
        DataLoader(datasets['train'], config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, collate_fn=collate),
        DataLoader(datasets['val'], config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=collate),
        DataLoader(datasets['test'], config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=collate)
    )
