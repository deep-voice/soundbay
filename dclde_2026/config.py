import torch

# --- GCS & Data Paths ---
GCS_ANNOTATION_PATH = "gs://noaa-passive-bioacoustic/dclde/2026/dclde_2026_killer_whales/Annotations.csv"
GCS_AUDIO_BUCKET_NAME = "noaa-passive-bioacoustic"
LOCAL_AUDIO_CACHE_DIR = "./audio_cache" # For on-demand download
ENABLE_AUDIO_CACHE = False  # Set to False to disable caching (load directly from GCS/ffmpeg)

# --- Data Preprocessing ---
# To capture 125kHz, we MUST use a 250kHz sample rate (Nyquist theorem)
SAMPLE_RATE = 32000  # Target sample rate (audio will be resampled to this)
SPECTROGRAM_TYPE = 'linear' # 'mel' is WRONG for this task
MIN_FREQ_HZ = 0   # Min freq to consider
MAX_FREQ_HZ = 16000 # Max freq to consider (will be adjusted based on model type)
# If bounding box upper limit exceeds MAX_FREQ_HZ, it will be clipped to MAX_FREQ_HZ
#
# Model-specific frequency limits:
# - BEATS: Resamples to 16kHz internally, so max freq = 8kHz (Nyquist)
# - YOLO: Uses spectrograms from full SAMPLE_RATE, so max freq = SAMPLE_RATE / 2 = 125kHz
# MAX_FREQ_HZ will be automatically adjusted based on MODEL_TYPE

# Original sample rates may vary. AudioSample will load at original SR, then resample to SAMPLE_RATE.
# If your annotations CSV has a 'SampleRate' column, it can be used for resampling.
# Otherwise, the system will attempt to infer it or use a fallback.
#
# IMPORTANT: If orig_sr < SAMPLE_RATE, upsampling will occur. Upsampling interpolates samples
# but does NOT add new frequency information. To capture frequencies up to MAX_FREQ_HZ, you need
# orig_sr >= 2 * MAX_FREQ_HZ (Nyquist theorem). Consider filtering out files with insufficient SR.
MIN_REQUIRED_SAMPLE_RATE = None  # Set to e.g., 250000 to filter files with lower SR

# Spectrogram parameters (only used for YOLO, not BEATS)
# BEATS processes raw audio with its own learned frontend, so these don't apply to BEATS
WINDOW_SEC = 5.0 # Duration of each audio "chip"
CHIP_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)

# Multi-channel spectrogram config (YOLO only)
# We'll use 3 different FFT sizes for 3 time-freq trade-offs
N_FFT_LIST = [512, 1024, 2048]
# Use a larger hop_length to keep time dimension manageable
HOP_LENGTH = 1024 

# The largest n_fft (2048) defines our max resolution
# (n_fft / 2) + 1 = 1025 bins.
# All spectrograms will be resized to this height.
MAX_N_FFT = 2048
TARGET_FREQ_BINS = 1024 # Resize to 1024 (multiple of 32) instead of 1025

# --- Data Filtering ---
# Column name for the class label in the annotations CSV
ANNOTATION_CLASS_COLUMN = 'ClassSpecies' 

# Value in annotation class column that represents noise/background to be excluded or used for augmentation
# If this string is found (case-insensitive) in the class column, it may be used to extract noise profiles
NOISE_CLASS_STRING = 'UndBio'

# Providers with median BB height of 0 are not useful for detection
PROVIDERS_TO_EXCLUDE = ["SIMRES", "SIO", "OrcaSound"]
# Target classes (actual labels from annotations)
CLASSES = ['AB', 'HW', 'KW', 'UndBio']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# --- Training ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
# Use GroupKFold split on 'Dataset' or 'Provider' column for generalization
# This ensures test/val contain different files from different locations
SPLIT_GROUP_COLUMN = 'Dataset'  # Group by Dataset to ensure different files/locations
N_SPLITS = 5
TEST_FOLD = 0 # Hold out fold 0 for testing
# Validation will use fold 1, ensuring different files from test

# --- Model ---
# Model type: 'beats' or 'yolo'
MODEL_TYPE = 'yolo'  # Options: 'beats' or 'yolo'
# Note: MAX_FREQ_HZ will be automatically adjusted in train.py based on MODEL_TYPE:
# - BEATS: Limited to 8kHz (resamples to 16kHz internally, Nyquist = 8kHz)
#          BEATS uses its own learned audio frontend, we don't control spectrograms
# - YOLO: Can use full SAMPLE_RATE / 2 = 125kHz (uses spectrograms we generate)
#          YOLO uses spectrograms we control (N_FFT_LIST, HOP_LENGTH, etc.)

# For BEATS Transformer
TRANSFORMER_MODEL_NAME = 'facebook/beats-base'  # BEATS transformer
USE_BEATS = True  # Deprecated: use MODEL_TYPE instead

# For YOLO
YOLO_MODEL_SIZE = 'n'  # Options: 'n', 's', 'm', 'l', 'x' (nano, small, medium, large, xlarge)
YOLO_IMG_SIZE = 1024  # Only used when calling YOLO's native training API (it will resize internally)
YOLO_PREPROC_DIR = "./yolo_dataset"
# Note: We keep natural spectrogram dimensions in our dataset (no pre-resizing)
# When using YOLO's native API with preprocessed data, YOLO will resize internally
# When loading from GCS, we use natural dimensions and YOLO_IMG_SIZE is not used

# --- Augmentations ---
# Audio-level augmentations (applied before spectrogram)
AUGMENT_AUDIO_NOISE = True  # Enable noise augmentation
AUGMENT_AUDIO_NOISE_P = 0.5  # Probability of applying noise
AUGMENT_AUDIO_GAIN = True  # Enable gain/volume augmentation
AUGMENT_AUDIO_GAIN_P = 0.5  # Probability of applying gain
AUGMENT_AUDIO_GAIN_MIN_DB = -6.0
AUGMENT_AUDIO_GAIN_MAX_DB = 6.0

# Spectrogram-level augmentations (applied to spectrogram image)
AUGMENT_SPEC_SPECAUGMENT = True  # Enable SpecAugment
AUGMENT_SPEC_SPECAUGMENT_P = 0.5  # Probability of applying SpecAugment
AUGMENT_SPEC_FREQ_MASK = True  # Enable frequency masking within SpecAugment
AUGMENT_SPEC_FREQ_MASK_MAX = 10  # Max frequency bins to mask
AUGMENT_SPEC_TIME_MASK = True  # Enable time masking within SpecAugment
AUGMENT_SPEC_TIME_MASK_MAX = 20  # Max time frames to mask

# Mixup/CutMix augmentations
AUGMENT_MIXUP = True  # Enable Mixup
AUGMENT_MIXUP_P = 0.3  # Probability of applying Mixup
AUGMENT_MIXUP_ALPHA = 0.2  # Beta distribution parameter
AUGMENT_CUTMIX = True  # Enable CutMix
AUGMENT_CUTMIX_P = 0.3  # Probability of applying CutMix
AUGMENT_CUTMIX_ALPHA = 1.0  # Beta distribution parameter

# --- WandB Logging ---
USE_WANDB = True  # Enable Weights & Biases logging
WANDB_PROJECT = "dclde_2026"  # WandB project name
WANDB_ENTITY = None  # WandB entity/team (None for personal)
LOG_SPECTROGRAMS_EVERY_N_EPOCHS = 5  # Log spectrogram samples every N epochs
NUM_SPECTROGRAM_SAMPLES = 4  # Number of spectrogram samples to log

# --- Debug Mode ---
DEBUG = True  # Enable debug mode for testing (smaller dataset, verbose logging, fewer epochs)
DEBUG_MAX_SAMPLES = 100  # Maximum number of samples per split in debug mode
DEBUG_EPOCHS = 2  # Number of epochs to run in debug mode

