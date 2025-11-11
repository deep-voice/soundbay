import pandas as pd
import soundfile as sf
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

# --- Configuration ---

# 1. SET THIS to the path of the 'Manatee_fulltraining_final' directory
DATA_DIR = Path(
    r"C:\Users\amitg\OneDrive\Documents\Deep_Voice\HF_WAV_Manatee_Samples\edi.2108.2\Manatee_fulltraining_final")

# 2. SET THIS to where you want to save the new metadata file
OUTPUT_CSV = Path("../datasets/Manatees/edi_manatee_metadata.csv")

# 3. Define your labels. (0 is typically background/noise)
LABEL_MAP = {
    "Noise": 0,
    "MV": 1  # Manatee Vocalization
}

# 4. Define your train/validation split ratio
VAL_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------

def create_metadata():
    """
    Scans the data directory and creates a metadata DataFrame compatible
    with the BaseDataset in data.py.
    """
    if not DATA_DIR.exists():
        print(f"Error: Data directory not found at: {DATA_DIR}")
        print("Please update the 'DATA_DIR' variable in this script.")
        sys.exit(1)

    print(f"Scanning {DATA_DIR} for .wav files...")

    # Find all .wav files recursively
    wav_files = list(DATA_DIR.rglob("*.wav"))

    if not wav_files:
        print(f"Error: No .wav files found in {DATA_DIR} or its subdirectories.")
        sys.exit(1)

    print(f"Found {len(wav_files)} total .wav files.")

    metadata_list = []
    for f in wav_files:
        try:
            # Get the parent folder name (e.g., 'MV' or 'Noise')
            parent_folder = f.parent.name

            # Get label from the folder name
            label = LABEL_MAP.get(parent_folder)
            if label is None:
                print(f"Skipping file with unknown parent folder: {f}")
                continue

            # Get audio duration
            info = sf.info(f)
            duration_seconds = info.duration

            # This 'filename' format is critical.
            # It must match the key format created by _create_audio_dict
            # when path_hierarchy=1. Format: "parent_folder/file_stem"
            filename_key = f"{parent_folder}/{f.stem}"

            metadata_list.append({
                "filename": filename_key,
                "begin_time": 0.0,
                "end_time": duration_seconds,
                "call_length": duration_seconds,
                "label": label
            })
        except Exception as e:
            print(f"Error processing file {f}: {e}")

    if not metadata_list:
        print("Error: No valid audio files were processed.")
        return

    df = pd.DataFrame(metadata_list)

    print(f"Successfully processed {len(df)} files.")
    print("\nLabel distribution:")
    print(df['label'].value_counts())

    # --- Create train/val split ---
    print(f"\nCreating train/val split with {VAL_SIZE * 100}% validation...")

    # Use stratify to ensure both train and val sets have a similar
    # distribution of classes (MV vs. Noise)
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['label']
    )

    # Add the 'split_type' column your dataloader expects
    train_df['split_type'] = 'train'
    val_df['split_type'] = 'val'

    final_df = pd.concat([train_df, val_df]).reset_index(drop=True)

    # Save to CSV
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSuccessfully created metadata file at: {OUTPUT_CSV}")
    print("\nFinal split distribution:")
    print(final_df.groupby('split_type')['label'].value_counts())


if __name__ == "__main__":
    create_metadata()