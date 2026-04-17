import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# Path to your fixed CSV from the previous step
csv_path = r'C:\Users\amitg\OneDrive\Documents\Deep_Voice\HF_WAV_Manatee_Samples\edi.2108.2\edi_manatee_metadata_fixed.csv'
# Path to the root folder containing your audio files (the parent of Noise_16k and MV)
audio_root = Path(r'C:\Users\amitg\OneDrive\Documents\Deep_Voice\HF_WAV_Manatee_Samples\edi.2108.2\Manatee_fulltraining_final')
# ---------------------

df = pd.read_csv(csv_path)

# Create a dictionary to map filenames to their full paths (handling the subdirectory structure)
# This mimics the logic in data.py to ensure we find the files
print("Indexing audio files...")
audio_files = {f.name: f for f in audio_root.rglob('*.wav')}

changed_count = 0
print("Checking and fixing durations...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    # Extract the filename part (e.g., removing 'Noise_16k/' if it's still there,
    # or just matching the name if your CSV is already fixed)
    fname = Path(row['filename']).name

    if fname in audio_files:
        file_path = audio_files[fname]
        try:
            # Get actual duration from the file
            actual_duration = sf.info(str(file_path)).duration

            # If metadata end_time is beyond the file duration, clamp it
            if row['end_time'] > actual_duration:
                # Update the dataframe
                df.at[idx, 'end_time'] = actual_duration

                # If it's a full-clip segment, we might want to fix begin_time too if it's weird,
                # but usually just clamping end_time is enough.
                changed_count += 1
        except Exception as e:
            print(f"Error reading {fname}: {e}")
    else:
        print(f"Warning: Could not find file {fname} in {audio_root}")

print(f"Fixed {changed_count} rows where end_time exceeded duration.")

# Save the corrected CSV
df.to_csv('edi_manatee_metadata_clamped.csv', index=False)
print("Saved fixed metadata to 'edi_manatee_metadata_clamped.csv'")
