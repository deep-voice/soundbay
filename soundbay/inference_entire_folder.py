import os
from pathlib import Path

# ——— CONFIG ———
folder_path = Path("../datasets/Manatees/wav/")
batch_size = 5  # set to 1 if you want to run one-at-a-time

# ——— inference function ———
def perform_inference(wav_paths):
    for wav in wav_paths:
        cmd = (
            f"python inference.py "
            f"--config-name runs/inference_single_audio_manatees "
            f"experiment.checkpoint.path=\"../datasets/Manatees/best.pth\" "
            f"data.test_dataset.file_path=\"{wav}\""
        )
        ret = os.system(cmd)
        if ret != 0:
            print(f"[ERROR] inference failed on {wav}")
        else:
            wav.unlink()  # delete after successful inference

# ——— gather all .wav files ———
all_wavs = sorted(folder_path.glob("*.wav"))
if not all_wavs:
    print(f"No .wav files found in {folder_path}")
    exit(0)

# ——— process in batches ———
for i in range(0, len(all_wavs), batch_size):
    batch = all_wavs[i : i + batch_size]
    print(f"Processing batch {i//batch_size + 1}: {[p.name for p in batch]}")
    perform_inference(batch)
