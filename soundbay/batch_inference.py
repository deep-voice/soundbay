import os
import boto3

# --- Configuration ---
s3_bucket = "deepvoice-user-uploads"
# IMPORTANT: Set this to the parent folder you want to process recursively
s3_prefix = "biobbrady@gmail.com/dropbox/Cameroon_Ntem_River/"
# A local folder to temporarily download files to
local_download_path = "./temp_audio_for_inference/"
batch_size = 5  # Number of files to process in each batch

# --- Main Script ---

# Ensure the local download directory exists
os.makedirs(local_download_path, exist_ok=True)

# Initialize the S3 client
s3_client = boto3.client("s3")


# Function to perform inference on a batch of WAV files
def perform_inference(file_paths):
    for wav_file_path in file_paths:
        print(f"-> Running inference on: {os.path.basename(wav_file_path)}")
        command = f'python inference.py --config-name runs/inference_single_audio_manatees experiment.checkpoint.path="../datasets/Manatees/best.pth" data.test_dataset.file_path="{wav_file_path}"'
        os.system(command)
        # Clean up the downloaded file after processing
        print(f"-> Deleting local file: {wav_file_path}")
        os.remove(wav_file_path)


# --- Step 1: Get the list of all WAV files from S3 recursively ---
print(f"Fetching file list from s3://{s3_bucket}/{s3_prefix}")
s3_object_keys = []
paginator = s3_client.get_paginator("list_objects_v2")
page_iterator = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)

for page in page_iterator:
    if "Contents" in page:
        # Filter for .wav files and ignore any "folder objects"
        for obj in page["Contents"]:
            if obj["Key"].lower().endswith((".wav", ".flac")):  # Or other audio types
                s3_object_keys.append(obj["Key"])

if not s3_object_keys:
    print("No .wav files found in the specified S3 path. Exiting.")
    exit()

print(f"Found {len(s3_object_keys)} audio files to process.")

# --- Step 2: Iterate over the WAV files in batches ---
for i in range(0, len(s3_object_keys), batch_size):
    # Get the current batch of S3 keys (e.g., 'folder/subfolder/audio.wav')
    batch_keys = s3_object_keys[i:i + batch_size]
    batch_num = (i // batch_size) + 1
    print(f"\n--- Processing Batch {batch_num} of {-(-len(s3_object_keys) // batch_size)} ---")

    downloaded_file_paths = []
    # Download the batch of files from S3
    for s3_key in batch_keys:
        file_name = os.path.basename(s3_key)
        local_file_path = os.path.join(local_download_path, file_name)

        print(f"Downloading s3://{s3_bucket}/{s3_key} to {local_file_path}")
        s3_client.download_file(s3_bucket, s3_key, local_file_path)
        downloaded_file_paths.append(local_file_path)

    # Perform inference on the newly downloaded batch
    perform_inference(downloaded_file_paths)

print("\n--- Batch inference complete. ---")