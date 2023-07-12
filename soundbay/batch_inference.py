# This script is for when we only want to run inference on a large dataset of wav files that is too heavy for the machine to handle.
import os
import boto3

s3_bucket = "deepvoice-user-uploads"
s3_prefix = "eric.angel.ramos@gmail.com/dropbox/Placencia1/"
folder_path = "../datasets/Manatees/placencia/"
batch_size = 5  # Number of files to process in each batch

s3_client = boto3.client("s3")
# Get a list of all wav files in the folder
s3_files = []
paginator = s3_client.get_paginator("list_objects_v2")
page_iterator = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)
for page in page_iterator:
    if "Contents" in page:
        s3_files.extend([obj["Key"] for obj in page["Contents"]])

# Iterate over the WAV files in batches
for i in range(0, len(s3_files), batch_size):
    # Get the current batch of WAV files
    batch_files = s3_files[i:i+batch_size]

    # Download the batch of files from S3
    print("downloading batch ", i)
    for s3_file in batch_files:
        file_name = os.path.basename(s3_file)
        local_file_path = os.path.join(folder_path, file_name)
        s3_client.download_file(s3_bucket, s3_file, local_file_path)

    # Perform inference on the downloaded batch
    batch_file_paths = [os.path.join(folder_path, os.path.basename(file_path)) for file_path in batch_files]
    for wav_file_path in batch_file_paths:
        command = f"python inference.py --config-name runs/inference_single_audio_manatees experiment.checkpoint.path=\"../datasets/Manatees/best.pth\" data.test_dataset.file_path=\"{wav_file_path}\""
        os.system(command)
        os.remove(wav_file_path)  # Delete the processed file
