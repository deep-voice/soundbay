import os
import boto3

s3_bucket = "deepvoice-user-uploads"
s3_prefix = "eric.angel.ramos@gmail.com/dropbox/Placencia1/"
folder_path = "../datasets/Manatees/sian_kaan/wav/"
batch_size = 5  # Number of files to process in each batch
file_list_path = "../datasets/Manatees/files_without_inference.txt"

# Initialize the S3 client
s3_client = boto3.client("s3")

# Function to perform inference on a batch of WAV files
def perform_inference(file_paths):
    for wav_file_path in file_paths:
        command = f"python inference.py --config-name runs/inference_single_audio_manatees experiment.checkpoint.path=\"../datasets/Manatees/best.pth\" data.test_dataset.file_path=\"{wav_file_path}\""
        os.system(command)
        os.remove(wav_file_path)  # Delete the processed file

# Read the list of files without inference
with open(file_list_path, "r") as file:
    files_without_inference = file.read().splitlines()

# Get the list of WAV files from the S3 bucket
# s3_files = []
# paginator = s3_client.get_paginator("list_objects_v2")
# page_iterator = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)
s3_files = [os.path.join(s3_prefix, file_name) for file_name in files_without_inference]

# for page in page_iterator:
#     if "Contents" in page:
#         s3_files.extend([obj["Key"] for obj in page["Contents"]])

# Iterate over the WAV files in batches
for i in range(0, len(s3_files), batch_size):
    # Get the current batch of WAV files
    batch_files = s3_files[i:i+batch_size]

    # Download the batch of files from S3
    for s3_file in batch_files:
        file_name = os.path.basename(s3_file)
        local_file_path = os.path.join(folder_path, file_name)
        s3_client.download_file(s3_bucket, s3_file, local_file_path)

    # Perform inference on the downloaded batch
    batch_file_paths = [os.path.join(folder_path, os.path.basename(file_path)) for file_path in batch_files]
    perform_inference(batch_file_paths)
