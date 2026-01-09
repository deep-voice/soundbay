import os
import csv
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import sys
import argparse
import shutil

# Configuration
CSV_FILE = os.path.expanduser("~/soundbay/shaye_annotations_3.1.26_extended.csv")
DEST_DIR = os.path.expanduser("~/soundbay/datasets/shaye_data_extended")
MAX_WORKERS = 10 
MIN_FREE_SPACE_BYTES = 200 * 1024 * 1024  # 200 MB

def check_disk_space(path):
    # Ensure path exists or check parent
    if not os.path.exists(path):
        return shutil.disk_usage(os.path.dirname(path)).free
    return shutil.disk_usage(path).free

def get_s3_client():
    # Use default credential chain (env vars, config file, IAM role, etc.)
    return boto3.client("s3")

def download_file(args):
    s3_url, dest_path, s3_client = args
    
    if os.path.exists(dest_path):
        return  # Skip if already exists

    # Check disk space (approximate check in thread)
    try:
        dest_dir = os.path.dirname(dest_path)
        if not os.path.exists(dest_dir):
             # Try to check parent if dir doesn't exist yet (though we created it in main)
             dest_dir = os.path.dirname(dest_dir)
        
        if shutil.disk_usage(dest_dir).free < MIN_FREE_SPACE_BYTES:
             raise OSError("Low disk space")
    except Exception:
        pass # Ignore check errors in thread if any, main loop handles global check better, but this is a safety net

    try:
        parsed_url = urlparse(s3_url)
        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip('/')

        s3_client.download_file(bucket_name, object_key, dest_path)
    except ClientError as e:
        print(f"\nError downloading {s3_url}: {e}")
    except Exception as e:
        print(f"\nUnexpected error for {s3_url}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download wav files from S3 based on CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without downloading files.")
    args = parser.parse_args()

    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file not found at {CSV_FILE}")
        return

    print(f"Reading CSV: {CSV_FILE}")
    s3_paths = set()
    try:
        with open(CSV_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 's3_path' not in reader.fieldnames:
                print("Error: 's3_path' column not found in CSV.")
                return
            for row in reader:
                if row['s3_path']:
                    s3_paths.add(row['s3_path'])
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Convert to list
    s3_paths = list(s3_paths)
    print(f"Found {len(s3_paths)} unique files to check/download.")

    if not os.path.exists(DEST_DIR):
        if not args.dry_run:
            print(f"Creating destination directory: {DEST_DIR}")
            os.makedirs(DEST_DIR, exist_ok=True)
        else:
            print(f"Destination directory {DEST_DIR} does not exist (will be created).")

    # Prepare list of files to download
    files_to_download = []
    
    print("Checking for existing files...")
    for s3_url in s3_paths:
        parsed_url = urlparse(s3_url)
        filename = os.path.basename(parsed_url.path)
        dest_path = os.path.join(DEST_DIR, filename)
        
        if not os.path.exists(dest_path):
            files_to_download.append((s3_url, dest_path))
    
    print(f"Files already existing: {len(s3_paths) - len(files_to_download)}")
    print(f"Files to download: {len(files_to_download)}")

    if not files_to_download:
        print("All files are already present.")
        return

    if args.dry_run:
        print("Dry run completed. No files were downloaded.")
        return

    s3_client = get_s3_client()
    
    # Add client to args
    download_args = [(url, path, s3_client) for url, path in files_to_download]

    print(f"Starting download with {MAX_WORKERS} workers...")
    
    # Check disk space initially
    if check_disk_space(DEST_DIR) < MIN_FREE_SPACE_BYTES:
        print(f"Error: Not enough disk space. Requires at least 200MB free.")
        return

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # map returns an iterator, we iterate to trigger downloads and exceptions
            for _ in tqdm(executor.map(download_file, download_args), total=len(download_args), unit="file"):
                pass
    except OSError as e:
        print(f"\nStopped: {e}")
        print("Download aborted due to low disk space.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")

    print("\nDownload process completed.")

if __name__ == "__main__":
    main()
