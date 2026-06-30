"""Upload prepared YOLO dataset to S3 for EC2 training."""
import argparse
import os
import boto3
from pathlib import Path


S3_BUCKET = "deepvoice-external"
S3_PREFIX = "humpback_yolo_detection"


def upload_directory(local_dir: str, bucket: str, prefix: str):
    """Upload a local directory to S3 recursively."""
    s3 = boto3.client("s3")
    local_path = Path(local_dir)

    files = [f for f in local_path.rglob("*") if f.is_file()]
    print(f"Uploading {len(files)} files to s3://{bucket}/{prefix}/")

    for i, filepath in enumerate(files):
        relative = filepath.relative_to(local_path)
        s3_key = f"{prefix}/{relative.as_posix()}"
        s3.upload_file(str(filepath), bucket, s3_key)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)} uploaded...")

    print(f"Done. Dataset at s3://{bucket}/{prefix}/")


def main():
    parser = argparse.ArgumentParser(description="Upload YOLO dataset to S3")
    parser.add_argument("--local-dir", required=True, help="Local dataset directory")
    parser.add_argument("--bucket", default=S3_BUCKET)
    parser.add_argument("--prefix", default=S3_PREFIX)
    args = parser.parse_args()

    upload_directory(args.local_dir, args.bucket, args.prefix)


if __name__ == "__main__":
    main()
