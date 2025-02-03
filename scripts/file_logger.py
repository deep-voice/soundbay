import json
import logging
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Optional


class FileLogger:
    def __init__(self, log_folder: Path, log_filename: str, s3_bucket: Optional[str] = None):
        self.log_folder = Path(log_folder)
        self.log_folder.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_folder / log_filename
        self.s3_bucket = s3_bucket
        self.logger = logging.getLogger(__name__)

        self.log_data = {
            "start_time": datetime.now().isoformat(),
            "processed_files": [],
            "errors": [],
            "stats": {
                "total_files_processed": 0,
                "successful_downloads": 0,
                "failed_downloads": 0,
                "successful_conversions": 0,
                "failed_conversions": 0,
                "successful_predictions": 0,
                "failed_predictions": 0
            }
        }

    def log_file_event(self, file_id: str, file_name: str, status: str,
                       stage: str, error: Optional[str] = None) -> None:
        """Log a file processing event."""
        entry = {
            "file_id": file_id,
            "file_name": file_name,
            "status": status,
            "stage": stage,
            "timestamp": datetime.now().isoformat()
        }

        if error:
            entry["error"] = error
            self.log_data["errors"].append(entry)

        self.log_data["processed_files"].append(entry)

        # Update statistics
        if stage == "download":
            if status == "success":
                self.log_data["stats"]["successful_downloads"] += 1
            else:
                self.log_data["stats"]["failed_downloads"] += 1
        elif stage == "conversion":
            if status == "success":
                self.log_data["stats"]["successful_conversions"] += 1
            else:
                self.log_data["stats"]["failed_conversions"] += 1
        elif stage == "prediction":
            if status == "success":
                self.log_data["stats"]["successful_predictions"] += 1
            else:
                self.log_data["stats"]["failed_predictions"] += 1

        self.log_data["stats"]["total_files_processed"] = len(set(
            entry["file_name"] for entry in self.log_data["processed_files"]
        ))

    def save_and_upload(self) -> None:
        """Save log file locally and upload to S3 if configured."""
        self.log_data["end_time"] = datetime.now().isoformat()

        with open(self.log_path, 'w') as f:
            json.dump(self.log_data, f, indent=2)

        self.logger.info(f"Log file saved to {self.log_path}")

        if self.s3_bucket:
            try:
                subprocess.check_call(
                    f'aws s3 cp {self.log_path} s3://{self.s3_bucket}/logs/{self.log_path.name}',
                    shell=True
                )
                self.logger.info(f"Log file uploaded to s3://{self.s3_bucket}/logs/{self.log_path.name}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to upload log file to S3: {str(e)}")