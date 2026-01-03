import os
import boto3
import pandas as pd
import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnnotationProcessor:
    """
    A class to process annotation files, checking against S3 storage for validity.
    """

    def __init__(self, s3_bucket: str, s3_prefix: str, aws_profile: Optional[str] = None):
        """
        Initialize the AnnotationProcessor.

        Args:
            s3_bucket: The name of the S3 bucket.
            s3_prefix: The prefix (folder path) in the S3 bucket to search for audio files.
            aws_profile: Optional AWS profile name to use.
        """
        self.bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip('/') + '/' # Ensure trailing slash
        
        # Initialize S3 client
        session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
        self.s3_client = session.client('s3')
        
        # Pre-fetch list of folders to optimize lookups
        self.folders_list = self._get_folders()

    def _get_folders(self) -> List[str]:
        """
        Retrieve a list of subfolders within the configured S3 prefix.
        
        Returns:
            List of folder names.
        """
        logger.info(f"Fetching folders from s3://{self.bucket}/{self.s3_prefix}...")
        folders = []
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket, 
                Prefix=self.s3_prefix, 
                Delimiter='/'
            )

            for page in page_iterator:
                for prefix in page.get('CommonPrefixes', []):
                    folder_path = prefix.get('Prefix')
                    # Extract just the immediate folder name relative to s3_prefix
                    # Logic matches original: remove the prefix from the full folder path
                    relative_folder = folder_path.replace(self.s3_prefix, '', 1) 
                    folders.append(relative_folder)
            
            logger.info(f"Found {len(folders)} folders.")
            return folders
        except Exception as e:
            logger.error(f"Error fetching folders: {e}")
            return []

    def check_file_exists_in_s3(self, filename: str) -> Optional[str]:
        """
        Check if a file exists in any of the known S3 subfolders.

        Args:
            filename: The name of the file to check.

        Returns:
            The full S3 path if found, otherwise None.
        """
        for folder in self.folders_list:
            # Construct candidate key
            # Let's stick strictly to constructing the key:
            full_key = f"{self.s3_prefix}{folder}{filename}"
            
            try:
                self.s3_client.head_object(Bucket=self.bucket, Key=full_key)
                return f"s3://{self.bucket}/{full_key}"
            except Exception:
                continue
        
        return None

    def get_local_annotation_files(self, directory: str) -> Dict[str, pd.DataFrame]:
        """
        recursively find .txt annotation files in the directory.
        Filters for files containing 'sounds' and NOT 'quiet'.
        
        Args:
            directory: Root directory to search.

        Returns:
            Dictionary mapping filename to DataFrame.
        """
        directory_path = Path(directory)
        all_dfs = {}
        
        logger.info(f"Scanning {directory} for annotation files...")
        
        # Walk through directory
        files_found = 0
        for path in directory_path.rglob('*.txt'):
            path_str = str(path)
            # Filter logic
            if ('sounds' in path_str.lower() and 'quiet' not in path_str.lower()) or ('adupdate' in path_str.lower() ):
                try:
                    df = pd.read_csv(path, sep='\t')
                    all_dfs[path.name] = df
                    files_found += 1
                except Exception as e:
                    logger.warning(f"Failed to read {path.name}: {e}")
            else:
                logger.warning(f"Skipping {path.name}: does not contain 'sounds' or 'ADupdate'")

        logger.info(f"Loaded {files_found} valid annotation files.")
        return all_dfs

    def process_single_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a raw annotation DataFrame to standardize columns and types.
        """
        try:
            df = df.copy()
            # Convert 'Begin File' to string
            df['Begin File'] = df['Begin File'].astype(str)
            df['filename'] = df['Begin File']
            
            # Calculate timings
            df['call_length'] = df['End Time (s)'] - df['Begin Time (s)']
            df['begin_time'] = df['File Offset (s)']
            df['end_time'] = df['begin_time'] + df['call_length']
            
            # Add label
            df['label'] = 1
            
            return df
        except KeyError as e:
            logger.warning(f"DataFrame missing required columns: {e}")
            return pd.DataFrame() # Return empty on failure due to schema mismatch

    def run(self, input_dir: str, output_csv: str):
        """
        Main execution flow: load files, process them, match with S3, and save output.
        """
        all_dfs = self.get_local_annotation_files(input_dir)
        
        columns_to_keep = ['filename', 'call_length', 'begin_time', 'end_time', 'label', 'Species']
        all_edited_dfs = []

        logger.info("Processing files and checking S3...")
        for annotations_filename, df in tqdm(all_dfs.items(), desc="Processing"):
            processed_df = self.process_single_dataframe(df)
            
            if processed_df.empty:
                continue

            # Filter columns ensuring they exist
            existing_cols = [c for c in columns_to_keep if c in processed_df.columns]
            processed_df = processed_df[existing_cols]
            processed_df['annotations_filename'] = annotations_filename
            
            # Check S3 for each row
            # Note: This row-by-row S3 check is slow. 
            # Ideally we'd batch this or check existency once, but sticking to original logic structure for safety.
            s3_paths = []
            for _, row in processed_df.iterrows():
                fname = row.get('filename')
                if not isinstance(fname, str):
                    s3_paths.append('None')
                    continue
                
                s3_path = self.check_file_exists_in_s3(fname)
                s3_paths.append(s3_path if s3_path else 'None')
            
            processed_df['s3_path'] = s3_paths
            all_edited_dfs.append(processed_df)

        if all_edited_dfs:
            final_df = pd.concat(all_edited_dfs, ignore_index=True)
            logger.info(f"Saving combined DataFrame to {output_csv}")
            final_df.to_csv(output_csv, index=False)
            logger.info("Done.")
        else:
            logger.warning("No data processed.")

if __name__ == "__main__":
    # Example usage configuration
    BUCKET_NAME = 'deepvoice-user-uploads'
    S3_PREFIX = 'shayetudor@gmail.com/dropbox/cods/'
    INPUT_DIR = '/home/ubuntu/soundbay/datasets/shaye_txt'
    OUTPUT_FILE = 'shaye_annotations_added_new_extended.csv'

    processor = AnnotationProcessor(
        s3_bucket=BUCKET_NAME,
        s3_prefix=S3_PREFIX
    )
    
    processor.run(input_dir=INPUT_DIR, output_csv=OUTPUT_FILE)
