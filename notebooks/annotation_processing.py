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
        
        # Pre-fetch list of ALL files to optimize lookups
        self.file_cache = self._build_file_cache()

    def _build_file_cache(self) -> Dict[str, str]:
        """
        Scan the S3 prefix and build a cache mapping filename -> full_s3_uri.
        This allows for O(1) existence checks.
        
        Returns:
            Dictionary mapping filename (basename) to full S3 URI.
        """
        logger.info(f"Building file cache from s3://{self.bucket}/{self.s3_prefix}...")
        file_cache = {}
        processed_count = 0
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            # Note: We do NOT use Delimiter here so we get all objects recursively
            page_iterator = paginator.paginate(
                Bucket=self.bucket, 
                Prefix=self.s3_prefix
            )

            for page in page_iterator:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    # Skip if key matches the prefix exactly (folder placeholder)
                    if key == self.s3_prefix:
                        continue
                        
                    filename = os.path.basename(key)
                    # If filename allows duplicates in different folders, we currently keep the LAST one found
                    # or first one? Dictionary overwrite keeps the last one encountered in iteration.
                    # Original logic was ambiguous but "first match" implies we might want to be careful.
                    # However, assuming unique filenames across folders is safer or acceptable default.
                    
                    if filename not in file_cache:
                        file_cache[filename] = f"s3://{self.bucket}/{key}"
                    
                    processed_count += 1
            
            logger.info(f"Cache built. Found {processed_count} files, {len(file_cache)} unique filenames.")
            return file_cache
        except Exception as e:
            logger.error(f"Error building file cache: {e}")
            return {}

    def check_file_exists_in_s3(self, filename: str) -> Optional[str]:
        """
        Check if a file exists in the cache.

        Args:
            filename: The name of the file to check.

        Returns:
            The full S3 path if found, otherwise None.
        """
        return self.file_cache.get(filename)

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

    def process_single_dataframe(self, annotations_filename: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a raw annotation DataFrame to standardize columns and types.
        """
        try:
            df = df.copy()
            # Convert 'Begin File' to string
            # 'Begin File' or 'Begin.File' are valid column names
            # rename 'Begin.File' to 'Begin File'
            df.rename(columns={'Begin.File': 'Begin File'}, inplace=True)
            df['Begin File'] = df['Begin File'].astype(str)
            df['filename'] = df['Begin File']
            
            # Calculate timings
            df['call_length'] = df['End Time (s)'] - df['Begin Time (s)']
            # check if file offst column exist, if not, use begin time as 'Begin Time (s)'
            if 'File Offset (s)' not in df.columns:
                df['File Offset (s)'] = df['Begin Time (s)']
            df['begin_time'] = df['File Offset (s)']
            df['end_time'] = df['begin_time'] + df['call_length']
            
            # Add label
            if '_fp.txt' in annotations_filename.lower():
                df['label'] = 0
            else:
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
            processed_df = self.process_single_dataframe(annotations_filename, df)
            
            if processed_df.empty:
                continue

            # Filter columns ensuring they exist
            existing_cols = [c for c in columns_to_keep if c in processed_df.columns]
            processed_df = processed_df[existing_cols]
            processed_df['annotations_filename'] = annotations_filename
            
            # Check S3 for each row
            # Optimized: using dictionary lookup for O(1) speed
            s3_paths = []
            for _, row in processed_df.iterrows():
                fname = row.get('filename')
                if not isinstance(fname, str):
                    s3_paths.append('None')
                    continue
                
                # Fast lookup
                s3_uri = self.check_file_exists_in_s3(fname)
                s3_paths.append(s3_uri if s3_uri else 'None')
            
            processed_df['s3_path'] = s3_paths
            all_edited_dfs.append(processed_df)

        if all_edited_dfs:
            final_df = pd.concat(all_edited_dfs, ignore_index=True)
            logger.info(f"Saving combined DataFrame to {output_csv}")
            final_df.to_csv(output_csv, index=False)
            
            # --- Statistics Reporting ---
            logger.info("="*30)
            logger.info("PROCESSING STATISTICS")
            logger.info("="*30)
            
            # 1. Unique files count
            unique_files = final_df['filename'].nunique()
            logger.info(f"Total Unique Audio Files: {unique_files}")

            # 1.1 S3 Existence Count
            found_count = (final_df['s3_path'] != 'None').sum()
            total_count = len(final_df)
            logger.info(f"Files Found in S3: {found_count} / {total_count} ({found_count/total_count:.1%})")
            
            # 2. Files per S3 folder
            # Extract folder structure from s3_path. 
            # Format: s3://bucket/prefix/folder/filename -> we want 'folder'
            # We know s3_path is 's3://bucket/prefix/folder/filename' OR 'None'
            
            def extract_s3_folder(path_str):
                if path_str == 'None' or not isinstance(path_str, str):
                    return 'Not Found in S3'
                # Remove bucket and prefix to get relative folder
                # Our stored s3_path is full URI.
                # simpler: parse standard URI or string manipulation
                try:
                    # s3://bucket/ ... /folder/filename
                    # lets just split by slash and take the parent dir of the file
                    # precise extraction depends on if we want full path or relative
                    # let's just show the parent directory
                    return os.path.dirname(path_str)
                except:
                    return 'Error Parsing Path'

            s3_folders = final_df['s3_path'].apply(extract_s3_folder)
            logger.info("\n--- Distribution by S3 Folder ---")
            logger.info(s3_folders.value_counts().to_string())

            # 3. Stats by annotation source file
            logger.info("\n--- Distribution by Source Annotation File ---")
            logger.info(final_df['annotations_filename'].value_counts().to_string())
            
            # 4. Label distribution
            if 'label' in final_df.columns:
                 logger.info("\n--- Label Distribution ---")
                 logger.info(final_df['label'].value_counts().to_string())

            logger.info("="*30)
            logger.info("Done.")
        else:
            logger.warning("No data processed.")

if __name__ == "__main__":
    # Example usage configuration
    BUCKET_NAME = 'deepvoice-user-uploads'
    S3_PREFIX = 'shayetudor@gmail.com/dropbox/cods/'
    INPUT_DIR = '/home/ubuntu/soundbay/datasets/shaye_txt'
    OUTPUT_FILE = 'shaye_annotations_15.1.26_extended.csv'

    processor = AnnotationProcessor(
        s3_bucket=BUCKET_NAME,
        s3_prefix=S3_PREFIX
    )
    
    processor.run(input_dir=INPUT_DIR, output_csv=OUTPUT_FILE)
