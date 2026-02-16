#!/usr/bin/env python3
"""
Export all annotation windows to S3.

Usage:
    python export_samples.py [--dry-run] [--workers N]
"""

import argparse
import hashlib
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import gcsfs
import pandas as pd
import torch
from torchaudio.io import StreamReader
from tqdm import tqdm

from config import Config
from dataset import map_to_gcs


S3_BUCKET = "deepvoice-datasets"
S3_PREFIX = "nature_dclde_2026_samples"
WINDOW_SEC = 5.0
TARGET_SR = 32000


def sample_id(gcs_url: str, ann_start: float, ann_end: float) -> str:
    """Unique ID for each annotation."""
    return hashlib.md5(f"{gcs_url}:{ann_start:.6f}:{ann_end:.6f}".encode()).hexdigest()[:16]


def extract_window(fs, url: str, ann_start: float, ann_end: float) -> tuple[torch.Tensor, float, float]:
    """Extract a random 5s window containing the annotation."""
    with fs.open(url, 'rb') as f:
        streamer = StreamReader(f)
        info = streamer.get_src_stream_info(0)
        
        # Get file duration
        if info.num_frames > 0 and info.sample_rate > 0:
            file_duration = info.num_frames / info.sample_rate
        elif info.bit_rate > 0:
            file_duration = f.details['size'] * 8 / info.bit_rate
        else:
            file_duration = ann_end + WINDOW_SEC
        
        # Random window that contains the annotation
        min_start = max(0, ann_end - WINDOW_SEC)
        max_start = min(ann_start, file_duration - WINDOW_SEC)
        if min_start > max_start:
            window_start = max(0, file_duration - WINDOW_SEC)
        else:
            window_start = random.uniform(min_start, max_start)
        
        # Read audio
        streamer.add_basic_audio_stream(
            frames_per_chunk=int(WINDOW_SEC * TARGET_SR),
            sample_rate=TARGET_SR
        )
        streamer.seek(window_start)
        
        expected = int(WINDOW_SEC * TARGET_SR)
        for (chunk,) in streamer.stream():
            wav = chunk.T
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if wav.shape[1] < expected:
                wav = torch.nn.functional.pad(wav, (0, expected - wav.shape[1]))
            elif wav.shape[1] > expected:
                wav = wav[:, :expected]
            return wav.to(torch.float16), window_start, window_start + WINDOW_SEC
        
        return torch.zeros(1, expected, dtype=torch.float16), window_start, window_start + WINDOW_SEC


def process_row(args):
    """Process a single annotation row."""
    idx, row, fs = args
    sid = sample_id(row['gcs_url'], row['FileBeginSec'], row['FileEndSec'])
    
    for attempt in range(3):
        try:
            audio, w_start, w_end = extract_window(
                fs, row['gcs_url'], row['FileBeginSec'], row['FileEndSec']
            )
            return {
                'idx': idx,
                'sample_id': sid,
                'audio': audio,
                'window_start': w_start,
                'window_end': w_end,
                'gcs_url': row['gcs_url'],
                'error': None,
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {'idx': idx, 'sample_id': sid, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=100, help='Process in batches to limit memory')
    parser.add_argument('--resume', action='store_true', help='Resume from existing progress')
    args = parser.parse_args()
    
    # Load annotations
    config = Config()
    print("Loading annotations...")
    df = pd.read_csv(config.csv_path)
    df['gcs_url'] = df['FilePath'].apply(map_to_gcs)
    df = df[df['gcs_url'].notna()].reset_index(drop=True)
    
    print(f"Annotations: {len(df)}")
    print(f"Estimated size: {len(df) * WINDOW_SEC * TARGET_SR * 2 / 1e9:.1f} GB")
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would export to s3://{S3_BUCKET}/{S3_PREFIX}/")
        return
    
    # Initialize
    fs = gcsfs.GCSFileSystem(token='anon')
    s3 = boto3.client('s3')
    tmp_dir = Path('/tmp/dclde_export')
    tmp_dir.mkdir(exist_ok=True)
    
    # Resume support: check which samples already exist
    completed_ids = set()
    metadata_file = tmp_dir / 'metadata_progress.csv'
    if args.resume and metadata_file.exists():
        existing = pd.read_csv(metadata_file)
        completed_ids = set(existing['sample_id'].tolist())
        print(f"Resuming: found {len(completed_ids)} already processed")
    
    # Filter out already processed
    df['_sample_id'] = df.apply(
        lambda r: sample_id(r['gcs_url'], r['FileBeginSec'], r['FileEndSec']), axis=1
    )
    if completed_ids:
        df = df[~df['_sample_id'].isin(completed_ids)].reset_index(drop=True)
        print(f"Remaining: {len(df)} annotations to process")
    
    success, failed = 0, 0
    
    # Open metadata file for incremental writes
    write_header = not metadata_file.exists() or not args.resume
    meta_fp = open(metadata_file, 'a')
    if write_header:
        meta_fp.write('ann_idx,sample_id,window_start,window_end,error\n')
    
    # Process in batches to limit memory
    total_batches = (len(df) + args.batch_size - 1) // args.batch_size
    pbar = tqdm(total=len(df), desc="Exporting")
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_row, (i + batch_start, row, fs)): i 
                for i, row in batch_df.iterrows()
            }
            
            for future in as_completed(futures):
                result = future.result()
                
                if result.get('error'):
                    failed += 1
                    meta_fp.write(f"{result['idx']},{result['sample_id']},,," 
                                  f"\"{result['error']}\"\n")
                    meta_fp.flush()
                    pbar.update(1)
                    continue
                
                # Save and upload immediately, then discard audio from memory
                path = tmp_dir / f"{result['sample_id']}.pt"
                torch.save({
                    'audio': result['audio'],
                    'window_start': result['window_start'],
                    'window_end': result['window_end'],
                }, path)
                
                s3.upload_file(str(path), S3_BUCKET, f"{S3_PREFIX}/samples/{result['sample_id']}.pt")
                path.unlink()
                
                # Write metadata immediately (don't accumulate in memory)
                meta_fp.write(f"{result['idx']},{result['sample_id']},"
                              f"{result['window_start']},{result['window_end']},\n")
                meta_fp.flush()
                
                success += 1
                pbar.update(1)
                
                # Explicitly clear reference to audio tensor
                del result['audio']
                del result
    
    pbar.close()
    meta_fp.close()
    
    # Create final samples.csv with mapping from annotations to windows
    print("Creating final samples.csv...")
    samples_df = pd.read_csv(metadata_file)
    
    # Reload full original df for merge
    df_full = pd.read_csv(config.csv_path)
    df_full['gcs_url'] = df_full['FilePath'].apply(map_to_gcs)
    df_full = df_full[df_full['gcs_url'].notna()].reset_index(drop=True)
    
    # Generate sample_id for df_full so we can merge on it (not ann_idx which has indexing issues)
    df_full['sample_id'] = df_full.apply(
        lambda r: sample_id(r['gcs_url'], r['FileBeginSec'], r['FileEndSec']), axis=1
    )
    
    samples_df = samples_df.merge(df_full, on='sample_id')
    samples_df.to_csv('/tmp/samples.csv', index=False)
    s3.upload_file('/tmp/samples.csv', S3_BUCKET, f"{S3_PREFIX}/samples.csv")
    
    print(f"\nDone! Success: {success}, Failed: {failed}")
    print(f"Output: s3://{S3_BUCKET}/{S3_PREFIX}/")


if __name__ == "__main__":
    main()
