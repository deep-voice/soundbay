#!/usr/bin/env python3
"""
Fix the samples.csv mapping by merging on sample_id instead of ann_idx.

The original export had a bug where ann_idx was the index in a filtered DataFrame,
but the merge used indices from the unfiltered DataFrame, causing annotations to be
paired with the wrong windows.

Usage:
    python fix_samples_csv.py
"""

import hashlib
import re
import pandas as pd
from pathlib import Path

from config import Config


LOCAL_DIR = Path("/opt/dlami/nvme/dclde_samples")

# Inlined from dataset.py to avoid torchaudio import issues
FOLDER_MAPPINGS = {
    'quin_can': 'qc',
    'straitofgeorgia': 'straitofgeorgia_globus-robertsbank',
}

def map_to_gcs(local_path):
    path = local_path.replace("\\", "/")
    match = re.search(r'DCLDE/(.*)', path, re.IGNORECASE)
    if not match:
        return None
    relative_path = match.group(1)
    parts = [p for p in relative_path.split("/") if p]
    normalized = []
    for p in parts[:-1]:
        p = p.lower().replace(" ", "-")
        p = FOLDER_MAPPINGS.get(p, p)
        normalized.append(p)
    directory = "/".join(normalized)
    filename = parts[-1]
    base_bucket = "gs://noaa-passive-bioacoustic/dclde/2027/dclde_2026_killer_whales"
    return f"{base_bucket}/{directory}/{filename}"


def sample_id(gcs_url: str, ann_start: float, ann_end: float) -> str:
    """Unique ID for each annotation (must match export_samples.py)."""
    return hashlib.md5(f"{gcs_url}:{ann_start:.6f}:{ann_end:.6f}".encode()).hexdigest()[:16]


def main():
    config = Config()
    
    # Load the original annotations
    print("Loading original annotations...")
    df_full = pd.read_csv(config.csv_path)
    df_full['gcs_url'] = df_full['FilePath'].apply(map_to_gcs)
    df_full = df_full[df_full['gcs_url'].notna()].reset_index(drop=True)
    
    # Generate sample_id for each annotation
    print("Generating sample IDs...")
    df_full['sample_id'] = df_full.apply(
        lambda r: sample_id(r['gcs_url'], r['FileBeginSec'], r['FileEndSec']), axis=1
    )
    
    # Load current samples.csv (has correct window_start/window_end per sample_id)
    print("Loading current samples.csv...")
    samples_df = pd.read_csv(LOCAL_DIR / "samples.csv")
    
    # Keep only the correct columns from samples_df (window info)
    window_info = samples_df[['sample_id', 'window_start', 'window_end', 'error']].copy()
    
    # Merge with correct annotation info using sample_id
    print("Merging on sample_id...")
    fixed_df = window_info.merge(df_full, on='sample_id')
    
    # Verify the fix
    print("\nVerifying fix...")
    fixed_df['ann_overlaps_window'] = (
        (fixed_df['FileBeginSec'] < fixed_df['window_end']) & 
        (fixed_df['FileEndSec'] > fixed_df['window_start'])
    )
    
    total = len(fixed_df)
    overlapping = fixed_df['ann_overlaps_window'].sum()
    print(f"Total samples: {total}")
    print(f"Annotations that overlap their window: {overlapping} ({100*overlapping/total:.1f}%)")
    print(f"Annotations that DO NOT overlap: {total - overlapping}")
    
    if overlapping / total < 0.95:
        print("\nWARNING: Less than 95% overlap - something may still be wrong!")
        return
    
    # Save fixed samples.csv
    fixed_df = fixed_df.drop(columns=['ann_overlaps_window'])
    backup_path = LOCAL_DIR / "samples.csv.bak"
    output_path = LOCAL_DIR / "samples.csv"
    
    print(f"\nBacking up original to {backup_path}")
    samples_df.to_csv(backup_path, index=False)
    
    print(f"Saving fixed samples.csv to {output_path}")
    fixed_df.to_csv(output_path, index=False)
    
    print("\nDone! The samples.csv has been fixed.")


if __name__ == "__main__":
    main()

