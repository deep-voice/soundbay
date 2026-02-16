"""
Debug script for dataset and model verification.

1. Dataset debug: Save spectrograms with annotation overlays for each site
2. Model debug: Verify model forward pass works correctly
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from torchaudio.transforms import Spectrogram, AmplitudeToDB

from config import Config
from dataset import DCLDEMultiLabelDataset, map_to_gcs
from model import BioacousticDetector
import pandas as pd


# Colors for each class
CLASS_COLORS = {
    'KW': '#FF6B6B',   # Red
    'HW': '#4ECDC4',   # Teal
    'AB': '#FFE66D',   # Yellow
    'UndBio': '#95E1D3' # Mint
}


def create_spectrogram_transform(config):
    """Create mel spectrogram transform."""
    return torch.nn.Sequential(
       Spectrogram(
            # sample_rate=config.target_sr,
            n_fft=2048,
            hop_length=512,
            # n_mels=128,
            # f_min=0,
            # f_max=config.target_sr // 2
        ),
        AmplitudeToDB(top_db=80)
    )


def plot_spectrogram_with_annotations(
    waveform, 
    target_frames, 
    window_start, 
    window_end,
    gcs_url,
    annotations,
    config,
    spec_transform,
    save_path
):
    """
    Plot spectrogram with annotation overlays.
    
    Args:
        waveform: (channels, samples) audio tensor
        target_frames: (num_frames, num_classes) frame labels
        window_start: start time in file (seconds)
        window_end: end time in file (seconds)
        gcs_url: source file URL
        annotations: list of annotation dicts with time and frequency bounds
        config: Config object
        spec_transform: spectrogram transform
        save_path: path to save the figure
    """
    # Ensure mono
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Compute spectrogram
    spec = spec_transform(waveform)
    spec_db = spec.squeeze().numpy()
    
    # Create figure
    fig, (ax_spec, ax_labels) = plt.subplots(
        2, 1, figsize=(14, 8), 
        gridspec_kw={'height_ratios': [4, 1]},
        sharex=True
    )
    
    # Plot spectrogram
    time_axis = np.linspace(0, config.window_sec, spec_db.shape[1])
    freq_axis = np.linspace(50, config.target_sr // 2, spec_db.shape[0])
    
    im = ax_spec.pcolormesh(
        time_axis, freq_axis, spec_db, 
        shading='auto', cmap='magma'
    )
    ax_spec.set_ylabel('Frequency (Hz)', fontsize=11)
    ax_spec.set_ylim(50, config.target_sr // 2)  # Up to Nyquist (16kHz for 32kHz SR)
    
    # Extract filename from URL
    filename = gcs_url.split('/')[-1] if gcs_url else 'Unknown'
    
    ax_spec.set_title(
        f'{filename}\nWindow: {window_start:.2f}s - {window_end:.2f}s',
        fontsize=12, fontweight='bold'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_spec, pad=0.01)
    cbar.set_label('dB', fontsize=10)
    
    # Draw annotation bounding boxes (with frequency bounds from CSV)
    class_names = list(config.label_map.keys())
    num_frames = target_frames.shape[0]
    frame_times = np.linspace(0, config.window_sec, num_frames + 1)
    
    for ann in annotations:
        color = CLASS_COLORS.get(ann['class'], '#888888')
        
        # Draw rectangle showing time and frequency bounds
        rect = mpatches.Rectangle(
            (ann['rel_start'], ann['low_freq']),
            ann['rel_end'] - ann['rel_start'],
            ann['high_freq'] - ann['low_freq'],
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=0.3
        )
        ax_spec.add_patch(rect)
        
        # Add label with class and file time
        ax_spec.text(
            ann['rel_start'] + 0.05,
            ann['high_freq'] + 100,
            f"{ann['class']} ({ann['file_start']:.2f}s-{ann['file_end']:.2f}s)",
            fontsize=8,
            color=color,
            fontweight='bold',
            va='bottom'
        )
    
    # Plot label timeline
    for class_idx, class_name in enumerate(class_names):
        class_labels = target_frames[:, class_idx].numpy()
        color = CLASS_COLORS.get(class_name, '#888888')
        
        y_pos = len(class_names) - class_idx - 1
        for i, val in enumerate(class_labels):
            if val > 0.5:
                ax_labels.barh(
                    y_pos, 
                    frame_times[i+1] - frame_times[i], 
                    left=frame_times[i],
                    height=0.8, 
                    color=color, 
                    alpha=0.9
                )
    
    ax_labels.set_yticks(range(len(class_names)))
    ax_labels.set_yticklabels(class_names[::-1], fontsize=10)
    ax_labels.set_xlabel('Time (seconds)', fontsize=11)
    ax_labels.set_xlim(0, config.window_sec)
    ax_labels.set_ylim(-0.5, len(class_names) - 0.5)
    ax_labels.set_title('Frame Labels', fontsize=11)
    ax_labels.grid(axis='x', alpha=0.3)
    
    # Legend
    patches = [mpatches.Patch(color=CLASS_COLORS[c], label=c) for c in class_names]
    ax_spec.legend(handles=patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def debug_dataset(config, num_samples_per_class=3):
    """
    Debug dataset by saving spectrograms with annotations for each site.
    Samples from each class to ensure representation of all classes.
    """
    print("=" * 60)
    print("DATASET DEBUG")
    print("=" * 60)
    
    output_dir = Path('/home/ubuntu/soundbay/dclde_2027/debug_outputs/spectrograms')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    df = pd.read_csv(config.csv_path)
    df['gcs_url'] = df['FilePath'].apply(map_to_gcs)
    
    # Filter out rows with no valid gcs_url
    df = df[df['gcs_url'].notna()].reset_index(drop=True)
    
    spec_transform = create_spectrogram_transform(config)
    
    all_sites = list(config.train_sites) + list(config.val_sites) + list(config.test_sites)
    class_names = list(config.label_map.keys())
    
    for site in ['StraitofGeorgia']:
    # all_sites:
        print(f"\nProcessing site: {site}")
        site_df = df[df['Dataset'] == site].reset_index(drop=True)
        
        if len(site_df) == 0:
            print(f"  No data found for site {site}, skipping...")
            continue
        
        # Create site output directory
        site_dir = output_dir / site
        site_dir.mkdir(exist_ok=True)
        
        # Get classes present in this site
        site_classes = site_df['ClassSpecies'].unique()
        print(f"  Classes in site: {list(site_classes)}")
        
        saved_count = 0
        
        # Sample from each class
        for class_name in class_names:
            class_df = site_df[site_df['ClassSpecies'] == class_name].reset_index(drop=True)
            
            if len(class_df) == 0:
                print(f"  No samples for class {class_name}, skipping...")
                continue
            
            # Create dataset for this class subset
            dataset = DCLDEMultiLabelDataset(class_df, config)
            
            class_saved = 0
            max_attempts = min(len(dataset), num_samples_per_class * 5)
            indices = np.random.permutation(len(dataset))[:max_attempts]
            
            for idx in indices:
                if class_saved >= num_samples_per_class:
                    break
                
                try:
                    sample = dataset[idx]
                    target_frames = sample['target_frames']
                    
                    # Only save if there are actual annotations in this window
                    if target_frames.sum() == 0:
                        continue
                    
                    save_path = site_dir / f'{class_name}_{class_saved:02d}.png'
                    
                    plot_spectrogram_with_annotations(
                        waveform=sample['audio'],
                        target_frames=target_frames,
                        window_start=sample['window_start'],
                        window_end=sample['window_end'],
                        gcs_url=sample['gcs_url'],
                        annotations=sample['annotations'],
                        config=config,
                        spec_transform=spec_transform,
                        save_path=save_path
                    )
                    
                    class_saved += 1
                    saved_count += 1
                    print(f"  Saved {save_path.name}")
                    
                except Exception as e:
                    print(f"  Error processing {class_name} sample {idx}: {e}")
                    continue
            
            print(f"  Saved {class_saved}/{num_samples_per_class} spectrograms for class {class_name}")
        
        print(f"  Total saved for {site}: {saved_count} spectrograms")
    
    print(f"\nSpectrograms saved to: {output_dir}")


def debug_model(config):
    """
    Debug model by running a forward pass with dummy data.
    """
    print("\n" + "=" * 60)
    print("MODEL DEBUG")
    print("=" * 60)
    
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("\n1. Creating model...")
    try:
        model = BioacousticDetector(config)
        model = model.to(device)
        model.eval()
        print("   ✓ Model created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        return False
    
    # Test with dummy input
    print("\n2. Testing forward pass with dummy input...")
    batch_size = 2
    num_samples = int(config.window_sec * config.target_sr)
    
    dummy_input = torch.randn(batch_size, 1, num_samples).to(device)
    print(f"   Input shape: {dummy_input.shape}")
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   ✓ Forward pass successful")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected: (batch={batch_size}, frames=?, classes={config.num_classes})")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        return False
    
    # Check output values
    print("\n3. Checking output statistics...")
    print(f"   Output min: {output.min().item():.4f}")
    print(f"   Output max: {output.max().item():.4f}")
    print(f"   Output mean: {output.mean().item():.4f}")
    print(f"   Output std: {output.std().item():.4f}")
    
    # Test with real data sample if possible
    print("\n4. Testing with real data sample...")
    try:
        df = pd.read_csv(config.csv_path)
        df['gcs_url'] = df['FilePath'].apply(map_to_gcs)
        df = df[df['gcs_url'].notna()].reset_index(drop=True)
        
        if len(df) > 0:
            # Get a small subset
            sample_df = df.head(5)
            dataset = DCLDEMultiLabelDataset(sample_df, config)
            sample = dataset[0]
            
            audio = sample['audio'].unsqueeze(0).to(device)
            print(f"   Real audio shape: {audio.shape}")
            
            with torch.no_grad():
                output = model(audio)
            
            print(f"   ✓ Real data forward pass successful")
            print(f"   Output shape: {output.shape}")
            print(f"   Target shape: {sample['target_frames'].shape}")
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(output)
            print(f"   Prediction probabilities range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
        else:
            print("   ⚠ No data available for real sample test")
            
    except Exception as e:
        print(f"   ⚠ Could not test with real data: {e}")
    
    print("\n" + "=" * 60)
    print("MODEL DEBUG COMPLETE")
    print("=" * 60)
    return True


def main():
    """Run all debug functions."""
    config = Config()
    
    print("\n" + "=" * 60)
    print("DCLDE 2027 DEBUG SCRIPT")
    print("=" * 60)
    print(f"Config:")
    print(f"  - Sample rate: {config.target_sr}")
    print(f"  - Window: {config.window_sec}s")
    print(f"  - Frames/sec: {config.frames_per_sec}")
    print(f"  - Num classes: {config.num_classes}")
    print(f"  - Label map: {config.label_map}")
    
    # Debug dataset
    # debug_dataset(config, num_samples_per_class=3)
    
    # Debug model
    debug_model(config)


if __name__ == '__main__':
    main()

