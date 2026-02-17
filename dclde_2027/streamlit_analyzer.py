"""
Streamlit app for analyzing model results.

Features:
- Load checkpoint and run inference on validation samples
- View spectrograms with ground truth and predictions
- Listen to audio samples
- Filter by class and error type (FP/FN)
- Class-balanced sampling
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchaudio.transforms import Spectrogram, AmplitudeToDB
import io
from collections import defaultdict

from config import Config
from model import BioacousticDetector, BioacousticDetectorBEATS, load_state_dict_compat
from local_dataset import get_local_dataloaders, download_dataset
from callbacks import SpectrogramVisualizer, CLASS_COLORS


@st.cache_resource
def load_model(checkpoint_path, config):
    """Load model from checkpoint."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Build model
    if config.model_type == "beats":
        model = BioacousticDetectorBEATS(config).to(device)
    elif config.model_type == "perch":
        model = BioacousticDetector(config).to(device)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    
    # Load checkpoint (with compat for old single-classifier checkpoints)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict, strict = load_state_dict_compat(checkpoint['model_state_dict'], model)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    
    return model, device, checkpoint


@st.cache_resource
def load_dataset(config):
    """Load validation dataset."""
    download_dataset()
    train_loader, val_loader = get_local_dataloaders(config)
    return val_loader


def compute_sample_metrics(pred_frames, target_frames, threshold=0.5):
    """
    Compute TP, FP, FN per sample and per class.
    
    Returns:
        dict with 'tp', 'fp', 'fn' per class, and 'has_fp', 'has_fn' flags
    """
    pred_bool = (torch.sigmoid(pred_frames) > threshold).float()
    target_bool = (target_frames > 0.5).float()
    
    # Only count on frames with annotations
    frame_mask = target_frames.sum(dim=-1) > 0
    
    # Sum over batch and frame dimensions to get per-class counts
    # pred_bool shape: (batch, frames, classes)
    # After masking and operations, sum over dims 0 and 1 to get (classes,)
    tp = ((pred_bool * target_bool) * frame_mask.unsqueeze(-1)).sum(dim=(0, 1))
    fp = ((pred_bool * (1 - target_bool)) * frame_mask.unsqueeze(-1)).sum(dim=(0, 1))
    fn = (((1 - pred_bool) * target_bool) * frame_mask.unsqueeze(-1)).sum(dim=(0, 1))
    
    # Convert to 1D numpy array
    tp_np = tp.cpu().numpy().flatten()
    fp_np = fp.cpu().numpy().flatten()
    fn_np = fn.cpu().numpy().flatten()
    
    return {
        'tp': tp_np,
        'fp': fp_np,
        'fn': fn_np,
        'has_fp': fp_np.sum() > 0,
        'has_fn': fn_np.sum() > 0,
        'has_correct': tp_np.sum() > 0 and fp_np.sum() == 0 and fn_np.sum() == 0,
    }


def get_classes_present(target_frames):
    """Get list of class indices that have annotations."""
    class_sums = target_frames.sum(dim=0)
    return [i for i, count in enumerate(class_sums) if count > 0]


def run_inference(model, dataloader, device, num_samples=500, threshold=0.5):
    """
    Run inference on samples and collect results.
    
    Returns:
        List of dicts with sample data and metrics
    """
    all_samples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(all_samples) >= num_samples:
                break
                
            audio = batch['audio'].to(device)
            targets = batch['target_frames'].to(device)
            
            outputs = model(audio)
            
            # Move entire batch to CPU once (more efficient than per-sample transfers)
            outputs_cpu = outputs.cpu()
            targets_cpu = targets.cpu()
            audio_cpu = batch['audio']  # Already on CPU from dataloader
            
            # Pre-process window_start/window_end if they're tensors (convert once per batch)
            window_starts = batch['window_start']
            window_ends = batch['window_end']
            if isinstance(window_starts, (list, tuple)):
                window_starts = [float(w) if not isinstance(w, (int, float)) else w for w in window_starts]
            elif isinstance(window_starts, torch.Tensor):
                window_starts = window_starts.cpu().tolist()
            
            if isinstance(window_ends, (list, tuple)):
                window_ends = [float(w) if not isinstance(w, (int, float)) else w for w in window_ends]
            elif isinstance(window_ends, torch.Tensor):
                window_ends = window_ends.cpu().tolist()
            
            batch_size = audio.shape[0]
            for i in range(batch_size):
                if len(all_samples) >= num_samples:
                    break
                
                sample = {
                    'audio': audio_cpu[i],
                    'target_frames': targets_cpu[i],
                    'pred_frames': outputs_cpu[i],  # Keep logits for visualization
                    'window_start': window_starts[i],
                    'window_end': window_ends[i],
                    'gcs_url': batch['gcs_url'][i],
                    'annotations': batch['annotations'][i],
                }
                
                # Compute metrics (still on GPU for efficiency, then move result)
                metrics = compute_sample_metrics(
                    outputs[i:i+1], 
                    targets[i:i+1], 
                    threshold
                )
                sample.update(metrics)
                
                # Get classes present
                sample['classes_present'] = get_classes_present(sample['target_frames'])
                
                all_samples.append(sample)
    
    return all_samples


def filter_samples(samples, selected_classes, error_type, class_names):
    """Filter samples by class and error type."""
    filtered = []
    
    for sample in samples:
        # Filter by class
        if selected_classes:
            sample_classes = [class_names[i] for i in sample['classes_present']]
            if not any(cls in selected_classes for cls in sample_classes):
                continue
        
        # Filter by error type
        if error_type == "False Positives (FP)":
            if not sample['has_fp']:
                continue
        elif error_type == "False Negatives (FN)":
            if not sample['has_fn']:
                continue
        elif error_type == "Correct":
            if not sample['has_correct']:
                continue
        # "All" - no filtering
    
        filtered.append(sample)
    
    return filtered


def get_class_balanced_samples(samples, num_samples, class_names):
    """Get class-balanced sample selection."""
    # Group samples by classes present
    samples_by_class = defaultdict(list)
    for sample in samples:
        for class_idx in sample['classes_present']:
            samples_by_class[class_idx].append(sample)
    
    # Sample evenly from each class
    samples_per_class = max(1, num_samples // len(class_names))
    selected = []
    used_indices = set()
    
    for class_idx in range(len(class_names)):
        class_samples = samples_by_class.get(class_idx, [])
        for i, sample in enumerate(class_samples):
            if len(selected) >= num_samples:
                break
            sample_id = id(sample)
            if sample_id not in used_indices:
                selected.append(sample)
                used_indices.add(sample_id)
                if len([s for s in selected if class_idx in s['classes_present']]) >= samples_per_class:
                    break
    
    # Fill remaining slots with any samples
    for sample in samples:
        if len(selected) >= num_samples:
            break
        if id(sample) not in used_indices:
            selected.append(sample)
            used_indices.add(id(sample))
    
    return selected[:num_samples]


def audio_to_wav_bytes(audio_tensor, sample_rate):
    """Convert audio tensor to WAV bytes for Streamlit audio player."""
    try:
        import torchaudio
        import io
        
        # Ensure mono and correct shape
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
        elif audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Normalize to [-1, 1] range using full dynamic range
        # This ensures quiet samples are boosted to use the full volume range
        audio_np = audio_tensor.squeeze().numpy()
        max_abs = np.abs(audio_np).max()
        if max_abs > 1e-8:  # Avoid division by zero for silent audio
            audio_np = audio_np / max_abs
        # If max_abs is very small, audio is essentially silent, leave as is
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        torchaudio.save(
            buffer, 
            torch.from_numpy(audio_np).unsqueeze(0), 
            sample_rate, 
            format='wav'
        )
        buffer.seek(0)
        return buffer.read()
    except ImportError:
        # Fallback: return numpy array (Streamlit can handle this)
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.mean(dim=0)
        elif audio_tensor.dim() == 1:
            pass
        else:
            audio_tensor = audio_tensor.squeeze()
        
        audio_np = audio_tensor.numpy()
        # Normalize to [-1, 1] range using full dynamic range
        max_abs = np.abs(audio_np).max()
        if max_abs > 1e-8:  # Avoid division by zero for silent audio
            audio_np = audio_np / max_abs
        # If max_abs is very small, audio is essentially silent, leave as is
        return audio_np


def main():
    st.set_page_config(page_title="Model Results Analyzer", layout="wide")
    
    st.title("🎵 Model Results Analyzer")
    st.markdown("Analyze model predictions with spectrograms and audio playback")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint Path",
        value="/home/ubuntu/soundbay/dclde_2027/checkpoints/fwwvq6zy/best.pt"
    )
    
    threshold = st.sidebar.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)
    num_samples = st.sidebar.number_input("Number of Samples to Analyze", 50, 1000, 500, 50)
    
    # Load config
    config = Config()
    
    # Load model and dataset
    if st.sidebar.button("Load Model & Dataset", type="primary"):
        with st.spinner("Loading model and dataset..."):
            try:
                model, device, checkpoint = load_model(checkpoint_path, config)
                dataloader = load_dataset(config)
                
                st.session_state['model'] = model
                st.session_state['device'] = device
                st.session_state['dataloader'] = dataloader
                st.session_state['checkpoint'] = checkpoint
                st.session_state['config'] = config
                st.session_state['visualizer'] = SpectrogramVisualizer(config)
                
                st.sidebar.success("Model and dataset loaded!")
            except Exception as e:
                st.sidebar.error(f"Error loading: {str(e)}")
                st.exception(e)
    
    if 'model' not in st.session_state:
        st.info("👈 Please load the model and dataset from the sidebar")
        return
    
    model = st.session_state['model']
    device = st.session_state['device']
    dataloader = st.session_state['dataloader']
    config = st.session_state['config']
    visualizer = st.session_state['visualizer']
    class_names = list(config.label_map.keys())
    
    # Run inference
    if st.sidebar.button("Run Inference", type="primary"):
        with st.spinner(f"Running inference on {num_samples} samples..."):
            samples = run_inference(model, dataloader, device, num_samples, threshold)
            st.session_state['samples'] = samples
            st.sidebar.success(f"Analyzed {len(samples)} samples!")
    
    if 'samples' not in st.session_state:
        st.info("👈 Please run inference from the sidebar")
        return
    
    samples = st.session_state['samples']
    
    # Filters
    st.sidebar.header("Filters")
    
    selected_classes = st.sidebar.multiselect(
        "Filter by Class",
        options=class_names,
        default=[]
    )
    
    error_type = st.sidebar.selectbox(
        "Filter by Error Type",
        options=["All", "False Positives (FP)", "False Negatives (FN)", "Correct"],
        index=0
    )
    
    use_class_balanced = st.sidebar.checkbox("Class-Balanced Sampling", value=True)
    samples_per_page = st.sidebar.number_input("Samples per Page", 1, 20, 5, 1)
    
    # Filter samples
    filtered_samples = filter_samples(samples, selected_classes, error_type, class_names)
    
    # Class-balanced selection if enabled
    if use_class_balanced and len(filtered_samples) > samples_per_page:
        display_samples = get_class_balanced_samples(
            filtered_samples, 
            samples_per_page * 10,  # Get more to allow pagination
            class_names
        )
    else:
        display_samples = filtered_samples
    
    st.sidebar.info(f"Showing {len(display_samples)} of {len(filtered_samples)} filtered samples")
    
    # Pagination
    total_pages = (len(display_samples) + samples_per_page - 1) // samples_per_page
    if total_pages > 1:
        page = st.sidebar.number_input("Page", 1, total_pages, 1, 1)
        start_idx = (page - 1) * samples_per_page
        end_idx = start_idx + samples_per_page
        display_samples = display_samples[start_idx:end_idx]
        st.sidebar.caption(f"Page {page} of {total_pages}")
    
    # Display samples
    st.header(f"Results ({len(display_samples)} samples)")
    
    for idx, sample in enumerate(display_samples):
        with st.expander(
            f"Sample {idx + 1}: {Path(sample['gcs_url']).name} "
            f"({sample['window_start']:.2f}s - {sample['window_end']:.2f}s)",
            expanded=True
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Spectrogram
                fig = visualizer.plot_spectrogram_with_predictions(
                    waveform=sample['audio'],
                    target_frames=sample['target_frames'],
                    pred_frames=sample['pred_frames'],
                    window_start=sample['window_start'],
                    window_end=sample['window_end'],
                    gcs_url=sample['gcs_url'],
                )
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                # Audio player
                st.subheader("🎧 Audio")
                try:
                    audio_bytes = audio_to_wav_bytes(sample['audio'], config.target_sr)
                    if isinstance(audio_bytes, bytes):
                        st.audio(audio_bytes, format='audio/wav')
                    else:
                        # Fallback: numpy array
                        st.audio(audio_bytes, sample_rate=config.target_sr)
                except Exception as e:
                    st.error(f"Error loading audio: {str(e)}")
                
                # Metrics
                st.subheader("📊 Metrics")
                
                # Per-class metrics
                for class_idx, class_name in enumerate(class_names):
                    # Extract scalar values (should be 1D array now)
                    tp = float(sample['tp'][class_idx])
                    fp = float(sample['fp'][class_idx])
                    fn = float(sample['fn'][class_idx])
                    
                    if tp + fp + fn > 0:
                        precision = tp / (tp + fp + 1e-8)
                        recall = tp / (tp + fn + 1e-8)
                        f1 = 2 * precision * recall / (precision + recall + 1e-8)
                        
                        st.markdown(f"**{class_name}**")
                        st.caption(f"TP: {int(tp)}, FP: {int(fp)}, FN: {int(fn)}")
                        st.caption(f"P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}")
                
                # Error flags
                st.subheader("⚠️ Errors")
                if sample['has_fp']:
                    st.error("Has False Positives")
                if sample['has_fn']:
                    st.error("Has False Negatives")
                if sample['has_correct']:
                    st.success("All Correct!")
                
                # Classes present
                st.subheader("🏷️ Classes Present")
                present_classes = [class_names[i] for i in sample['classes_present']]
                if present_classes:
                    st.write(", ".join(present_classes))
                else:
                    st.write("None (background)")
        
        st.divider()


if __name__ == "__main__":
    main()

