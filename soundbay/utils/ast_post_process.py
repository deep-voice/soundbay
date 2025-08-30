"""
Audio segmentation and clustering utilities for post-processing predictions.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import logging
from torch.utils.data import Dataset, DataLoader
import audiosample as ap


logger = logging.getLogger(__name__)


def load_audio_segment(file_path: str, start_time: float, end_time: float, target_sr: int = 16000) -> np.ndarray:
	"""Load audio segment using AudioSample."""
	audio = ap.AudioSample(file_path)
	segment = audio[start_time:end_time]
	if segment.sample_rate != target_sr:
		segment = segment.resample(target_sr)
	return segment.as_numpy()


class AudioSegmentDataset(Dataset):
	"""Dataset for a single audio file yielding overlapping segments."""
	
	def __init__(self, file_path: str, segment_duration: float = 1.0, overlap: float = 0.75, target_sr: int = 16000):
		audio = ap.AudioSample(file_path)
		self.file_path = file_path
		self.segment_duration = segment_duration
		self.target_sr = target_sr
		self.hop_duration = segment_duration * (1 - overlap)
		
		num_segments = int((audio.duration - segment_duration) / self.hop_duration) + 1
		self.segments = [
			(i * self.hop_duration, i * self.hop_duration + segment_duration, i)
			for i in range(max(0, num_segments))
		]

	def __len__(self):
		return len(self.segments)

	def __getitem__(self, idx):
		start_time, end_time, seg_idx = self.segments[idx]
		audio = load_audio_segment(self.file_path, start_time, end_time, self.target_sr)
		return {'audio': audio, 'start_time': start_time, 'end_time': end_time, 'seg_idx': seg_idx}


class AudioSegmenter:
	"""Audio segmentation utility for post-processing predictions."""
	
	def __init__(self, segment_duration: float = 1.0, overlap: float = 0.75, target_sr: int = 16000, batch_size: int = 32):
		self.segment_duration = segment_duration
		self.overlap = overlap
		self.target_sr = target_sr
		self.batch_size = batch_size
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.hop_duration = segment_duration * (1 - overlap)
		logger.info(f"Using device: {self.device}")
	
	def get_largest_cluster_id(self, cluster_labels): 
		return np.argmax(np.bincount(cluster_labels))

	def process_audio_file(self, file_path: str, model, feature_extractor, reference_predictions: Optional[pd.DataFrame] = None, file_length: Optional[float] = 840) -> List[Tuple[float, float]]:
		"""Process a single audio file and return call segments."""
		logger.info(f"Processing: {file_path}")
		
		# Extract embeddings
		dataset = AudioSegmentDataset(file_path, self.segment_duration, self.overlap, self.target_sr)
		dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
		
		embeddings, segment_times = [], []
		model.eval()
		with torch.no_grad():
			for batch in dataloader:
				inputs = feature_extractor(
					[a.numpy() if isinstance(a, torch.Tensor) else a for a in batch['audio']],
					sampling_rate=self.target_sr, return_tensors="pt", padding=True
				)
				outputs = model(**{k: v.to(self.device) for k, v in inputs.items()})
				embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu())
				segment_times.extend([(b['start_time'], b['end_time']) for b in batch.values()])
		
		if not embeddings:
			return []
		
		# Cluster and identify calls
		embeddings = torch.cat(embeddings, dim=0)
		cluster_labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(embeddings.numpy())
		
		# Determine which cluster is calls
		if reference_predictions is not None:
			filename = Path(file_path).stem
			file_preds = reference_predictions[reference_predictions['filename'] == filename]
			calls_duration = (file_preds['end_time']- file_preds['begin_time']).sum()
			noise_duration =  file_length - calls_duration
			if noise_duration >= calls_duration:
				call_cluster = 1 - self.get_largest_cluster_id(cluster_labels)
			else: 
				call_cluster = self.get_largest_cluster_id(cluster_labels)

		else:
			call_cluster =  1 - self.get_largest_cluster_id(cluster_labels)
		
		# Merge adjacent calls
		is_call = (cluster_labels == call_cluster)
		return self._merge_calls(is_call, segment_times)

	def _merge_calls(self, is_call: np.ndarray, segment_times: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
		"""Merge adjacent call segments."""
		segments = []
		in_segment = False
		for i, (is_call_seg, (start, end)) in enumerate(zip(is_call, segment_times)):
			if is_call_seg and not in_segment:
				in_segment = True
				seg_start = start
			elif not is_call_seg and in_segment:
				in_segment = False
				segments.append((seg_start, end))
		if in_segment:
			segments.append((seg_start, segment_times[-1][1]))
		return segments

	def process_multiple_files(self, audio_paths: List[str], model, feature_extractor, reference_predictions: Optional[pd.DataFrame] = None) -> Dict[str, List[Tuple[float, float]]]:
		"""Process multiple audio files."""
		results = {}
		for file_path in tqdm(audio_paths, desc="Processing files"):
			try:
				results[file_path] = self.process_audio_file(file_path, model, feature_extractor, reference_predictions)
			except Exception as e:
				logger.error(f"Failed to process {file_path}: {e}")
				results[file_path] = []
		return results


def write_raven_selections(segments: List[Tuple[float, float]], output_path: str, low_freq: int = 0, high_freq: int = 10000) -> None:
	"""Write segments to Raven selection format."""
	import csv
	with open(output_path, 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerow(["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"])
		for idx, (start, end) in enumerate(segments, 1):
			writer.writerow([idx, "", 1, round(start, 6), round(end, 6), low_freq, high_freq])
	logger.info(f"Wrote {len(segments)} selections to {output_path}")


def merge_predictions_with_ast_segments(ref_df: pd.DataFrame, ast_segments: Dict[str, List[Tuple[float, float]]], tolerance: float = 1e-6) -> pd.DataFrame:
	"""Merge predictions based on AST segments."""
	logger.info("Merging predictions based on AST segments")
	merged_data = []
	merged_indices = set()
	for ast_begin, ast_end in ast_segments:
		# Find overlapping reference segments and sort them
		overlapping = ref_df[
			(ref_df['begin_time'] < ast_end) & 
			(ref_df['end_time'] > ast_begin)
		].sort_values('begin_time')
		
		# Identify adjacent pairs to merge
		i = 0
		while i < len(overlapping) - 1:
			current = overlapping.iloc[i]
			next_seg = overlapping.iloc[i+1]
			
			# Check for strict adjacency with tolerance
			if abs(next_seg['begin_time'] - current['end_time']) <= tolerance:
				merged_data.append({
					'begin_time': current['begin_time'],
					'end_time': next_seg['end_time']
				})
				merged_indices.update([current.name, next_seg.name])
				i += 2  # Skip the next segment since it's now part of the merge
			else:
				i += 1
	
	merged_df = pd.DataFrame(merged_data)
	unmerged_df = ref_df[~ref_df.index.isin(merged_indices)]
	
	return pd.concat([unmerged_df, merged_df], ignore_index=True) \
		.sort_values('begin_time').reset_index(drop=True)


def load_ast_model(model_name: str, device: str) -> tuple:
    from transformers import ASTModel, AutoFeatureExtractor
    """
    Load AST model and feature extractor.
    
    Args:
        model_name: Name of the pre-trained model
        device: Device to load the model on
        
    Returns:
        Tuple of (model, feature_extractor)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading AST model: {model_name}")
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ASTModel.from_pretrained(model_name).to(device)
    model.eval()
    
    return model, feature_extractor


def load_reference_predictions(predictions_path: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Load reference predictions if provided.
    
    Args:
        predictions_path: Path to predictions CSV file
        
    Returns:
        DataFrame with reference predictions or None
    """
    if predictions_path is None or not Path(predictions_path).exists():
        return None
        
    logger = logging.getLogger(__name__)
    logger.info(f"Loading reference predictions from: {predictions_path}")
    
    try:
        predictions_df = pd.read_csv(predictions_path)
        required_cols = ['filename', 'begin_time', 'end_time']
        
        if not all(col in predictions_df.columns for col in required_cols):
            logger.warning(f"Reference predictions missing required columns: {required_cols}")
            return None
            
        return predictions_df
        
    except Exception as e:
        logger.error(f"Failed to load reference predictions: {e}")
        return None


def main(audio_file_paths: List[str], reference_file_path: str, output_dir: str = "postprocess_outputs", 
         model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593", device: str = None):
    """
    Main function to process multiple audio files and merge predictions with AST segments.
    
    Args:
        audio_file_paths: List of paths to .wav files
        reference_file_path: Path to reference CSV file with 'Begin File' column
        output_dir: Directory to save merged outputs
        model_name: AST model name to use
        device: Device to run model on (auto-detected if None)
    """
    import os
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_path}")
    
    # Load reference predictions
    logger.info(f"Loading reference predictions from: {reference_file_path}")
    try:
        ref_df = pd.read_csv(reference_file_path)
        if 'Begin File' not in ref_df.columns:
            raise ValueError("Reference file must contain 'Begin File' column")
        logger.info(f"Loaded {len(ref_df)} reference predictions")
    except Exception as e:
        logger.error(f"Failed to load reference file: {e}")
        return
    
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load AST model
    try:
        model, feature_extractor = load_ast_model(model_name, device)
        logger.info("AST model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load AST model: {e}")
        return
    
    # Initialize audio segmenter
    segmenter = AudioSegmenter()
    
    # Process each audio file
    for audio_path in audio_file_paths:
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_path}")
                continue
                
            logger.info(f"Processing: {audio_path.name}")
            
            # Get filename without extension for matching
            filename = audio_path.stem
            
            # Filter reference predictions for this file
            file_refs = ref_df[ref_df['Begin File'] == filename].copy()
            if file_refs.empty:
                logger.warning(f"No reference predictions found for file: {filename}")
                continue
            
            # Rename columns to match expected format
            file_refs = file_refs.rename(columns={
                'Begin File': 'filename',
                'Begin Time (s)': 'begin_time',
                'End Time (s)': 'end_time'
            })
            
            # Get audio duration for reference
            try:
                audio = ap.AudioSample(str(audio_path))
                file_length = audio.duration
            except Exception as e:
                logger.warning(f"Could not get duration for {filename}: {e}")
                file_length = 840  # Default fallback
            
            # Process audio with AST
            ast_segments = segmenter.process_audio_file(
                str(audio_path), 
                model, 
                feature_extractor, 
                file_refs, 
                file_length
            )
            
            if not ast_segments:
                logger.info(f"No AST segments found for {filename}")
                continue
            
            # Convert AST segments to expected format
            ast_segments_dict = {filename: ast_segments}
            
            # Merge predictions with AST segments
            merged_df = merge_predictions_with_ast_segments(file_refs, ast_segments_dict)
            
            # Save merged results
            output_file = output_path / f"{filename}_merged_predictions.csv"
            merged_df.to_csv(output_file, index=False)
            logger.info(f"Saved merged predictions to: {output_file}")
            
            # Also save AST segments in Raven format
            raven_file = output_path / f"{filename}_ast_segments.txt"
            write_raven_selections(ast_segments, str(raven_file))
            
        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {e}")
            continue
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio files with AST and merge predictions")
    parser.add_argument("audio_files", nargs="+", help="Paths to .wav files")
    parser.add_argument("--reference", "-r", required=True, help="Path to reference CSV file")
    parser.add_argument("--output-dir", "-o", default="postprocess_outputs", help="Output directory")
    parser.add_argument("--model", "-m", default="MIT/ast-finetuned-audioset-10-10-0.4593", help="AST model name")
    parser.add_argument("--device", "-d", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    main(args.audio_files, args.reference, args.output_dir, args.model, args.device)