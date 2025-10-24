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
	
	def __init__(self, segment_duration: float = 1.0, overlap: float = 0.75, target_sr: int = 16000, batch_size: int = 200):
		self.segment_duration = segment_duration
		self.overlap = overlap
		self.target_sr = target_sr
		self.batch_size = batch_size
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.hop_duration = segment_duration * (1 - overlap)
		logger.info(f"Using device: {self.device}")
		logger.info(f"Using half precision (float16) for model inference")
	
	def get_largest_cluster_id(self, cluster_labels): 
		return np.argmax(np.bincount(cluster_labels))

	def process_audio_file(self, file_path: str, model, feature_extractor, reference_predictions: Optional[pd.DataFrame] = None, file_length: Optional[float] = 840) -> List[Tuple[float, float]]:
		"""Process a single audio file and return call segments."""
		logger.info(f"Processing: {file_path}")
		
		# Extract embeddings
		dataset = AudioSegmentDataset(file_path, self.segment_duration, self.overlap, self.target_sr)
		dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
		
		embeddings, segment_times = [], []
		model.eval()
		
		# Add progress bar for processing segments
		total_batches = len(dataloader)
		with torch.no_grad():
			for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing segments", total=total_batches)):
				try:
					inputs = feature_extractor(
						[a.numpy() if isinstance(a, torch.Tensor) else a for a in batch['audio']],
						sampling_rate=self.target_sr, return_tensors="pt", padding=True
					)
					# Convert inputs to half precision to match model dtype
					inputs = {k: v.to(self.device, dtype=torch.float16) for k, v in inputs.items()}
					outputs = model(**inputs)
					
					# Extract 768-dimensional embeddings from ASTModel
					embedding_batch = outputs.last_hidden_state.mean(dim=1).cpu()
					if batch_idx == 0:  # Log embedding dimensions for first batch
						logger.info(f"Embedding dimensions: {embedding_batch.shape}")
					embeddings.append(embedding_batch)
					
					# Fix: properly access batch data structure
					batch_size = len(batch['audio'])
					for i in range(batch_size):
						start_time = batch['start_time'][i].item() if torch.is_tensor(batch['start_time'][i]) else batch['start_time'][i]
						end_time = batch['end_time'][i].item() if torch.is_tensor(batch['end_time'][i]) else batch['end_time'][i]
						segment_times.append((start_time, end_time))
				except RuntimeError as e:
					if "Input type" in str(e) and "bias type" in str(e):
						logger.error(f"Data type mismatch in batch {batch_idx}: {e}")
						logger.info("Attempting to convert model to full precision...")
						# Try to convert model to full precision as fallback
						model = model.float()
						inputs = {k: v.to(self.device, dtype=torch.float32) for k, v in inputs.items()}
						outputs = model(**inputs)
						
						# Extract 768-dimensional embeddings from ASTModel
						embedding_batch = outputs.last_hidden_state.mean(dim=1).cpu()
						embeddings.append(embedding_batch)
						
						# Continue with the rest of the batch
						batch_size = len(batch['audio'])
						for i in range(batch_size):
							start_time = batch['start_time'][i].item() if torch.is_tensor(batch['start_time'][i]) else batch['start_time'][i]
							end_time = batch['end_time'][i].item() if torch.is_tensor(batch['end_time'][i]) else batch['end_time'][i]
							segment_times.append((start_time, end_time))
					else:
						raise e
		
		if not embeddings:
			return []
		
		# Cluster and identify calls
		logger.info("Clustering embeddings...")
		embeddings = torch.cat(embeddings, dim=0)
		cluster_labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(embeddings.numpy())
		
		# Determine which cluster is calls
		if reference_predictions is not None:
			filename = Path(file_path).name
			file_preds = reference_predictions[reference_predictions['filename'] == filename]
			calls_duration = (file_preds['end_time']- file_preds['begin_time']).sum()
			noise_duration =  file_length - calls_duration
			if noise_duration >= calls_duration:
				call_cluster = 1 - self.get_largest_cluster_id(cluster_labels)
			else: 
				call_cluster = self.get_largest_cluster_id(cluster_labels)

		else:
			call_cluster =  1 - self.get_largest_cluster_id(cluster_labels)
		
		logger.info(f"Call cluster identified: {call_cluster}")
		
		# Merge adjacent calls
		logger.info("Merging adjacent call segments...")
		is_call = (cluster_labels == call_cluster)
		merged_segments = self._merge_calls(is_call, segment_times)
		logger.info(f"Found {len(merged_segments)} merged call segments")
		return merged_segments

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


def write_raven_selections(segments: List[Tuple[float, float]], output_path: str, low_freq: int = 0, high_freq: int = 48000) -> None:
	"""Write segments to Raven selection format."""
	import csv
	with open(output_path, 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerow(["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"])
		for idx, (start, end) in enumerate(segments, 1):
			writer.writerow([idx, "", 1, round(start, 6), round(end, 6), low_freq, high_freq])
	logger.info(f"Wrote {len(segments)} selections to {output_path}")


def write_raven_selections_from_df(df: pd.DataFrame, output_path: str, low_freq: int = 0, high_freq: int = 48000) -> None:
	"""Write DataFrame with begin_time and end_time columns to Raven selection format."""
	import csv
	
	# Ensure we have the required columns
	if 'begin_time' not in df.columns or 'end_time' not in df.columns:
		logger.error("DataFrame must contain 'begin_time' and 'end_time' columns")
		logger.error(f"Available columns: {list(df.columns)}")
		return
	
	# Clean the data - remove any rows with NaN or invalid values
	df_clean = df.dropna(subset=['begin_time', 'end_time'])
	
	# Sort by begin_time to ensure proper ordering
	df_sorted = df_clean.sort_values('begin_time').reset_index(drop=True)
	
	# Filter out invalid time ranges (end_time > begin_time)
	df_sorted = df_sorted[df_sorted['end_time'] > df_sorted['begin_time']]
	
	if df_sorted.empty:
		logger.warning("No valid time segments found after cleaning")
		return
	
	with open(output_path, 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerow(["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"])
		
		for idx, row in df_sorted.iterrows():
			begin_time = row['begin_time']
			end_time = row['end_time']
			
			# Ensure times are numeric
			try:
				begin_time = float(begin_time)
				end_time = float(end_time)
			except (ValueError, TypeError):
				logger.warning(f"Skipping row {idx}: invalid time values {begin_time}, {end_time}")
				continue
			
			writer.writerow([
				idx + 1,  # Selection number (1-based)
				"",        # View (empty)
				1,         # Channel
				round(begin_time, 6),  # Begin Time (s)
				round(end_time, 6),    # End Time (s)
				low_freq,              # Low Freq (Hz)
				high_freq              # High Freq (Hz)
			])
	
	logger.info(f"Wrote {len(df_sorted)} selections to {output_path}")


def merge_predictions_with_ast_segments(ref_df: pd.DataFrame, ast_segments: Dict[str, List[Tuple[float, float]]], tolerance: float = 1e-6) -> pd.DataFrame:
	"""Merge predictions based on AST segments."""
	logger.info("Merging predictions based on AST segments")
	merged_data = []
	merged_indices = set()
	
	# Handle the case where ast_segments is a dictionary with filename as key
	if isinstance(ast_segments, dict):
		# Extract the first (and should be only) list of segments
		segments_list = list(ast_segments.values())[0] if ast_segments else []
	else:
		# Handle case where ast_segments is directly a list
		segments_list = ast_segments if isinstance(ast_segments, list) else []
	
	logger.info(f"Processing {len(segments_list)} AST segments for merging")
	
	for ast_begin, ast_end in tqdm(segments_list, desc="Merging segments", total=len(segments_list)):
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
	
	# Combine and clean the data
	combined_df = pd.concat([unmerged_df, merged_df], ignore_index=True) \
		.sort_values('begin_time').reset_index(drop=True)
	
	# Keep only essential columns for Raven format
	essential_columns = ['begin_time', 'end_time']
	available_columns = [col for col in essential_columns if col in combined_df.columns]
	
	if len(available_columns) == 2:
		return combined_df[available_columns]
	else:
		logger.warning(f"Missing essential columns. Available: {list(combined_df.columns)}")
		return combined_df


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
    logger.info(f"Using ASTModel for 768-dimensional embeddings")
    logger.info(f"Using half precision (float16) for model and inputs")
    
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, dtype=torch.float16)
        
        # Always use ASTModel to get 768-dimensional embeddings
        model = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593", 
            attn_implementation="sdpa", dtype=torch.float16).to(device)
        
        model.eval()
        
        # Verify model is in half precision
        if next(model.parameters()).dtype != torch.float16:
            logger.warning("Model is not in half precision, converting...")
            model = model.half()
        
        logger.info(f"Model loaded successfully in {next(model.parameters()).dtype}")
        return model, feature_extractor
        
    except Exception as e:
        logger.error(f"Failed to load model in half precision: {e}")
        logger.info("Falling back to full precision")
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        # Always use ASTModel to get 768-dimensional embeddings
        model = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593", 
            attn_implementation="sdpa").to(device)
        
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


def main(audio_files_csv: str, reference_file_path: str, output_dir: str = "postprocess_outputs", 
         model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593", device: str = None, 
         save_csv: bool = False, use_half_precision: bool = True):
    """
    Main function to process multiple audio files and merge predictions with AST segments.
    
    Args:
        audio_files_csv: Path to CSV file containing audio file paths in 'files' column
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
        if use_half_precision:
            logger.info("Loading model with half precision for faster inference")
            model, feature_extractor = load_ast_model(model_name, device)
        else:
            logger.info("Loading model with full precision for maximum compatibility")
            # Override the load_ast_model function to use full precision
            from transformers import ASTForAudioClassification, AutoFeatureExtractor
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = ASTForAudioClassification.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593", 
                attn_implementation="sdpa").to(device)
            model.eval()
        
        logger.info("AST model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load AST model: {e}")
        return
    
    # Load audio file paths from CSV
    try:
        if not Path(audio_files_csv).exists():
            raise FileNotFoundError(f"Audio files CSV not found: {audio_files_csv}")
            
        audio_files_df = pd.read_csv(audio_files_csv)
        if 'files' not in audio_files_df.columns:
            available_cols = list(audio_files_df.columns)
            raise ValueError(f"CSV file must contain a 'files' column. Available columns: {available_cols}")
        
        # Clean the file paths and remove any empty rows
        audio_file_paths = audio_files_df['files'].dropna().astype(str).tolist()
        # audio_file_paths = [path.strip() for path in audio_file_paths if path.strip()]
        
        if not audio_file_paths:
            raise ValueError("No valid audio file paths found in CSV")
            
        logger.info(f"Loaded {len(audio_file_paths)} audio file paths from {audio_files_csv}")
        
        # Show first few paths for verification
        logger.info(f"First few audio files: {audio_file_paths[:3]}")
        
    except Exception as e:
        logger.error(f"Failed to load audio files CSV: {e}")
        return
    
    # Initialize audio segmenter
    segmenter = AudioSegmenter()
    
    # Process each audio file with progress bar
    for audio_path in tqdm(audio_file_paths, desc="Processing audio files", total=len(audio_file_paths)):
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_path}")
                continue
                
            logger.info(f"Processing: {audio_path.name}")
            
            # Get filename without extension for matching
            filename = audio_path.name
            
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
            
            # Save merged results in clean Raven format
            raven_output_file = output_path / f"{filename}_merged_predictions.txt"
            write_raven_selections_from_df(merged_df, str(raven_output_file))
            logger.info(f"Saved merged predictions in Raven format to: {raven_output_file}")
            
            # Optionally save CSV format for debugging
            if save_csv:
                csv_output_file = output_path / f"{filename}_merged_predictions.csv"
                merged_df.to_csv(csv_output_file, index=False)
                logger.info(f"Saved merged predictions CSV to: {csv_output_file}")
            
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
    parser.add_argument("--audio_files_csv", 
                        default='/home/ubuntu/files_to_process_07_09.csv', 
                        help="Path to CSV file containing audio file paths in 'files' column")
    parser.add_argument("--reference", "-r", 
                        default='/home/ubuntu/refrence_df_2021_2022.csv', 
                        help="Path to reference CSV file")
    parser.add_argument("--output-dir", "-o", default="/home/ubuntu/postprocess_outputs_2021_2022", help="Output directory")
    parser.add_argument("--model", "-m", default="MIT/ast-finetuned-audioset-10-10-0.4593", help="AST model name")
    parser.add_argument("--device", "-d", help="Device to use (cuda/cpu)")
    parser.add_argument("--save-csv", action="store_true", help="Also save CSV format for debugging")
    parser.add_argument("--no-half-precision", action="store_true", help="Disable half precision (use full precision instead)")
    
    args = parser.parse_args()
    
    main(args.audio_files_csv, args.reference, args.output_dir, args.model, args.device, 
         save_csv=args.save_csv, use_half_precision=not args.no_half_precision)
