import os
import pandas as pd
import numpy as np
from dclde_2026 import config


def map_filepath_to_gcs(row):
    try:
        csv_path = str(row['FilePath'])
        gcs_base = f"gs://{config.GCS_AUDIO_BUCKET_NAME}/dclde/2026/dclde_2026_killer_whales"
        
        if 'DFO_CRP' in csv_path:
            path_fragment = csv_path.split('DFO_CRP/')[-1]
            gcs_path = f"{gcs_base}/dfo_crp/{path_fragment}"
        elif 'UAF_NGOS' in csv_path:
            path_fragment = csv_path.split('UAF/')[-1]
            gcs_path = f"{gcs_base}/uaf_ngos/{path_fragment}"
        else:
            path_fragment = csv_path.split('Audio/')[-1]
            gcs_path = f"{gcs_base}/{row['Provider'].lower()}/audio/{path_fragment}"
            
        return gcs_path.replace('\\', '/')
    except Exception:
        return None


def linear_freq_to_normalized_y(freq_hz):
    max_hz = config.SAMPLE_RATE / 2.0
    clipped_freq = np.clip(freq_hz, config.MIN_FREQ_HZ, config.MAX_FREQ_HZ)
    # We flipped the spectrogram so 0Hz is at index H-1 (Bottom, y=1)
    # and MaxHz is at index 0 (Top, y=0).
    # So Normalized Y = 1.0 - (freq / max_hz)
    return 1.0 - (clipped_freq / max_hz)


def convert_labels_to_yolo(df_labels, chip_start_sec, chip_duration_sec, class_column=None):
    if class_column is None:
        class_column = config.ANNOTATION_CLASS_COLUMN
        
    boxes = []
    chip_end_sec = chip_start_sec + chip_duration_sec

    for _, row in df_labels.iterrows():
        box_start_sec, box_end_sec = row['FileBeginSec'], row['FileEndSec']
        box_low_freq, box_high_freq = row['LowFreqHz'], row['HighFreqHz']
        class_name = row[class_column]
        class_id = config.CLASS_TO_ID.get(class_name, config.CLASS_TO_ID['UndBio'])  # Default to UndBio if not found

        clip_start_sec = max(chip_start_sec, box_start_sec)
        clip_end_sec = min(chip_end_sec, box_end_sec)
        clip_low_freq = max(config.MIN_FREQ_HZ, box_low_freq)
        clip_high_freq = min(config.MAX_FREQ_HZ, box_high_freq)
        
        if box_high_freq > config.MAX_FREQ_HZ:
            clip_high_freq = config.MAX_FREQ_HZ
            if box_low_freq >= config.MAX_FREQ_HZ:
                continue

        if (clip_end_sec <= clip_start_sec) or (clip_high_freq <= clip_low_freq):
            continue

        box_start_x_norm = (clip_start_sec - chip_start_sec) / chip_duration_sec
        box_end_x_norm = (clip_end_sec - chip_start_sec) / chip_duration_sec
        box_start_y_norm = linear_freq_to_normalized_y(clip_low_freq)
        box_end_y_norm = linear_freq_to_normalized_y(clip_high_freq)
        
        x_center = (box_start_x_norm + box_end_x_norm) / 2.0
        y_center = (box_start_y_norm + box_end_y_norm) / 2.0
        width = box_end_x_norm - box_start_x_norm
        # Calculate height ensuring it's positive
        height = abs(box_end_y_norm - box_start_y_norm)

        boxes.append([class_id, np.clip(x_center, 0.0, 1.0), np.clip(y_center, 0.0, 1.0),
                     np.clip(width, 0.0, 1.0), np.clip(height, 0.0, 1.0)])

    return np.array(boxes)


def convert_predictions_to_raven(predictions, df_chips, df_ann, chip_start_sec_col='chip_start_sec',
                                 confidence_threshold=0.5, output_path=None, simplified=False):
    """
    Convert model predictions to Raven selection table format.
    
    Args:
        predictions: List of prediction tensors, each [num_boxes, 5 + num_classes]
                     Format: [x_center, y_center, width, height, obj_score, class_scores...]
        df_chips: DataFrame with chip information (gcs_path, chip_start_sec, etc.)
        df_ann: Full annotation DataFrame (for getting file metadata)
        chip_start_sec_col: Column name for chip start time
        confidence_threshold: Minimum objectness * class_probability to keep
        output_path: Path to save Raven format file (tab-separated)
        simplified: If True, ignore frequency info (min=0, max=Nyquist), only use time
    
    Returns:
        DataFrame in Raven format
    """
    raven_rows = []
    
    for chip_idx, pred in enumerate(predictions):
        if pred is None or len(pred) == 0:
            continue
            
        chip_info = df_chips.iloc[chip_idx]
        gcs_path = chip_info['gcs_path']
        chip_start_sec = chip_info[chip_start_sec_col]
        chip_duration_sec = config.WINDOW_SEC
        
        # Get file metadata
        file_anns = df_ann[df_ann['gcs_path'] == gcs_path]
        if len(file_anns) == 0:
            continue
        
        # Extract filename from gcs_path
        filename = gcs_path.split('/')[-1].replace('.wav', '')
        
        # Convert predictions from normalized coords to absolute time/freq
        max_freq_hz = config.SAMPLE_RATE / 2.0
        
        for box_idx, box in enumerate(pred):
            if len(box) < 5 + config.NUM_CLASSES:
                continue
                
            x_center, y_center, width, height = box[0], box[1], box[2], box[3]
            obj_score = box[4]
            class_scores = box[5:5+config.NUM_CLASSES]
            
            # Get predicted class
            class_id = int(np.argmax(class_scores))
            class_prob = float(class_scores[class_id])
            class_name = config.ID_TO_CLASS[class_id]
            
            # Filter by confidence
            confidence = obj_score * class_prob
            if confidence < confidence_threshold:
                continue
            
            # Convert normalized coords to absolute time/freq
            # x is time, y is frequency
            begin_time = chip_start_sec + (x_center - width/2) * chip_duration_sec
            end_time = chip_start_sec + (x_center + width/2) * chip_duration_sec
            
            if end_time <= begin_time:
                continue
            
            if simplified:
                # Simplified version: ignore frequency, use full range
                low_freq = 0.0
                high_freq = max_freq_hz
            else:
                # y_center is normalized frequency (0=min, 1=max)
                # Convert back to Hz
                high_freq = (1 - (y_center - height/2)) * max_freq_hz
                low_freq = (1 - (y_center + height/2)) * max_freq_hz
                
                # Clip to valid range
                low_freq = max(config.MIN_FREQ_HZ, min(low_freq, config.MAX_FREQ_HZ))
                high_freq = max(config.MIN_FREQ_HZ, min(high_freq, config.MAX_FREQ_HZ))
                
                if high_freq <= low_freq:
                    continue
            
            raven_rows.append({
                'Selection': len(raven_rows) + 1,
                'View': 'Spectrogram 1',
                'Channel': 1,
                'Begin Time (s)': begin_time,
                'End Time (s)': end_time,
                'Low Freq (Hz)': low_freq,
                'High Freq (Hz)': high_freq,
                'Delta Time (s)': end_time - begin_time,
                'Delta Freq (Hz)': high_freq - low_freq,
                'Annotation': class_name,
                'Confidence': confidence,
                'filename': filename
            })
    
    raven_df = pd.DataFrame(raven_rows)
    
    if output_path and len(raven_df) > 0:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        raven_df.to_csv(output_path, sep='\t', index=False)
        print(f"Saved {len(raven_df)} predictions to {output_path}")
    
    return raven_df
