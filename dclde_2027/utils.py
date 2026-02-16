import torch
import pandas as pd


def frames_to_raven_table(predictions, window_start, frames_per_sec, class_names, threshold=0.5):
    """
    Convert frame predictions to Raven-style selection table.
    
    Args:
        predictions: (num_frames, num_classes) tensor of probabilities
        window_start: start time of window in file (seconds)
        frames_per_sec: frame resolution
        class_names: list of class names indexed by class_idx
        threshold: detection threshold
    
    Returns:
        List of dicts with 'Begin Time (s)', 'End Time (s)', 'Class'
    """
    detections = []
    binary = (predictions > threshold).int()
    
    for class_idx in range(predictions.shape[1]):
        class_binary = binary[:, class_idx]
        # Find contiguous segments
        diff = torch.diff(class_binary, prepend=torch.tensor([0]), append=torch.tensor([0]))
        starts = (diff == 1).nonzero().squeeze(-1)
        ends = (diff == -1).nonzero().squeeze(-1)
        
        for s, e in zip(starts, ends):
            detections.append({
                'Begin Time (s)': window_start + s.item() / frames_per_sec,
                'End Time (s)': window_start + e.item() / frames_per_sec,
                'Class': class_names[class_idx]
            })
    
    return detections


def detections_to_dataframe(all_detections, file_path=None):
    """
    Convert list of detections to a Raven-compatible DataFrame.
    
    Args:
        all_detections: list of detection dicts from frames_to_raven_table
        file_path: optional file path to add to each row
    
    Returns:
        DataFrame with Raven selection table columns
    """
    if not all_detections:
        return pd.DataFrame(columns=['Selection', 'Begin Time (s)', 'End Time (s)', 'Class'])
    
    df = pd.DataFrame(all_detections)
    df = df.sort_values('Begin Time (s)').reset_index(drop=True)
    df.insert(0, 'Selection', range(1, len(df) + 1))
    
    if file_path:
        df['File'] = file_path
    
    return df


def save_raven_table(detections, output_path, file_path=None):
    """
    Save detections as a Raven-compatible selection table (.txt).
    
    Args:
        detections: list of detection dicts from frames_to_raven_table
        output_path: path to save the .txt file
        file_path: optional file path to add to each row
    """
    df = detections_to_dataframe(detections, file_path)
    df.to_csv(output_path, sep='\t', index=False)

