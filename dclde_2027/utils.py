import torch
import pandas as pd


def split_long_detections(detections, max_duration_sec=1.75):
    """
    Split any detection longer than max_duration_sec into consecutive segments of at most max_duration_sec.

    Args:
        detections: list of dicts with 'Begin Time (s)', 'End Time (s)', 'Class'
        max_duration_sec: maximum duration per segment (default 1.75). None or 0 = no splitting.

    Returns:
        list of detection dicts (long ones split into multiple)
    """
    if not detections or not max_duration_sec or max_duration_sec <= 0:
        return detections
    out = []
    for d in detections:
        b, e, c = d['Begin Time (s)'], d['End Time (s)'], d['Class']
        dur = e - b
        if dur <= max_duration_sec:
            out.append(dict(d))
            continue
        t = b
        while t < e:
            end_seg = min(t + max_duration_sec, e)
            out.append({'Begin Time (s)': t, 'End Time (s)': end_seg, 'Class': c})
            t = end_seg
    return out


def filter_short_detections(detections, min_duration_by_class=None):
    """
    Remove detections shorter than the per-class minimum duration.

    Args:
        detections: list of dicts with 'Begin Time (s)', 'End Time (s)', 'Class'
        min_duration_by_class: dict class_name -> min_sec (e.g. {'KW': 0.4}). Missing class or 0 = no filter.

    Returns:
        list of detection dicts (short ones removed)
    """
    if not detections or not min_duration_by_class:
        return detections
    out = []
    for d in detections:
        dur = d['End Time (s)'] - d['Begin Time (s)']
        min_sec = min_duration_by_class.get(d['Class'], 0) or 0
        if min_sec <= 0 or dur >= min_sec:
            out.append(d)
    return out


def merge_adjacent_detections(detections, gap_sec=0.5):
    """
    Merge same-class detections that are close in time (e.g. split across 5s windows).
    If two detections of the same class are within gap_sec of each other, merge into one.

    Args:
        detections: list of dicts with 'Begin Time (s)', 'End Time (s)', 'Class'
        gap_sec: max gap (seconds) between end of one and start of next to merge (default 0.5)

    Returns:
        list of merged detection dicts
    """
    if not detections:
        return []
    sorted_d = sorted(detections, key=lambda x: (x['Class'], x['Begin Time (s)']))
    out = [dict(sorted_d[0])]
    for d in sorted_d[1:]:
        prev = out[-1]
        if d['Class'] == prev['Class'] and (d['Begin Time (s)'] - prev['End Time (s)']) <= gap_sec:
            prev['End Time (s)'] = max(prev['End Time (s)'], d['End Time (s)'])
        else:
            out.append(dict(d))
    return out


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


def raven_table_to_tsv(detections, file_path=None):
    """
    Return detections as a Raven-compatible tab-separated string (for download / in-memory use).

    Args:
        detections: list of detection dicts from frames_to_raven_table
        file_path: optional file path to add to each row (e.g. sound file name)

    Returns:
        str: TSV content
    """
    df = detections_to_dataframe(detections, file_path)
    return df.to_csv(sep='\t', index=False)


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

