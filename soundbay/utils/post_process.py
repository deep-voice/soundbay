import torch
import numpy as np
import librosa


def post_process_predictions(preds: torch.Tensor, label_type: str, th: float = 0.5) -> tuple:
    """
    Post-process the predictions to probabilities
    """
    if label_type == 'single_label':
        proba = torch.softmax(preds, 1).cpu().numpy()
        predicted = torch.max(preds, 1).indices.cpu().numpy()
    elif label_type == 'multi_label':
        proba = torch.special.expit(preds).cpu().numpy()
        predicted = (proba > th).astype(int)
    else:
        raise ValueError(f"Label type {label_type} is not allowed")
    return  proba, predicted

def merge_start_end_predictions(preds_list: list, crop_length: int,
                                overlap: float, threshold: float = 0.98,
                                min_time_sec: float=5.0) -> tuple:
    """
    Merge the predictions from multiple crops
    @param
    preds_list: List of predictions from different crops
    seq_length: Length of the original crops
    overlap: Overlap ratio between crops
    threshold: Threshold to consider a prediction valid
    """
    times = []
    start_time, end_time = 0.0, 0.0
    prev_val = False
    for i in range(len(preds_list)):
        curr = preds_list[i] >= threshold
        if curr and not prev_val:
            start_time = i * crop_length * (1 - overlap)
        if not curr and prev_val:
            end_time = (i-1) * crop_length * (1 - overlap)
            times.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'prob': preds_list[i-1]
                })
        prev_val = curr

    if prev_val:
        end_time = (len(preds_list) - 1) * crop_length * (1 - overlap)
        times.append({
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
            })
        
    filtered_preds = []
    for v in times:
        if v['duration'] >= min_time_sec:
            filtered_preds.append(v)
    return filtered_preds

def get_frequency_boundaries(segment, sr, percentile=95, min_freq_threshold=4, max_freq_threshold=None):
    spec = librosa.stft(segment, n_fft=512, hop_length=256)
    spec = np.abs(spec)

    # remove time frames with abnormally high broadband power
    time_power = np.sum(spec ** 2, axis=0)
    q1, q3 = np.percentile(time_power, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        upper_thresh = time_power.mean() + 3 * time_power.std()
    else:
        upper_thresh = q3 + 3 * iqr
    keep_mask = time_power <= upper_thresh
    print(f'Removed {np.sum(~keep_mask)} out of {len(keep_mask)} time frames due to high broadband power.')
    if not np.any(keep_mask):  # fallback: keep all if everything is flagged
        keep_mask = np.ones_like(time_power, dtype=bool)
    spec = spec[:, keep_mask]

    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    freqs_diff = freqs[1] - freqs[0]
    spec = spec[freqs >= min_freq_threshold, :]
    freqs = freqs[freqs >= min_freq_threshold]
    if max_freq_threshold is not None:
        spec = spec[freqs <= max_freq_threshold, :]
        freqs = freqs[freqs <= max_freq_threshold]

    mean_spec_db = np.mean(spec, axis=1)
    
    threshold_db = np.percentile(mean_spec_db, percentile)
    min_freq = None
    max_freq = None
    for f, db in zip(freqs, mean_spec_db):
        if db >= threshold_db:
            if min_freq is None:
                min_freq = f - freqs_diff / 2
            max_freq = f + freqs_diff / 2
    return min_freq, max_freq