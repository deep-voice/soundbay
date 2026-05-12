import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import librosa
import numpy as np
import datetime
import pandas as pd

from Code.Models import FlexibleTinyDetector, DeepSpectrogramDetector, GlobalDetectorLongerTime
from Code.data import MultiCallDetectionDataset
from Code.Metrics import MetricsCalculator2Detection
from Code.Config import Config, load_config
from Code.fannie_file_handeling import get_all_rec_of_month
def get_model(config_path: Path):
    config = load_config(config_path)
    model_config = config.model
    data_config = config.data
    n_classes = config.model.num_classes
    if model_config.model_name == "flexible_tiny_detector":
        model = FlexibleTinyDetector(max_boxes=data_config.max_overlap_labels, n_classes=n_classes)
    elif model_config.model_name == "tiny_detector":
        model = DeepSpectrogramDetector(max_boxes=data_config.max_overlap_labels, n_classes=n_classes)
    elif model_config.model_name == "global_detector_longer_time":
        model = GlobalDetectorLongerTime(max_boxes=data_config.max_overlap_labels, n_classes=n_classes, pooling_size=model_config.pooling_size)
    else:
        raise NotImplementedError(f"Model {model_config.model_name} not implemented yet.")
    return model

def get_model_and_config(model_path, config_path: Path, device='cuda'):
    model = get_model(config_path)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.eval()
    config = load_config(config_path)
    return model, config

def activate_predictions(preds):
    class_logits = preds[..., 4:-1]
    conf_logits = preds[..., -1]

    class_proba = torch.softmax(class_logits, dim=-1)
    conf_proba = torch.sigmoid(conf_logits)

    loc_preds = torch.sigmoid(preds[..., :4]) # x,y,w,h to [0,1] range
    return torch.cat([loc_preds, class_proba, conf_proba.unsqueeze(-1)], dim=-1)

def plot_spec_detection(spec, pred_boxes):
    plt.figure(figsize=(10, 5))
    plt.imshow(spec[0], aspect='auto', origin='lower')
    if spec.ndim > 2:
        spec_h, spec_w = spec.shape[-2], spec.shape[-1]
    else:
        spec_h, spec_w = spec.shape

    for box in pred_boxes:
        x_rel, y_rel, w_rel, h_rel = box[:4]
        x = x_rel * spec_w
        y = y_rel * spec_h
        w = w_rel * spec_w
        h = h_rel * spec_h
        rect = plt.Rectangle((x, y), w, h, edgecolor='k', facecolor='none', linewidth=2)
        plt.gca().add_patch(rect)
    plt.title("Predicted Boxes on Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.axis('off')
    plt.show()

def convert_detection_to_df(detections, turn_class_int=True, remove_segment_num=True):
    if not isinstance(detections, pd.DataFrame):
        df = pd.DataFrame(detections)
    else:
        df = detections.copy() # to avoid modifying the original dataframe if it's already a df
    if turn_class_int:
        df['class'] = df['class'].astype(int)
    if remove_segment_num:
        df = df.drop('segment_num', axis=1)
    return df

def get_boxes_for_each_segment(detections):
    segments_ids = [det['segment_num'] for det in detections]
    boxes = [[det['start'], det['freq_low'], det['end'], det['freq_high']] for det in detections ]
    return segments_ids, boxes

def calculate_iou_matrix(boxes):
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1) * (y2 - y1)

    iou_matrix = np.zeros((len(boxes), len(boxes)))

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            inter_x1 = max(x1[i], x1[j])
            inter_y1 = max(y1[i], y1[j])
            inter_x2 = min(x2[i], x2[j])
            inter_y2 = min(y2[i], y2[j])

            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                union_area = area[i] + area[j] - inter_area
                iou_matrix[i, j] = inter_area / union_area
                iou_matrix[j, i] = iou_matrix[i, j]

    return iou_matrix

def cluster_based_on_iou(segments_ids, boxes, iou_matrix, iou_threshold=0.3):
    clusters = []
    visited = set()

    for i in range(len(boxes)):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i + 1, len(boxes)):
            if segments_ids[i] != segments_ids[j] and iou_matrix[i, j] > iou_threshold:
                cluster.append(j)
                visited.add(j)
        clusters.append(cluster)

    return clusters

def merge_detection_based_on_clusters(detections_df: pd.DataFrame, 
                                      clusters: list[list[int]]):
    merged_detections = []
    for cluster in clusters:
        cluster_dets = detections_df.iloc[cluster]
        merged_det = {
            'start': cluster_dets['start'].min(),
            'end': cluster_dets['end'].max(),
            'freq_low': cluster_dets['freq_low'].min(), # maybe min would be better
            'freq_high': cluster_dets['freq_high'].max(), # maybe max would be better
            'conf': cluster_dets['conf'].max(), # take max confidence in the cluster
            'class': cluster_dets['class'].mode()[0], # take most common class in the cluster
        }
        merged_detections.append(merged_det)
    # convert to dataframe for better handling:
    merged_detections_df = pd.DataFrame(merged_detections)
    return merged_detections_df

def merge_consecutive_detections(detections: list[dict], iou_threshold: float = 0.3) -> pd.DataFrame:
    if not detections:
        return pd.DataFrame([])

    segments_ids, boxes = get_boxes_for_each_segment(detections)
    iou_matrix = calculate_iou_matrix(boxes)
    
    clusters = cluster_based_on_iou(segments_ids, boxes, iou_matrix, iou_threshold=iou_threshold)
    detections_df = convert_detection_to_df(detections, turn_class_int=True, remove_segment_num=False)
    
    merged_detections = merge_detection_based_on_clusters(detections_df, clusters)
    return merged_detections

def run_inference(model: torch.nn.Module, audio_path: Path, 
                  config: Config, device='cuda', conf_thresh=0.5, overlap=0.25, iou_threshold=0.3) -> pd.DataFrame:
    model.eval()
    
    # 1. Load the long audio
    # Using your config's data_sample_rate
    audio, sr = torchaudio.load(str(audio_path))
    # resample to wanted sample rate if needed
    if sr != config.data.wanted_sample_rate:
        audio = T.Resample(orig_freq=sr, new_freq=config.data.wanted_sample_rate)(audio)
        # audio = resampler(audio)
        sr = config.data.wanted_sample_rate
    
    # 2. Setup sliding window parameters
    seq_len_samples = int(config.data.seq_length * sr)
    hop_len_samples = int(seq_len_samples * overlap) # 50% overlap to catch sounds on edges
    
    all_detections = []

    # 3. Slide through the audio
    to_spectrogram = T.Spectrogram(n_fft=config.data.n_fft, hop_length=config.data.hop_length, power=2.0)
    to_db = T.AmplitudeToDB(stype="power", top_db=80)

    for i, start_sample in enumerate(range(0, audio.shape[1] - seq_len_samples + 1, hop_len_samples)):
        end_sample = start_sample + seq_len_samples
        chunk = audio[:, start_sample:end_sample]
        
        # 4. Preprocess to Spectrogram
        # Use the same logic as your Dataset (n_fft, hop_length)        
        spec = to_spectrogram(chunk)
        spec = to_db(spec)
        spec = (spec - spec.mean()) / (spec.std() + 1e-9) # normalize 

        input_tensor = spec.unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(input_tensor)
            preds = activate_predictions(preds.clone())
            
        # 5. Filter and Shift Timestamps
        chunk_offset_sec = start_sample / sr
        
        for box in preds[0]:
            conf = box[-1].item()
            if conf > conf_thresh: # Confidence Threshold
                
                max_class_idx = torch.argmax(box[4:-1]).item()
                if box[4 + max_class_idx] < conf_thresh: # Class confidence threshold (optional, can be tuned)
                    continue
                x_rel = box[0].item() * config.data.seq_length
                y_rel = box[1].item() * (config.data.wanted_sample_rate / 2)
                w_rel = box[2].item() * config.data.seq_length
                h_rel = box[3].item() * (config.data.wanted_sample_rate / 2)

                start_time = x_rel + chunk_offset_sec
                end_time = start_time + w_rel
                
                all_detections.append({
                    'start': start_time,
                    'end': end_time,
                    'conf': conf,
                    'class': torch.argmax(box[4:6]).item(),
                    'freq_low': y_rel,
                    'freq_high': y_rel + h_rel,
                    'segment_num': i
                })
            # else: # removed because bad for ant_blue
            #     break

    all_detections = merge_consecutive_detections(all_detections, iou_threshold=iou_threshold)

    # remove detections shorted than 5.0 seconds:
    if not all_detections.empty:
        all_detections['too_short'] = all_detections['end'] - all_detections['start'] < 5.0
        all_detections = all_detections[~all_detections['too_short']].drop('too_short', axis=1).reset_index(drop=True)
    
    return all_detections

import soundfile as sf
def run_inference_multiple_files(model: torch.nn.Module, 
                                 audio_paths: list[Path], 
                                 config: Config, 
                                 device='cuda', conf_thresh=0.5,
                                 iou_threshold=0.5) -> pd.DataFrame:
    results = []
    current_start_time = 0.0
    for audio_path in tqdm(audio_paths, desc="Running inference on multiple files"):
        detections = run_inference(model, audio_path, config, device=device, conf_thresh=conf_thresh, iou_threshold=iou_threshold)
        
        # Shift detections by the cumulative duration of previous files
        if not detections.empty:
            detections['start'] += current_start_time
            detections['end'] += current_start_time
            results.append(detections)

        # update current_start_time for the next file
        audio_duration = sf.info(str(audio_path)).duration
        current_start_time += audio_duration
        
    return pd.concat(results, ignore_index=True)

def visualize_long_detections(audio_path: Path, detections: list[dict], 
                              config: Config, start_time=None, 
                              end_time=None, save_path=None):
    """
    Visualizes the filtered detections on a full-length spectrogram.
    """
    # 1. Load full audio and create spectrogram
    audio, sr = librosa.load(str(audio_path), sr=config.data.data_sample_rate)
    if start_time is not None and end_time is not None:
        audio = audio[start_time*sr:end_time*sr] # trim to the relevant section for faster processing
    # resanple to wanted sample rate if needed
    if config.data.data_sample_rate != config.data.wanted_sample_rate:
        audio = librosa.resample(audio, orig_sr=config.data.data_sample_rate, target_sr=config.data.wanted_sample_rate)
    

    # 2. Setup Plot
    plt.figure(figsize=(30, 10))
    plt.specgram(
        audio,
        NFFT=config.data.n_fft,
        Fs=config.data.wanted_sample_rate,
        noverlap=config.data.n_fft - config.data.hop_length,
        cmap='magma'
    )

    def format_time(x, pos):
        return str(datetime.timedelta(seconds=int(x)))
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(format_time))

    # 3. Draw Detections
    # all_detections looks like: {'start': sec, 'end': sec, 'conf': 0.9, 'class': 1, ...}
    for det in detections:
        start_t = det['start']
        end_t = det['end']
        duration = end_t - start_t
        
        # Frequency mapping: Model predicts 0-1 relative to spectrogram height
        # In librosa linear display, we map 0-1 to 0-Nyquist (SR/2)
        freq_min = det['freq_low'] 
        freq_max = det['freq_high']
        freq_height = freq_max - freq_min
        
        # Color coding by class
        color = 'cyan' if det['class'] == 1 else 'lime'
        label = f"C{det['class']} ({det['conf']:.2f})"
        
        # Create Rectangle: (x, y), width, height
        rect = plt.Rectangle(
            (start_t, freq_min), 
            duration, 
            freq_height, 
            fill=False, 
            edgecolor=color, 
            linewidth=2
        )
        ax.add_patch(rect)
        
        # Label the box
        plt.text(
            start_t, 
            freq_max + 2, 
            label, 
            color=color, 
            fontsize=10, 
            fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1)
        )

    plt.title(f"Detections for {audio_path.name}")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def convert_detections_to_raven_csv(detections, ouput_path, class_names: list[str] = None):
    """
    Converts detections to Raven Pro compatible CSV format.
    Raven expects columns like: Begin Time, End Time, Low Freq, High Freq, Label
    """
    # convert detections to pandas dataframe:
    # df = pd.DataFrame(detections)
    if isinstance(detections, list):
        df = convert_detection_to_df(detections, turn_class_int=True, remove_segment_num=False)
    elif isinstance(detections, pd.DataFrame):
        df = detections
    else:
        raise ValueError(f"Unsupported detections format: {type(detections)}")

    # change column names to raven format:
    df = df.rename(columns={
        'start': 'Begin Time (s)',
        'end': 'End Time (s)',
        'freq_low': 'Low Freq (Hz)',
        'freq_high': 'High Freq (Hz)',
        'class': 'Label',
        'conf': 'Confidence' # we can use the Annotation column to store confidence for filtering in Raven
    })

    if class_names is not None:
        df['Label'] = df['Label'].apply(lambda x: class_names[x])
    
    df['Delta Time (s)'] = df['End Time (s)'] - df['Begin Time (s)']
    df['Delta Freq (Hz)'] = df['High Freq (Hz)'] - df['Low Freq (Hz)']
    df['Label'] = df['Label'] #+ " | Conf: " + df['Confidence'].apply(lambda x: f"{x:.2f}")

    # add unit number for mad calls if Low Freq (Hz) > 27 unit 1 o.w. unit 2:
    df['Unit'] = df['Low Freq (Hz)'].apply(lambda x: 1 if x > 27 else 2)
    # if label is mad_blue, add "Unit: " to the label:
    df['Label'] = df.apply(lambda row: f"{row['Label']} | Unit: {row['Unit']}" if row['Label'] == "mad_blue" else row['Label'], axis=1)


    # add selection column as enumeration of rows:
    df['Selection'] = range(1, len(df) + 1)
    df['Channel'] = 1 # assuming mono audio, so channel 1
    df['View'] = 'Spectrogram 1' # default view name in Raven

    # reorder columns to match Raven's expected format:
    df = df[['Selection', 'Channel', 'View', 'Begin Time (s)', 'End Time (s)', 'Delta Time (s)', 
           'Low Freq (Hz)', 'High Freq (Hz)', 'Delta Freq (Hz)', 'Label']]
    df.to_csv(ouput_path, sep='\t', index=False)


if __name__ == "__main__":
    # Example usage:
    chechpoint_dir = Path("Checkpoints/2d_detector_15_sec_seq_length")
    # model = load_model(chechpoint_dir / "best.pth", chechpoint_dir / "config.yaml")
    model, config = get_model_and_config(chechpoint_dir / "last.pth", chechpoint_dir / "config.yaml")

    audio_path = Path("../soundbay/datasets/fannie_project/")

    april_2022_files = get_all_rec_of_month(audio_path, year=2022, month=4)
    nov_2021_files = get_all_rec_of_month(audio_path, year=2021, month=11)
    detections = run_inference_multiple_files(model, nov_2021_files, config, device='cuda', conf_thresh=0.95, iou_threshold=0.01)

    # detections = run_inference(model, example_audio, config, conf_thresh=0.75, overlap=0.25,
    #                            iou_threshold=0.05)
    detection_csv = Path("Results") / f"nov_2021_files_detections_15_sec_conf95.txt"
    convert_detections_to_raven_csv(detections, detection_csv, class_names=config.data.label_names)