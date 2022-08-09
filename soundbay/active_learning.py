import argparse

import argparse as argparse
import pandas as pd
# import numpy as np
import os

# import re

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
pd.options.display.max_colwidth = None

# PARAMS
GAIN_THRESHOLD = 0.30
CHUNK_LEN_IN_SECONDS = 300
# INFERENCE_DIR = '/mnt/c/Mine/Code/DeepVoice//active_learning/inference_files/inference_by_partial_positives_model'
INFERENCE_DIR = 'C:\\Mine\\Code\\DeepVoice\\active_learning\\inference_files\\inference_by_partial_positives_model'
OUTPUT_DIR = 'C:\Mine\Code\DeepVoice\\active_learning\\ranked_segments'


def get_recording_name_from_inference_file_name(inference_file_name):
    recording_name = inference_file_name.split('\\')[-1].split('.')[0].split('-')[-1]
    return recording_name


def load_data_from_dir(inference_dir: str) -> pd.DataFrame:
    df = pd.DataFrame(
        columns=['segment_id', 'recording', 'class0_prob', 'class1_prob', 'segment_start_sec', 'segment_end_sec'])

    for filename in os.listdir(inference_dir):
        recording_name = get_recording_name_from_inference_file_name(filename)
        inference_full_path = os.path.join(inference_dir, filename)
        one_recording_df = pd.read_csv(inference_full_path)
        one_recording_df.insert(0, 'recording', recording_name)
        one_recording_df['segment_start_sec'] = one_recording_df.index
        one_recording_df['segment_id'] = one_recording_df['recording'] + '_' + one_recording_df[
            'segment_start_sec'].astype(
            str)
        df = pd.concat([df, one_recording_df], ignore_index=True)

    df['segment_end_sec'] = df['segment_start_sec'] + 1
    df.insert(0, 'chunk_id', '')
    return df


def validate_chunk_end_sec(chunk_end_sec, chunk_start_sec, chunk_actual_size):
    if chunk_end_sec - chunk_start_sec == chunk_actual_size:
        return chunk_end_sec
    elif chunk_end_sec - chunk_start_sec > chunk_actual_size:
        chunk_end_sec = chunk_start_sec + chunk_actual_size
        return chunk_end_sec
    else:
        print('chunk size error!')
        exit()


def add_chunk_id_per_recording(df: pd.DataFrame, recording_id: str, chunk_len_in_seconds: int = 300):
    recording_len_in_secs = df[df.recording == recording_id].segment_end_sec.max()
    earliest_recording_sec = df[df.recording == recording_id].segment_start_sec.min()
    chunk_start_sec = max(0, earliest_recording_sec)
    while chunk_start_sec < recording_len_in_secs:
        chunk_end_sec = min(chunk_start_sec + chunk_len_in_seconds, recording_len_in_secs)
        chunk_bool_mask = ((df.recording == recording_id) & (df.segment_start_sec >= chunk_start_sec) & (
                df.segment_end_sec <= chunk_end_sec))
        chunk_actual_size = df[chunk_bool_mask].shape[0]
        chunk_end_sec = validate_chunk_end_sec(chunk_end_sec, chunk_start_sec, chunk_actual_size)
        chunk_id = f'recording_{recording_id}_sec_{chunk_start_sec}_to_{chunk_end_sec}'
        df.loc[chunk_bool_mask, 'chunk_id'] = chunk_id
        chunk_start_sec += chunk_len_in_seconds
    return df


def compute_potential_gain_per_segment(df: pd.DataFrame) -> pd.DataFrame:
    df['max'] = df[['class0_prob', 'class1_prob']].max(axis=1)
    df['gain'] = 1 - df['max']
    df = df.drop('max', axis=1)
    return df


def n_high_priority_segments(df_chunk: pd.DataFrame, threshold) -> int:
    """
    returns the number of high priority segments (where gain > threshold) in given dataframe
    """
    return df_chunk[df_chunk['gain'] > threshold].shape[0]


def get_ranked_segments_from_inference_dir(inference_dir: str, chunk_len_in_seconds: int = 90,
                                           gain_threshold: float = 0.35) -> pd.Series:
    df = load_data_from_dir(INFERENCE_DIR)
    for recording_id in df.recording.unique():
        df = add_chunk_id_per_recording(df, recording_id, chunk_len_in_seconds)
    df = compute_potential_gain_per_segment(df)
    ranked_chunks = df.groupby('chunk_id').apply(
        lambda x: n_high_priority_segments(x, threshold=gain_threshold)).sort_values(
        ascending=False)
    return ranked_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('inference_dir', type=str, help='path to the folder containing inference files')
    parser.add_argument('output_dir', type=str, default='.', help='path where the output file should be saved')
    parser.add_argument('--chunk_len', dest='chunk_len_in_seconds', type=int, default='90',
                        help='required chunk length (in seconds)')
    parser.add_argument('--thresh', dest='gain_threshold', type=int, default='0.35',
                        help='desired gain threshold for considering a segment as valuable')
    args = parser.parse_args()
    ranked_chunks = get_ranked_segments_from_inference_dir(args.inference_dir, args.chunk_len_in_seconds,
                                                           args.gain_threshold)

    # save results
