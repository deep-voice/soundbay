import argparse
import numpy as np
import pandas as pd
import os


def get_recording_name_from_inference_file_name(inference_file_name):
    recording_name = inference_file_name.split('\\')[-1].split('.')[0].split('-')[-1]
    return recording_name


def load_inference_results_from_dir(inference_dir: str) -> pd.DataFrame:
    df_all_recordings_inference = pd.DataFrame(
        columns=['segment_id', 'recording', 'class0_prob', 'class1_prob', 'segment_start_sec', 'segment_end_sec'])

    for filename in os.listdir(inference_dir):
        df_one_recording_inference = create_inference_df_for_one_recording(filename, inference_dir)
        df_all_recordings_inference = pd.concat([df_all_recordings_inference, df_one_recording_inference],
                                                ignore_index=True)

    df_all_recordings_inference['segment_end_sec'] = df_all_recordings_inference['segment_start_sec'] + 1
    df_all_recordings_inference.insert(0, 'chunk_id', '')
    return df_all_recordings_inference


def create_inference_df_for_one_recording(filename, inference_dir):
    """
    Load inference file and create DataFrame for one recording.
    :param filename: str
    :param inference_dir: str
    :return: pd.DataFrame
    """
    recording_name = get_recording_name_from_inference_file_name(filename)
    inference_full_path = os.path.join(inference_dir, filename)
    df_one_recording_inference = pd.read_csv(inference_full_path)
    df_one_recording_inference.insert(0, 'recording', recording_name)
    df_one_recording_inference['segment_start_sec'] = df_one_recording_inference.index
    df_one_recording_inference['segment_id'] = df_one_recording_inference['recording'] + '_' + \
                                               df_one_recording_inference[
                                                   'segment_start_sec'].astype(
                                                   str)
    return df_one_recording_inference


def validate_chunk_end_sec(chunk_end_sec: int, chunk_start_sec: int, chunk_actual_size: int) -> int:
    """
    Make sure the chunk_end_sec is correct (that it fits the chunk_start_sec and the desired chunk length).
    Else, correct it.
    """
    if chunk_end_sec - chunk_start_sec == chunk_actual_size:
        return chunk_end_sec
    elif chunk_end_sec - chunk_start_sec > chunk_actual_size:
        chunk_end_sec = chunk_start_sec + chunk_actual_size
        return chunk_end_sec
    else:
        print('chunk size error!')
        exit()


def add_chunk_id_per_recording(df: pd.DataFrame, recording_id: str, chunk_len_in_seconds: int = 300) -> pd.DataFrame:
    """
    Gets a DataFrame of recording segments; for a specific recording id, assigns all segments to chunks of desired length (defined by chunk_len_in_seconds).
    :param df: Pandas DataFrame containing segments of at least one recording
    :param recording_id: original id for recording to be processed into chinks
    :param chunk_len_in_seconds: required number of seconds for each chunk
    :return: DataFrame of recording segments with the specified recording separated into chunks (by assigned chunk_id).
    """
    recording_len_in_secs = df[df.recording == recording_id].segment_end_sec.max()
    chunk_start_sec = compute_first_chunk_start_sec(df, recording_id)
    while chunk_start_sec < recording_len_in_secs:
        df = assign_chunk_id(chunk_len_in_seconds, chunk_start_sec, df, recording_id, recording_len_in_secs)
        chunk_start_sec += chunk_len_in_seconds
    return df


def compute_first_chunk_start_sec(df: pd.DataFrame, recording_id: str) -> int:
    earliest_recording_sec = df[df.recording == recording_id].segment_start_sec.min()
    chunk_start_sec = max(0, earliest_recording_sec)
    return chunk_start_sec


def assign_chunk_id(chunk_len_in_seconds: int, chunk_start_sec: int, df: pd.DataFrame, recording_id: str,
                    recording_len_in_secs: int) -> pd.DataFrame:
    """
    Compute and assign the next chunk id.
    Only one chunk id is applied per run.
    :return: Pandas DataFrame with the assigned chunk_id for the relevant segments.
    """
    chunk_bool_mask, chunk_end_sec = compute_chunk_boundaries(chunk_len_in_seconds, chunk_start_sec, df, recording_id,
                                                              recording_len_in_secs)
    chunk_id = f'recording_{recording_id}_sec_{chunk_start_sec}_to_{chunk_end_sec}'
    df.loc[chunk_bool_mask, 'chunk_id'] = chunk_id
    return df


def compute_chunk_boundaries(chunk_len_in_seconds: int, chunk_start_sec: int, df: pd.DataFrame, recording_id: str,
                             recording_len_in_secs: int):
    """
    Figure out which segments should be included in chunk.
    :return:
        chunk_bool_mask: a boolean mask of all segments (=rows) included in chunk; pd.Series
        chunk_end_sec: the last recording second included in the chunk; int
    """
    chunk_end_sec = min(chunk_start_sec + chunk_len_in_seconds, recording_len_in_secs)
    chunk_bool_mask = ((df.recording == recording_id) & (df.segment_start_sec >= chunk_start_sec) & (
            df.segment_end_sec <= chunk_end_sec))
    chunk_actual_size = df[chunk_bool_mask].shape[0]
    chunk_end_sec = validate_chunk_end_sec(chunk_end_sec, chunk_start_sec, chunk_actual_size)
    return chunk_bool_mask, chunk_end_sec


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


def format_ranked_chunks_for_output(ranked_chunks: pd.Series) -> pd.DataFrame:
    '''
    Reformat a Pandas series of ranked chunks into the dataframe that serves as final output.
    :param ranked_chunks: pandas series containing chunk_id (as index) and the chunk gain score (the higher score, the more valuable the chunk).
    :return: pandas DataFrame with chunk_id (as index), chunk score (see above), and ranking (starting at 1; the lower the ranking, the more valuable the chunk).
    '''
    df_ranked_chunks = pd.DataFrame(ranked_chunks)
    df_ranked_chunks.columns = ['chunk_score']
    df_ranked_chunks.insert(0, 'ranking', np.arange(df_ranked_chunks.shape[0]) + 1)
    return df_ranked_chunks


def get_ranked_segments_from_inference_dir(inference_dir: str, chunk_len_in_seconds: int,
                                           gain_threshold: float) -> pd.DataFrame:
    df = load_inference_results_from_dir(inference_dir)
    for recording_id in df.recording.unique():
        df = add_chunk_id_per_recording(df, recording_id, chunk_len_in_seconds)
    df = compute_potential_gain_per_segment(df)
    ranked_chunks = df.groupby('chunk_id').apply(
        lambda x: n_high_priority_segments(x, threshold=gain_threshold)).sort_values(
        ascending=False)
    df_ranked_chunks = format_ranked_chunks_for_output(ranked_chunks)
    return df_ranked_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get annotation rankings for recording chunks.')
    parser.add_argument('inference_dir', type=str, help='path to the folder containing inference files')
    parser.add_argument('output_dir', type=str, default='.', help='path where the output file should be saved')
    parser.add_argument('--chunk_len', dest='chunk_len_in_seconds', type=int, default='180',
                        help='required chunk length (in seconds)')
    parser.add_argument('--thresh', dest='gain_threshold', type=float, default='0.35',
                        help='desired gain threshold for considering a segment as valuable')
    args = parser.parse_args()

    ranked_chunks = get_ranked_segments_from_inference_dir(args.inference_dir, args.chunk_len_in_seconds,
                                                           args.gain_threshold)
    output_full_path = os.path.join(args.output_dir, 'ranked_chunks.csv')
    ranked_chunks.to_csv(output_full_path)
    print(f'Ranked chunks csv saved at <{output_full_path}>')
