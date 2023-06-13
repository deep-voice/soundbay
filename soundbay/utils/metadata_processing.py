import os
from pathlib import Path
import pandas as pd
import numpy as np
import re
import soundfile as sf
from typing import List


# TODO add tests to the utils in this file

def load_n_adapt_raven_annotation_table_to_dv_dataset_requirements(annotation_file_path: str,
                                                                   annotation_filename_dict: dict = None,
                                                                   filename_suffix: str = ".Table.1.selections.txt"
                                                                   ) -> pd.DataFrame:
    # todo: decide whether to add annotation treatment
    df_annotations = pd.read_csv(annotation_file_path, sep="\t")
    if annotation_filename_dict is not None:
        try:
            df_annotations['filename'] = annotation_filename_dict[os.path.basename(annotation_file_path)].replace('.txt', '')
        except KeyError:
            print(f"KeyError: {os.path.basename(annotation_file_path)}. Using default filename.")
            df_annotations['filename'] = os.path.basename(annotation_file_path).replace(filename_suffix, '')
    else:
        df_annotations['filename'] = os.path.basename(annotation_file_path).replace(filename_suffix, '')
    df_annotations = df_annotations.rename(columns={'Begin Time (s)': 'begin_time', 'End Time (s)': 'end_time'})
    df_annotations['call_length'] = df_annotations['end_time'] - df_annotations['begin_time']
    return df_annotations


def load_n_adapt_raven_annotation_dir_to_dv_dataset_requirements(annotation_dir_path: str,
                                                                 filename_suffix: str = '.Table.1.selections.txt',
                                                                 ) -> pd.DataFrame:
    df_list = []
    for filename in os.listdir(annotation_dir_path):
        annotation_file_path = os.path.join(annotation_dir_path, filename)
        small_df = load_n_adapt_raven_annotation_table_to_dv_dataset_requirements(annotation_file_path)
        df_list.append(small_df)
    df_all_annotations = pd.concat(df_list)
    return df_all_annotations


def annotations_df_to_csv(annotations_dataset, dataset_name: str = 'recordings_2018_filtered'):
    filename = 'combined_annotations_' + dataset_name + '.csv'
    annotations_dataset.to_csv(filename, index=False)


def merge_calls(sorted_df: pd.DataFrame) -> List[pd.Series]:
    """
    Args:
        sorted_df: DataFrame with sorted calls by begin_time field

    Returns: List of non-overlapping merged calls from the DataFrame
    """
    merged = [sorted_df.iloc[0].copy()]
    for _, higher in sorted_df.iterrows():
        lower = merged[-1]
        # test for intersection between lower and higher:
        # we know via sorting that lower[0] <= higher[0]
        if higher.begin_time <= lower.end_time:
            max_end_time = max(lower.end_time, higher.end_time)
            merged[-1].end_time = max_end_time  # replace by merged interval
        else:
            merged.append(higher.copy())
    return merged


def non_overlap_df(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        input_df: DataFrame with possibly overlapping calls

    Returns: a DataFrame object with non-overlapping calls (after merge).
    """
    non_overlap = []
    for file_name, file_df in input_df.groupby(by='filename'):
        file_df.sort_values(by='begin_time', inplace=True)
        merged = merge_calls(file_df)
        non_overlap.extend(merged)
    non_overlap = pd.DataFrame(non_overlap)
    non_overlap['call_length'] = non_overlap['end_time'] - non_overlap['begin_time']

    return non_overlap


def reorder_columns_to_default_view(df: pd.DataFrame):
    """
    Args:
        df: dataframe of the annotations metadata

    Returns: a dataframe with reordered column, so the default order view will be kept
    """

    def get_metadata_fields():
        return ['begin_time', 'end_time', 'filename', 'call_length', 'label']

    orig_cols = df.columns.tolist()
    default_cols = get_metadata_fields()
    remaining = list(set(orig_cols) - set(default_cols))
    new_cols = default_cols + remaining
    return df[new_cols]


def correct_call_times_with_duration(df: pd.DataFrame, audio_files_path: str):
    """
    Args:
        df: dataframe of the annotations metadata
        audio_files_path: str indicates the path to the folder of wav files (given flat hierarchy of audio files)

    Returns:
        df with 'end_time' no longer than the file duration
        it also removes the calls with 'begin_time' longer than duration and prints out a warning
    """

    audio_lengths = [sf.info(f'{audio_files_path}/{file}.wav').duration for file in df['filename']]
    df['audio_length'] = audio_lengths

    end_time_to_long_ind = df['end_time'] > df['audio_length']
    begin_time_to_long_ind = df['begin_time'] > df['audio_length']

    df.loc[end_time_to_long_ind, 'end_time'] = df.loc[end_time_to_long_ind, 'audio_length']
    df.loc[end_time_to_long_ind, 'call_length'] = \
        df.loc[end_time_to_long_ind, 'end_time'] - df.loc[end_time_to_long_ind, 'begin_time']

    if begin_time_to_long_ind.sum() > 0:
        df = df[~begin_time_to_long_ind]
        print(f'removed {begin_time_to_long_ind.sum()} files with begin_time > duration, verify annotations please!')

    return df.drop('audio_length', axis=1)


def extract_background_from_non_overlap_calls_df(df: pd.DataFrame):
    """
    Args:
        df: a dataframe of the annotations metadata, with calls that don't overlap

    Returns: a dataframe with bg calls taken from the gaps of the positive calls in a given file
    """
    bg_calls = []
    for _, df_per_file in df.groupby(by='filename'):
        # df_per_file is already sorted by begin_time!
        df_per_file_copy = df_per_file.iloc[1:].copy()
        if len(df_per_file_copy) < 1:
            continue
        bg_begin_times = np.array(df_per_file['end_time'])[:-1]
        bg_call_len = np.array(df_per_file['begin_time'])[1:] - bg_begin_times
        df_per_file_copy['begin_time'] = bg_begin_times
        df_per_file_copy['call_length'] = bg_call_len
        df_per_file_copy['end_time'] = df_per_file_copy['begin_time'] + df_per_file_copy['call_length']
        bg_calls.append(df_per_file_copy)
    bg_df = pd.concat(bg_calls)
    bg_df['label'] = 0
    return pd.concat([bg_df, df], ignore_index=True)


def merge_overlapping_calls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Receives an annotation dataframe with (possibly) overlapping calls, and goes through merge-and-drop iterations until
    no more overlaps are found.
    :param df: Pandas DataFrame with the following columns: ['filename', 'begin_time', 'end_time']
    :return: pd.DataFrame
    """
    df = df.sort_values(['filename', 'begin_time']).reset_index(drop=True)
    df = reset_overlap_accessory_columns(df)
    df = mark_overlapping_rows(df)

    while 1 in df.overlap.unique():
        df = merge_overlapping_rows(df)
        df = reset_overlap_accessory_columns(df)
        df = mark_overlapping_rows(df)

    df = df.drop(['overlap', 'next_begin_time', 'next_end_time'], axis=1)
    return df


def merge_overlapping_rows(df) -> pd.DataFrame:
    """
    Merge (and drop) overlapping rows.
    """
    df.loc[df.overlap == 1, 'end_time'] = df[df.overlap == 1]['next_end_time']
    df = df.drop_duplicates(subset=['filename', 'end_time'], keep='first')
    return df


def reset_overlap_accessory_columns(df) -> pd.DataFrame:
    df['overlap'] = np.NaN
    df['next_begin_time'] = df.groupby('filename').begin_time.shift(-1)
    df['next_end_time'] = df.groupby('filename').end_time.shift(-1)
    return df


def mark_overlapping_rows(df) -> pd.DataFrame:
    df.loc[df.next_begin_time < df.end_time, 'overlap'] = 1
    return df


if __name__ == "__main__":
    annotations_dir = '../../datasets/2021_annotations/'
    cols2drop = ['View', 'Channel', 'Low Freq (Hz)', 'High Freq (Hz)', 'Delta Time (s)', 'Delta Freq (Hz)',
                 'Avg Power Density (dB FS/Hz)']

    # treat annotation naming, duplicates, etc...
    annotation_filename_dict = {'210825-135601_Tr1.Table.1.selections.txt': '210825-135601_Tr1.wav',
                                '210825-135601_Tr1.Table.1.selections (1).txt': 'duplicate',
                                '210904-111316_Tr2.txt': '210904-111316_Tr2.wav',
                                '210904-111316_Tr2restofrecord.txt': 'corrupt',
                                '210904-111316_Tr2(first 4 minutes)(1).txt': 'duplicate',
                                '210904-111316_Tr2(first 4 minutes).txt': 'duplicate',
                                '210827-133618_Tr2.Table.1.selections.txt': '210827-133618_Tr2.wav',
                                '210904-093942_Tr2.Table.1.selections.txt': '210904-093942_Tr2.wav',
                                '210828-080644_Tr1.Table.1.selections.txt': '210828-080644_Tr1.wav',
                                '210827-081513_Tr1.Table.1.selections.txt': '210827-081513_Tr1.wav',
                                '210825-132034_Tr1.Table.1.selections.txt': '210825-132034_Tr1.wav',
                                '210824-104507_Tr1.Table.1.selections.txt': '210824-104507_Tr1.wav',
                                '210824-104507_Tr1.txt': '210824-104507_Tr1.wav',
                                '210825-112937_Tr1.txt': '210825-112937_Tr1.wav',
                                '210904-074321_Tr1.Table.1.selections.txt': '210904-074321_Tr1.wav',
                                '25-115438_Tr2.Table.1.selections.txt': 'unknown',
                                '210903-110841_Tr2.Table.1.selections.txt': '210903-110841_Tr2.wav',
                                '210825-102141_Tr1.txt': '210825-102141_Tr1.wav',
                                '210903-095104_Tr2.Table.1.selections.txt': '210903-095104_Tr2.wav',
                                '210903-095104_Tr1.Table.1.selections.txt': '210903-095104_Tr1.wav',
                                '210825-135601_Tr1.txt': '210825-135601_Tr1.wav',
                                '210824-125439_Tr1.txt': '210824-125439_Tr1.wav',
                                '180913_081527 (1).Table.1.selections.txt': 'unknown',
                                '210824-115331_Tr1.txt': '210824-115331_Tr1.wav',
                                '210825-112937_Tr2.txt': '210825-112937_Tr2.wav',
                                '210825-132034_Tr2.Table.1.selections.txt': '210825-132034_Tr2.wav',
                                '210827-133618_Tr1.Table.1.selections.donetxt.txt': '210827-133618_Tr1.wav',
                                '210826-083608_Tr1.Table.1.selections.txt': '210826-083608_Tr1.wav',
                                '210827-081513_Tr2.Table.1.selections.txt': '210827-081513_Tr2.wav',
                                '210824-100209_Tr1.txt': '210824-100209_Tr1.wav',
                                '210824-095226_Tr2.txt': '210824-095226_Tr2.wav',
                                '210827-090209_Tr1.Table.1.selections.txt': '210827-090209_Tr2.wav',
                                '210903-110841_Tr1.Table.1.selections.txt': '210903-110841_Tr1.wav',
                                '210824-100209_Tr2.txt': '210824-100209_Tr2.wav',
                                '210824-095226_Tr1.txt': '210824-095226_Tr1.wav',
                                '210904-093942_Tr1.Table.1.selections- Annotated.txt': '210904-093942_Tr1.wav',
                                '210903-095104_Tr1.Table.1.selections.txt': '210903-095104_Tr1.wav'
                                }

    df_list = []
    for filename in os.listdir(annotations_dir):
        try:
            #         print(filename)
            annotation_file_path = os.path.join(annotations_dir, filename)
            small_df = load_n_adapt_raven_annotation_table_to_dv_dataset_requirements(annotation_file_path,
                                                                                      annotation_filename_dict)
            df_list.append(small_df)
        except UnicodeDecodeError:
            print('exception:', 'UnicodeDecodeError')
            print('occured in file:', filename)
            continue
        except Exception as e:
            print('exception:', e)
            print('occured in file:', filename)
            continue

    print(len(df_list))
    df_all_annotations = pd.concat(df_list)
    df_all_annotations = df_all_annotations.drop(cols2drop, axis=1)
    # df_all_annotations.head()