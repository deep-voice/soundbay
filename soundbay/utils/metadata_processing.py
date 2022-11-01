import os
from pathlib import Path
import pandas as pd
import numpy as np
import re
import soundfile as sf
from typing import List


# TODO add tests to the utils in this file

def load_n_adapt_raven_annotation_table_to_dv_dataset_requirements(annotation_file_path: str,
                                                                   filename_suffix: str = ".Table.1.selections.txt"
                                                                   ) -> pd.DataFrame:
    # todo: decide whether to add annotation treatment
    df_annotations = pd.read_csv(annotation_file_path, sep="\t")
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


# <<<<<<< feature/EDA_script
def raven_to_df_annotations(annotations_path: str,
                            recording_path: str,
                            positive_tag_names: list = ['w', 'sc']):
    """
    Takes annotation files (selection table) created in Raven and turns it to a compatible annotations csv.
    """
    # create dataframe
    annotations = Path(annotations_path)
    recording = Path(recording_path)
    # ignore irrelevent files
    filelist = list(annotations.glob('*selections.txt'))
    metadata = []
    for file in filelist:
        dfTemp = pd.read_csv(file, sep="\t")
        dfTemp['filename'] = re.search(
            "\.Table.1.selections", file.as_posix()).group()
        dfTemp['FirstCallTime'] = np.amin(dfTemp['Begin Time (s)'])
        dfTemp['LastCallTime'] = np.amax(dfTemp['End Time (s)'])
        metadata.append(dfTemp)

    metadata = pd.concat(metadata)
    metadata.rename(columns={'Begin Time (s)': 'begin_time', 'End Time (s)': 'end_time'}, inplace=True)
    print('Number of Labels:', metadata.shape[0])
    for tag in positive_tag_names:
        metadata['Annotation'] = metadata['Annotation'].replace(np.nan, tag, regex=True)
    # add recording length to dataframe
    wav_filelist = list(recording.glob('*.wav'))
    wav_filedict = {re.search("\.Table.1.selections", file.as_posix()).group(): {'path': file} for file in wav_filelist
                    if re.search("\.Table.1.selections", file.as_posix())}
    for key, value in wav_filedict.items():
        record_length = sf.info(value['path']).duration
        value.update({'length': record_length})
        wav_filedict[key] = value
    annotation_lengths = []
    for filename_txt in metadata['filename']:
        annotation_lengths.append(wav_filedict[filename_txt]['length'])
    metadata['TotalRecordLength'] = annotation_lengths
    # filter metadata to contain only desired call types
    filters = positive_tag_names
    metadata_calls = metadata[metadata.Annotation.isin(filters)]
    unique_files, idx = np.unique(metadata['filename'], return_index=True)
    # Find true length of call sequences ( get rid of over lapping-sequences)
    non_overlap_all = non_overlap_df(metadata)
    non_overlap_calls = non_overlap_df(metadata_calls)
    # label background segments
    bg_segments = []
    for file in (unique_files):
        file_df = non_overlap_all[non_overlap_all['filename'] == file]
        begin = np.array(file_df['begin_time'])
        end = np.array(file_df['end_time'])
        for item in end:
            next_beginning = begin[begin > item]
            if next_beginning.size == 0:
                break
            next_beginning = np.min(next_beginning)
            bg_segments.append([item, next_beginning, file])
    bg_segments = pd.DataFrame(bg_segments, columns=['begin_time', 'end_time', 'filename'])
    bg_segments['call_length'] = bg_segments['end_time'] - bg_segments['begin_time']
    bg_segments.sort_values(by=['call_length'])
    # add labels: 0 for background and 1 for call. TODO: modify if there are different call types
    bg_segments['label'] = np.zeros(bg_segments.shape[0], dtype=int)
    non_overlap_calls['label'] = np.ones(non_overlap_calls.shape[0], dtype=int)
    # combine to a csv
    combined_annotations = pd.concat([bg_segments, non_overlap_calls])
    return combined_annotations


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


def bg_from_non_overlap_calls(df: pd.DataFrame):
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
