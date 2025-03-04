import os
from collections import namedtuple
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
    #add recording length to dataframe
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

def get_overlap_pct(ref_start: float, ref_end: float, curr_start: float, curr_end: float) -> float:
    """
    Calculates the percentage of overlap between two time intervals.

    Ensures `ref_start` and `ref_end` represent the earlier interval.

    Parameters:
    - ref_start (float): Start time of the reference interval (earlier interval).
    - ref_end (float): End time of the reference interval.
    - curr_start (float): Start time of the current interval (may start later).
    - curr_end (float): End time of the current interval.

    Returns:
    - float: The overlap percentage, computed as:
      (overlap duration) / (shorter interval length).
      Returns a negative percentage if there is a gap between intervals.
    """
    # Ensure `ref_start` is earlier than `curr_start`
    assert ref_start <= ref_end, 'Reference interval should start before it ends.'
    assert curr_start <= curr_end, 'Current interval should start before it ends.'
    if curr_start < ref_start:
        ref_start, ref_end, curr_start, curr_end = curr_start, curr_end, ref_start, ref_end

    shorter_interval = min(ref_end - ref_start, curr_end - curr_start)

    # Compute overlap as negative if there's no overlap (ref_end < curr_start)
    overlap_duration = min(ref_end, curr_end) - curr_start

    # If the intervals don't overlap, the overlap_duration will be negative
    return overlap_duration / shorter_interval if shorter_interval != 0 else 0


def merge_calls(sorted_df: pd.DataFrame, overlap_pct_th: float = 0) -> List[pd.Series]:
    """
    Args:
        sorted_df: DataFrame with sorted calls by begin_time field
        overlap_pct_th: determines the min [%] overlap between two calls to merge them:
                        * if no overlap - do nothing
                        * if overlap [%] >= overlap_pct_th - merge two calls
                        * if overlap [%] < overlap_pct_th - split equally the overlapping part between cals

    Returns: List of non-overlapping merged calls from the DataFrame

    example:
    >>> df = pd.DataFrame({'begin_time': [5, 8, 10, 18], 'end_time': [6, 16, 20, 27]})
    >>> merge_calls(sorted_df=df, overlap_pct_th=0.5)
    output:
    [
        begin_time    end_time
        5             6          ,
        begin_time    end_time
        8             19         ,
        begin_time    end_time
        19            27
    ]
    """
    if 'call_type' in sorted_df.columns:
        assert sorted_df.call_type.nunique() == 1, 'The function is designed for a single call type.'
    if 'label' in sorted_df.columns:
        assert sorted_df.label.nunique() <= 2, 'The function is designed for a binary label.'

    merged = [sorted_df.iloc[0].copy()]
    for _, higher in sorted_df.iterrows():
        lower = merged[-1].copy()
        overlap_duration = lower.end_time - higher.begin_time
        # test for intersection between lower and higher:
        # we know via sorting that lower[0] <= higher[0]
        if overlap_duration >= 0:
            max_end_time = max(lower.end_time, higher.end_time)
            merged[-1].end_time = max_end_time  # replace by merged interval
            overlap_pct = get_overlap_pct(lower.begin_time, lower.end_time, higher.begin_time, higher.end_time)
            if overlap_pct < overlap_pct_th:
                merged = _split_calls_with_low_overlap(merged, higher, lower, overlap_duration)
        else:
            merged.append(higher.copy())
    return merged

def _split_calls_with_low_overlap(
        merged: List[pd.Series],
        higher: pd.Series,
        lower: pd.Series,
        overlap_duration: float) -> List[pd.Series]:
    """
    Split the overlapping duration equally between two calls.

    merged: list of non-overlapping calls.
    higher: the call that overlaps with the last call in merged.
    lower: the last call in merged.
    overlap_duration: the duration of the overlap between the last call in merged and higher.

    """
    merged[-1].end_time = lower.end_time - overlap_duration / 2
    higher.begin_time += overlap_duration / 2
    merged.append(higher.copy())
    return merged


def non_overlap_df(input_df: pd.DataFrame, overlap_pct_th: float = 0) -> pd.DataFrame:
    """
    Args:
        input_df: DataFrame with possibly overlapping calls

    Returns: a DataFrame object with non-overlapping calls (after merge).
    """
    non_overlap = []
    for file_name, file_df in input_df.groupby(by='filename'):
        file_df.sort_values(by='begin_time', inplace=True)
        merged = merge_calls(file_df, overlap_pct_th)
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
    # Sort by begin time to avoid negative call lengths and erroneously reverse times
    df = df.sort_values(by='begin_time', ascending=True)
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


def multi_target_from_time_intervals_df(
        df: pd.DataFrame,
        n_classes: int,
        overlap_threshold_pct: float = 0.0,
        noise_class_value: int = 0) -> pd.Series:
    """
    Args:
        df: a dataframe with the columns: 'begin_time', 'end_time', 'label'.
        n_classes: the number of classes in the multi-label target not including noise class.
        overlap_threshold_pct: the minimum overlap between two calls to be considered as a true overlap.
        noise_class_value: the value of the noise class, e.g. 0.

    Returns: a pd.Series of the multi-label target with the df original index.

    example:
        >>> start_times = np.random.uniform(0, 10, 3)
        >>> end_times = start_times + 1
        >>> labels = np.random.choice([1,2], 3)
        >>> df = pd.DataFrame({'begin_time': start_times, 'end_time': end_times, 'label': labels})
        >>> df
                begin_time	end_time	label
            0	4.051811	6.051811	2
            1	8.789995	9.789995	2
            2	5.861857	6.861857	1
    >>> multi_target_from_time_intervals_df(df, overlap_threshold_pct=0, noise_class_value=0)
        0    [1, 1]
        1    [0, 1]
        2    [1, 1]

    If df contains multiple files use it with groupby:
    >>> df.groupby('filename').apply(multi_target_from_time_intervals_df, n_classes=2, overlap_threshold_pct=0, noise_class_value=0)
    """
    assert 0 <= overlap_threshold_pct <= 1, 'overlap_threshold_pct should be in the range [0, 1]'
    assert pd.api.types.is_integer_dtype(df.label), 'label should be an integer type'
    assert n_classes > 0, 'n_classes should be greater than 0'

    Interval = namedtuple('Interval', ['start', 'end', 'label'])
    overlaps = {idx: [0] * n_classes for idx in df.index}
    reference_intervals = {}

    # Process intervals in chronological order
    for idx, row in df.query(f'label != {noise_class_value}').sort_values('begin_time').iterrows():
        interval = Interval(row.begin_time, row.end_time, row.label)

        # Mark the interval as overlapping with itself
        overlaps[idx][int(interval.label) - 1] = 1

        # Remove expired previous intervals (end time < current interval start time)
        reference_intervals = {idx: reference_interval for idx, reference_interval in reference_intervals.items()
                               if reference_interval.end >= interval.start}

        # Check overlaps with reference intervals (overlap >= min_overlap_threshold) and update overlaps
        for ref_idx, ref in reference_intervals.items():
            overlap_pct = get_overlap_pct(ref.start, ref.end, interval.start, interval.end)
            if overlap_pct >= overlap_threshold_pct:
                overlaps[idx][int(ref.label) - 1] = 1
                overlaps[ref_idx][int(interval.label) - 1] = 1

        reference_intervals[idx] = interval

    return pd.Series(overlaps, name='label')
