import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
from soundbay.utils.metadata_processing import load_n_adapt_raven_annotation_table_to_dv_dataset_requirements


class DatasetCreator:
    def __init__(self, annotations_dir, dataset_name, save_dir, cols_to_use=None, label_col_name='label',
                 desired_calls_label='w'):

        self.annotations_dir = annotations_dir
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.label_col_name = label_col_name
        self.desired_calls_label = desired_calls_label
        self.df_all_annotations = None
        self.df_clean_annotations = None
        self.df_no_overlap = None
        self.df_bg = None
        self.df_concat = None
        self.df_meta = None

        if cols_to_use is None:
            self.cols_to_use = ['channel', 'begin_time', 'end_time', 'filename', 'call_length', 'label']
        else:
            self.cols_to_use = cols_to_use

    def create_annotation_df(self, annotation_file_dict=None):
        if annotation_file_dict == {}:
            annotation_file_dict = None
        annotations_dir = self.annotations_dir
        filenames = os.listdir(annotations_dir)
        df_list = []
        print('FILENAMES:')
        for filename in sorted(filenames):
            print(filename)
            try:
                annotation_file_path = os.path.join(annotations_dir, filename)
                small_df = load_n_adapt_raven_annotation_table_to_dv_dataset_requirements(annotation_file_path, annotation_file_dict)
                df_list.append(small_df)
            except UnicodeDecodeError:
                print(f'\nUnicodeDecodeError: {filename}')
                continue

        print(f'\nlen(df_list): {len(df_list)}')
        df_all_annotations = pd.concat(df_list)

        if self.label_col_name != 'label':
            df_all_annotations.rename(columns={f'{self.label_col_name}': 'label'}, inplace=True)

        cols2drop = [col for col in df_all_annotations.columns if col not in self.cols_to_use]
        df_all_annotations = df_all_annotations.drop(cols2drop, axis=1)

        print(f'\n unique filenames: \n{df_all_annotations.filename.unique()}')

        self.df_all_annotations = df_all_annotations

    def extract_unique_labels(self):
        print(f'\nAnnotations that are used: \n{self.df_all_annotations.label.unique()}')
        print(f'\nAnnotations and their counts: \n{self.df_all_annotations.label.value_counts(dropna=False)}')

        # find labels that only appear once
        all_unique_labels = self.df_all_annotations.label.value_counts()[
            self.df_all_annotations.label.value_counts() == 1].index.values

        # build small df containing the above unique labels
        df_unique_labels = self.df_all_annotations[self.df_all_annotations.label.isin(all_unique_labels)].copy()

        # save the unique labels as a csv for potential analysis
        save_path = os.path.join(self.save_dir, 'unique_labels_info.csv')
        if not os.path.exists(save_path):
            # create path if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_unique_labels.to_csv(save_path, index=False)

    def standardize_labels(self, labels_that_are_calls=None, nan_is_call=False):
        target_label = self.desired_calls_label

        self.df_clean_annotations = self.df_all_annotations.copy()

        # Consolidate call labels if required
        if labels_that_are_calls is None:
            print("Since no labels to consolidate as calls were provided, we will skip this step.")
            return
        else:
            self.df_clean_annotations.replace(to_replace=labels_that_are_calls, value=target_label, inplace=True,
                                              limit=None, regex=False)

        # fill NaNs with 'sc' if nan_is_call is True
        # This is sometimes necessary, because some annotation teams (such as ours) do not necessarily give
        # labels to calls
        if nan_is_call:
            self.df_clean_annotations['label'] = self.df_clean_annotations.label.fillna(f'{target_label}')

        # remove all rows that are not the target label (e.g., if we have dolphin calls along with whale calls, but we
        # only want to train on whale calls, we remove all dolphin calls from the annotation file)
        self.df_clean_annotations = self.df_clean_annotations[self.df_clean_annotations.label == f'{target_label}']

    def merge_overlapping_calls(self):
        """
        Receives an annotation dataframe with (possibly) overlapping calls, and goes through merge-and-drop iterations
        until no more overlaps are found.
        self.df_clean_annotations must be a Pandas DataFrame with the following columns: ['filename', 'begin_time', 'end_time']
        """
        self.df_no_overlap = self.df_clean_annotations.sort_values(['filename', 'begin_time']).reset_index(drop=True)
        self.df_no_overlap = self.reset_overlap_accessory_columns(self.df_no_overlap)
        self.df_no_overlap = self.mark_overlapping_rows(self.df_no_overlap)

        while 1 in self.df_no_overlap.overlap.unique():
            self.df_no_overlap = self.merge_overlapping_rows(self.df_no_overlap)
            self.df_no_overlap = self.reset_overlap_accessory_columns(self.df_no_overlap)
            self.df_no_overlap = self.mark_overlapping_rows(self.df_no_overlap)

        self.df_no_overlap = self.df_no_overlap.drop(['overlap', 'next_begin_time', 'next_end_time'], axis=1)

    def merge_overlapping_rows(self, df) -> pd.DataFrame:
        """
        Merge (and drop) overlapping rows.
        """
        df.loc[df.overlap == 1, 'end_time'] = df[df.overlap == 1]['next_end_time']
        df = df.drop_duplicates(subset=['filename', 'end_time'], keep='first')
        return df

    def reset_overlap_accessory_columns(self, df) -> pd.DataFrame:
        df['overlap'] = np.NaN
        df['next_begin_time'] = df.groupby('filename').begin_time.shift(-1)
        df['next_end_time'] = df.groupby('filename').end_time.shift(-1)
        return df

    def mark_overlapping_rows(self, df) -> pd.DataFrame:
        df.loc[df.next_begin_time < df.end_time, 'overlap'] = 1
        return df

    def mark_background(self):
        self.df_bg = self.df_no_overlap[['filename', 'begin_time', 'end_time']]\
            .sort_values(['filename', 'begin_time']).reset_index(drop=True).copy()
        self.df_bg['next_begin_time'] = self.df_bg.groupby('filename').begin_time.shift(-1)
        self.df_bg = self.df_bg.rename({'end_time': 'bg_begin_time', 'next_begin_time': 'bg_end_time'}, axis=1)
        self.df_bg = self.df_bg[~self.df_bg.bg_end_time.isna()]
        self.df_bg = self.df_bg.drop('begin_time', axis=1)
        self.df_bg['label'] = 'bg'
        self.df_bg['call_length'] = self.df_bg['bg_end_time'] - self.df_bg['bg_begin_time']
        self.df_bg = self.df_bg.rename({'bg_begin_time': 'begin_time', 'bg_end_time': 'end_time'}, axis=1)

    def concat_bg_and_calls(self):
        self.df_concat = pd.concat([self.df_bg, self.df_no_overlap[['begin_time', 'end_time', 'filename',
                                                                    'call_length', 'label']]])
        print('Sanity check \n')
        print(f'label counts: \n{self.df_concat.label.value_counts()}')
        print(f'number of unique files: \n{self.df_concat.filename.nunique()}')
        print(f'difference between calls and bg: \n{self.df_concat.label.value_counts()[0] - self.df_concat.label.value_counts()[1]}')
        print('if the difference is equal to the unique number of files, it all makes sense :)')

        save_path = os.path.join(self.save_dir, f'{self.dataset_name}_prepped_with_bg.csv')

        self.df_concat.to_csv(f'{save_path}', index=False)

    def split_to_sets(self, train_val_test_split=(70, 20, 10)):
        self.df_meta = self.df_concat[self.df_concat.label == self.desired_calls_label].groupby('filename').agg(
            count_calls=pd.NamedAgg('begin_time', 'count'),
            sum_call_length=pd.NamedAgg('call_length', 'sum'),
            avg_call_length=pd.NamedAgg('call_length', 'mean'),
        ).sort_values('filename')
        self.df_meta['cum_call_length'] = self.df_meta.sum_call_length.cumsum()
        self.df_meta['cum_perc'] = 100 * self.df_meta['cum_call_length'] / (self.df_meta['sum_call_length'].sum())
        train_split, val_split, test_split = train_val_test_split
        train_files = self.df_meta.loc[self.df_meta['cum_perc'] < train_split].index.tolist()
        val_files = self.df_meta.loc[((self.df_meta['cum_perc'] > train_split) &
                                      (self.df_meta['cum_perc'] < train_split + val_split))].index.tolist()

        test_files = self.df_meta.loc[(self.df_meta['cum_perc'] > train_split + val_split)].index.tolist()

        call_length_total = self.df_meta.sum_call_length.sum()
        print("The percentage of splits between train, val and test is: \n")
        for files in [train_files, val_files, test_files]:
            print(np.round(self.df_meta.loc[files].sum_call_length.sum() / call_length_total * 100))

        df_train = self.df_concat[self.df_concat.filename.isin(train_files)]
        df_val = self.df_concat[self.df_concat.filename.isin(val_files)]
        df_test = self.df_concat[self.df_concat.filename.isin(test_files)]

        save_path_train = os.path.join(self.save_dir, f'{self.dataset_name}_prepped_with_bg_train.csv')
        save_path_val = os.path.join(self.save_dir, f'{self.dataset_name}_prepped_with_bg_val.csv')
        save_path_test = os.path.join(self.save_dir, f'{self.dataset_name}_prepped_with_bg_test.csv')

        df_train.to_csv(f'{save_path_train}', index=False)
        df_val.to_csv(f'{save_path_val}', index=False)
        df_test.to_csv(f'{save_path_test}', index=False)


if __name__ == "__main__":
    '''This script was used to prepare the brazil dataset. A similar script can be used to prepare other datasets, as
    can be seen below this one. The script below the brazil script was used to prepare that Mozambique2021 dataset.'''
    # You may need to run the script more than once, if you have multiple random labels that are all intended to be
    # calls. The script will consolidate all call labels into one label, if provided a list of labels to consolidate.
    # The extract_unique_labels() method will help with the above potential issue.

    # # #    These args are a MUST for the script to run     # # #
    '''
    annotations_dir = '../../../datasets/brazil/annotations_brazil3/annotations'     # path to annotations dir
    dataset_name = 'brazil_test'                                                     # name of dataset
    save_dir = '../../../datasets/test_folder'                                       # path to annotations save dir
    cols_to_use = ['begin_time', 'end_time', 'filename', 'call_length', 'label']     # columns to use for annotation
    label_col_name = 'Type'                                                          # original name of label column
    desired_calls_label = 'call'                                                     # how to label calls (default = 'w')
    train_val_test_split = (70, 20, 10)                                              # train, val, test split

    # These labels will be consolidated into one label, as a "call" label.
    # The labels will be as defined in desired_calls_label.

    labels_that_are_calls = ['Click', 'Whistle']
    # labels_that_are_calls = ['Click']
    # labels_that_are_calls = ['Whistle']

    # # #    These args are optional     # # #
    # If the annotation files are messy and un-coordinated with the audio files, you can define a custom dictionary
    # that will map the annotation file name to the audio file name. The dictionary should be in the following format:
    annotation_file_dict = {}

    # # #    The script starts here     # # #

    # The object will be created using the args provided above
    brazil_test_dataset = DatasetCreator(annotations_dir, dataset_name, save_dir, cols_to_use,
                                         label_col_name, desired_calls_label)

    # Remove the annotation_file_dict argument if you don't need it (or leave it as an empty dict)
    brazil_test_dataset.create_annotation_df(annotation_file_dict)

    # This method will extract the unique labels from the annotation file and save them to a csv file
    # for further exploration and analysis
    brazil_test_dataset.extract_unique_labels()

    # This method will standardize the labels in the annotation file so that we end up with a single label
    # for the call type that we are interested in. It will also replace the NaN values with the calls_label if the
    # nan_is_call argument is set to True
    brazil_test_dataset.standardize_labels(labels_that_are_calls=labels_that_are_calls, nan_is_call=False)

    # This method will merge overlapping calls in the annotation file
    brazil_test_dataset.merge_overlapping_calls()

    # This method will mark background between the annotated calls
    brazil_test_dataset.mark_background()

    # This method will merge the background and the calls into a single dataframe
    brazil_test_dataset.concat_bg_and_calls()

    # This method will split the dataset into train, validation and test sets
    # default split is 70% train, 20% validation and 10% test
    brazil_test_dataset.split_to_sets(train_val_test_split=train_val_test_split)

    '''
    annotations_dir = '../../../datasets/mozambique_2021/annotation_files/'         # path to annotations dir
    dataset_name = 'mozambique2021'                                                 # name of dataset
    save_dir = '../../../datasets/test_folder'                                      # path to annotations save dir
    cols_to_use = ['begin_time', 'end_time', 'filename', 'call_length', 'label']    # columns to use in for annotation
    label_col_name = 'Annotation'                                                   # original name of label column
    desired_calls_label = 'call'                                                     # how to label calls (default = 'w')
    train_val_test_split = (70, 20, 10)                                              # train, val, test split

    # # #    These args are optional     # # #

    # These labels will be consolidated into one label, as a "call" label. The labels will be as defined in calls_label.
    labels_that_are_calls = ['SC', 'sc ?', 'un- weird whale sound probably', 'cs ?', 'baby whale?', 'song (s)', 's', 'sc?']

    # If the annotation files are messy and un-coordinated with the audio files, you can define a custom dictionary
    # that will map the annotation file name to the audio file name. The dictionary should be in the following format:
    annotation_file_dict = {'210825-135601_Tr1.Table.1.selections.txt': '210825-135601_Tr1.wav',
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
                            '210825-112937_Tr1.txt' : '210825-112937_Tr1.wav',
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
                            '210824-095226_Tr1.txt' : '210824-095226_Tr1.wav',
                            '210904-093942_Tr1.Table.1.selections- Annotated.txt': '210904-093942_Tr1.wav',
                            '210903-095104_Tr1.Table.1.selections.txt': '210903-095104_Tr1.wav'
                            }

    # The object will be created using the args provided above
    mozambique2021_dataset = DatasetCreator(annotations_dir, dataset_name, save_dir, cols_to_use,
                                            label_col_name, desired_calls_label)

    # Remove the annotation_file_dict argument if you don't need it
    mozambique2021_dataset.create_annotation_df(annotation_file_dict)

    # This method will extract the unique labels from the annotation file and save them to a csv file
    # for further exploration and analysis
    mozambique2021_dataset.extract_unique_labels()

    # This method will standardize the labels in the annotation file so that we end up with a single label
    # for the call type that we are interested in. It will also replace the NaN values with the calls_label if the
    # nan_is_call argument is set to True
    mozambique2021_dataset.standardize_labels(labels_that_are_calls=labels_that_are_calls, nan_is_call=True)

    # This method will merge overlapping calls in the annotation file
    mozambique2021_dataset.merge_overlapping_calls()

    # This method will mark background between the annotated calls
    mozambique2021_dataset.mark_background()

    # This method will merge the background and the calls into a single dataframe
    mozambique2021_dataset.concat_bg_and_calls()

    # This method will split the dataset into train, validation and test sets
    # default split is 70% train, 20% validation and 10% test
    mozambique2021_dataset.split_to_sets(train_val_test_split=train_val_test_split)
