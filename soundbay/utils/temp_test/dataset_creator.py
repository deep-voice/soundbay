import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
from soundbay.utils.metadata_processing import load_n_adapt_raven_annotation_table_to_dv_dataset_requirements


class DatasetCreator:
    def __init__(self, annotations_dir, dataset_path, dataset_name, save_dir, cols_to_use=None, label_col_name='label'):
        self.annotations_dir = annotations_dir
        self.dataset_path = dataset_path
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.label_col_name = label_col_name
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

    def create_annotation_df(self):
        annotations_dir = self.annotations_dir
        filenames = os.listdir(annotations_dir)
        df_list = []
        print('FILENAMES:')
        for filename in sorted(filenames):
            print(filename)
            try:
                annotation_file_path = os.path.join(annotations_dir, filename)
                small_df = load_n_adapt_raven_annotation_table_to_dv_dataset_requirements(annotation_file_path)
                df_list.append(small_df)
            except UnicodeDecodeError:
                print(f'\nUnicodeDecodeError: {filename}')
                continue

        print(f'\nlen(df_list): {len(df_list)}')
        df_all_annotations = pd.concat(df_list)

        if self.label_col_name != 'label':
            self.df_all_annotations.rename(columns={f'{self.label_col_name}': 'label'}, inplace=True)

        cols2drop = [col for col in df_all_annotations.columns if col not in self.cols_to_use]
        df_all_annotations = df_all_annotations.drop(cols2drop, axis=1)

        print(f'\n unique filenames: \n {df_all_annotations.filename.unique()}')

        self.df_all_annotations = df_all_annotations

    def extract_unique_labels(self):
        print('\nAnnotations that are used: \n', self.df_all_annotations.label.unique())
        print('\nAnnotations and their counts: \n', self.df_all_annotations.label.value_counts(dropna=False))

        # find labels that only appear once
        all_unique_labels = self.df_all_annotations.label.value_counts()[
            self.df_all_annotations.label.value_counts() == 1].index.values

        # build small df containing the above unique labels
        df_unique_labels = self.df_all_annotations[self.df_all_annotations.label.isin(all_unique_labels)].copy()

        # save the unique labels as a csv for potential analysis
        save_path = os.path.join(self.save_dir, 'unique_labels_info.csv')
        df_unique_labels.to_csv(save_path, index=False)

    def standardize_labels(self, labels_to_consolidate=None, target_label='sc', nan_is_call=False):

        self.df_clean_annotations = self.df_all_annotations.copy()

        # Consolidate labels if required
        if labels_to_consolidate is None:
            print("Since no labels to consolidate were provided, we will skip this step.")
            return
        else:
            self.df_clean_annotations.replace(to_replace=labels_to_consolidate, value=target_label, inplace=True,
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
        self.df_bg['label'] = 'bg'
        self.df_bg['call_length'] = self.df_bg['bg_end_time'] - self.df_bg['bg_begin_time']
        self.df_bg = self.df_bg.rename({'bg_begin_time': 'begin_time', 'bg_end_time': 'end_time'}, axis=1)

    def concat_bg_and_calls(self):
        self.df_concat = pd.concat([self.df_bg, self.df_no_overlap[['begin_time', 'end_time', 'filename',
                                                                    'call_length', 'label']]])
        print("Sanity check \n")
        print("label counts: \n", self.df_concat.label.value_counts())
        print("number of unique files: \n", self.df_concat.filename.nunique())
        print("difference between calls and bg: \n", self.df_concat.label.value_counts()[0] - self.df_concat.label.value_counts()[1])
        print("if the difference is equal to the unique number of files, it all makes sense :)")

        save_path = os.path.join(self.save_dir, f'{self.dataset_name}_prepped_with_bg.csv')

        self.df_concat.to_csv(f'{save_path}', index=False)

    def split_to_sets(self, train_val_test_split=(0.7, 0.2, 0.1)):
        self.df_meta = self.df_concat[self.df_concat.label == 'w'].groupby('filename').agg(
            count_calls=pd.NamedAgg('begin_time', 'count'),
            sum_call_length=pd.NamedAgg('call_length', 'sum'),
            avg_call_length=pd.NamedAgg('call_length', 'mean'),
        ).sort_values('filename')
        self.df_meta['cum_call_length'] = self.df_meta.sum_call_length.cumsum()
        self.df_meta['cum_perc'] = 100 * self.df_meta['cum_call_length'] / (self.df_meta['sum_call_length'].sum())
        train_split, val_split, test_split = train_val_test_split*100
        train_files = self.df_meta.loc[self.df_meta['cum_perc'] < train_split].index.tolist()
        val_files = self.df_meta.loc[((self.df_meta['cum_perc'] > train_split) &
                                      (self.df_meta['cum_perc'] < train_split + val_split))].index.tolist()

        test_files = self.df_meta.loc[(self.df_meta['cum_perc'] > train_split + val_split)].index.tolist()

        call_length_total = self.df_meta.sum_call_length.sum()
        print("The percentage of splits between train, val and test is: \n")
        for files in [train_files, test_files, val_files]:
            print(np.round(self.df_meta.loc[files].sum_call_length.sum() / call_length_total * 100))

        df_train = self.df_concat[self.df_concat.filename.isin(train_files)]
        df_val = self.df_concat[self.df_concat.filename.isin(val_files)]
        df_test = self.df_concat[self.df_concat.filename.isin(test_files)]

        save_path_train = os.path.join(self.save_dir, f'{self.dataset_name}_prepped_with_bg_train.csv')
        save_path_val = os.path.join(self.save_dir, f'{self.dataset_name}_prepped_with_bg_train.csv')
        save_path_test = os.path.join(self.save_dir, f'{self.dataset_name}_prepped_with_bg_train.csv')

        df_train.to_csv(f'{save_path_train}', index=False)
        df_val.to_csv(f'{save_path_val}', index=False)
        df_test.to_csv(f'{save_path_test}', index=False)


if __name__ == "__main__":
    annotations_dir = '../../../datasets/mozambique_2021/annotation_files/'
    dataset_path = '../../../tests/assets/mozambique2021/recordings/'
    dataset_name = 'mozambique2021'
    save_dir = '../../../datasets/test_folder/'
    cols_to_use = ['begin_time', 'end_time', 'filename', 'call_length', 'label']
    label_call_name = 'Annotation'
    mozambique2021_dataset = DatasetCreator(annotations_dir, dataset_path, dataset_name, save_dir, cols_to_use, label_call_name)
    print(2)

    mozambique2021_dataset.create_annotation_df

    print(3)

    print (mozambique2021_dataset.df_all_annotations)


    mozambique2021_dataset.extract_unique_labels()

    '''
    extract_unique_labels(self)

    standardize_labels(self, labels_to_consolidate=None, target_label='sc', nan_is_call=False)

    merge_overlapping_calls(self):

    merge_overlapping_rows(self, df) -> pd.DataFrame:

    reset_overlap_accessory_columns(self, df) -> pd.DataFrame:

    mark_overlapping_rows(self, df) -> pd.DataFrame:

    mark_background(self):

    concat_bg_and_calls(self):

    split_to_sets(self, train_val_test_split=(0.7, 0.2, 0.1)):
    '''