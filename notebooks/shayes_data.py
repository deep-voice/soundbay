import pandas as pd
import os
import sys
import numpy as np
from tqdm import tqdm
from soundbay.utils.metadata_processing import bg_from_non_overlap_calls
import boto3

# Function to get files from a directory
def get_files(main_dir):
    files = []
    for r, d, f in os.walk(main_dir):
        for file in f:
            files.append(os.path.join(r, file))
    files = [f for f in files if f.endswith('.txt')]
    all_dfs = {}
    for file in files:
        df = pd.read_csv(file, sep='\t')
        file = os.path.basename(file)
        all_dfs[file] = df
    return all_dfs

# Function to process a dataframe
def process_df(df):
    # convert 'Begin File' to string type
    df['Begin File'] = df['Begin File'].astype(str)
    df['filename'] = df['Begin File'][:-4]
    df['call_length'] = df['End Time (s)'] - df['Begin Time (s)']
    df['begin_time'] = df['File Offset (s)']
    df['end_time'] = df['begin_time'] + df['call_length']
    df['label'] = 1
    return df

# Function to get folders from S3
def get_folders(s3_path, my_bucket='deepvoice-user-uploads'):
    list_of_folders = []
    client = boto3.client('s3')
    result = client.list_objects(Bucket=my_bucket, Prefix=s3_path, Delimiter='/')
    for o in result.get('CommonPrefixes'):
        fold = o.get('Prefix').strip(s3_path)
        if 'wav' in fold:
            list_of_folders.append(fold)
    return list_of_folders

# Function to check if a file exists in S3
def check_file_exists(s3_path, filename, s3, folders_list, my_bucket='deepvoice-user-uploads'):
    for folder in folders_list:
        try:
            path = os.path.join(s3_path, folder, filename)
            s3.head_object(Bucket=my_bucket, Key=path)
            return 's3://' + my_bucket + '/' + s3_path + folder
        except:
            pass
    return None

# Main processing logic
def process_annotations_files(get_files, process_df, get_folders, check_file_exists, main_dir):
    all_dfs = get_files(main_dir)

    s3_parent_folder = 'shayetudor@gmail.com/dropbox/cods/'
    list_of_folders = [i[:-1] + ' tudor' for i in get_folders(s3_parent_folder)]

    columns_to_keep = ['filename', 'call_length', 'begin_time', 'end_time', 'label', 'Species']
    s3 = boto3.client('s3')

    all_edited_dfs = []
    for annotations_filename, df in tqdm(all_dfs.items()):
        if 'sounds' in annotations_filename.lower() and 'quiet' not in annotations_filename.lower():
            processed_df = process_df(df)
            processed_df = processed_df[columns_to_keep]
            processed_df.loc[:, 'annotations_filename'] = annotations_filename
            for i, row in processed_df.iterrows():
                if type(row['filename']) != str:
                    continue
                filename = row['filename']
                s3_specific_path = check_file_exists(s3_parent_folder, filename, s3, list_of_folders)
                processed_df.loc[i, 's3_path'] = s3_specific_path if s3_specific_path else 'None'
            all_edited_dfs.append(processed_df)

    all_edited_dfs = pd.concat(all_edited_dfs)
    # all_edited_dfs = all_edited_dfs[(all_edited_dfs['s3_path'] != 'None')].dropna()
    # all_edited_dfs = all_edited_dfs[all_edited_dfs['call_length'] < 10]

    # sorted_dfs = all_edited_dfs.sort_values('begin_time', ascending=True)
    # newdf = bg_from_non_overlap_calls(sorted_dfs)

    # newdf['filename'] = newdf['filename'].apply(lambda x: x[:-4])
    # newdf['s3_path'] = newdf['s3_path'] + '/' + newdf['filename'] + '.wav'

    all_edited_dfs.to_csv('shaye_annotations_added_new.csv', index=False)

if __name__ == "__main__":
    main_dir = '/Users/tomernahshon/Documents/soundbay/shaye_data/all_txt_19.7.25'
    # process_annotations_files(get_files, process_df, get_folders, check_file_exists, main_dir)
    #load the saved csv file
    df = pd.read_csv('shaye_annotations_added_new.csv')
    print(df.shape)
    print(f'Unique values in annotations_filename: {df["annotations_filename"].unique()}')
    df2 = df.dropna()
    print(f'Number of rows after dropping NaN: {df2.shape[0]}')
    # df2 = df2[df2['call_length'] < 10]
    # find the diff in annotations_filename between the original and the processed
    original_annotations_filenames = set(df['annotations_filename'].unique())
    processed_annotations_filenames = set(df2['annotations_filename'].unique())
    diff_annotations_filenames = original_annotations_filenames - processed_annotations_filenames

    print(f'Number of rows after filtering by call length: {df2.shape[0]}')

