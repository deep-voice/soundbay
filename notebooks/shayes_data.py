import pandas as pd
import os
from tqdm import tqdm
from soundbay.utils.metadata_processing import bg_from_non_overlap_calls
import boto3
from urllib.parse import urlparse
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split








# split to val and train and add bg noise
def process_val_train_bg(df_path):
    df = pd.read_csv(df_path, index_col=0)
    df = df.dropna()
    # df = df[df['Species'] !='UN']
    sorted_dfs = df.sort_values('begin_time', ascending=True)
    newdf = bg_from_non_overlap_calls(sorted_dfs)

    newdf['filename'] = newdf['filename'].apply(lambda x: x[:-4])
    newdf['s3_path'] = newdf['s3_path'] + '/' + newdf['filename'] + '.wav'
    newdf['filename'] = newdf['filename'].astype('str')
    newdf['s3_path'] = newdf['s3_path'].astype('str')


    group_labels = newdf.groupby('annotations_filename')['label'].agg(lambda x: x.mode()[0])

    train_groups, test_groups = train_test_split(
        group_labels.index,
        stratify=group_labels.values,
        test_size=0.2,
        random_state=42
    )

    df_train = newdf[newdf['annotations_filename'].isin(train_groups)].reset_index(drop=True)
    df_val = newdf[newdf['annotations_filename'].isin(test_groups)].reset_index(drop=True)


    return df_train, df_val

#download files
def download_files(df2, local_folder):
    # Set your local download folder
    df2['full_s3_path'] = df2['s3_path'] + '/' + df2['filename']


    # Make sure the folder exists
    os.makedirs(local_folder, exist_ok=True)

    # Initialize S3 client
    s3 = boto3.client('s3')

    # Loop through the full_s3_path column
    for s3_path in df2['full_s3_path']:
        try: 
            parsed = urlparse(s3_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            filename = os.path.basename(key)
            local_path = os.path.join(local_folder, filename)
            if os.path.exists(local_path):
                print(f'file {filename} exists')
                continue

            print(f"Downloading {s3_path} to {local_path}...")
            s3.download_file(bucket, key, local_path)
        except:
            print(f'error in :{s3_path}')


# Function to get files from a directory
def get_files(main_dir):
    files = []
    for r, d, f in os.walk(main_dir):
        for file in f:
            files.append(os.path.join(r, file))
    files = [f for f in files if f.endswith('.txt') and 'sounds' in f.lower() and 'quiet' not in f.lower()]
    all_dfs = {}
    for file in files:
        df = pd.read_csv(file, sep='\t')
        df = df.reset_index()
        file = os.path.basename(file)
        all_dfs[file] = df
    return all_dfs

# Function to process a dataframe
def process_df(df):
    # convert 'Begin File' to string type
    df['Begin File'] = df['Begin File'].astype(str)
    df['filename'] = df['Begin File']
    # .apply(lambda x:x[:-4])
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
        fold = o.get('Prefix').lstrip(s3_path)
        # if 'wav' in fold:
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
            continue
    print(f'No file for: {filename} in none of the folders')
    return None

# Main processing logic
def process_annotations_files(get_files, process_df, get_folders, check_file_exists, main_dir):
    all_dfs = get_files(main_dir)

    s3_parent_folder = 'shayetudor@gmail.com/dropbox/cods/'
    list_of_folders = [i[:-1] for i in get_folders(s3_parent_folder)]

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

    all_edited_dfs.to_csv('shaye_annotations_added_new_extended.csv')

if __name__ == "__main__":
    df2_path = '/home/ubuntu/soundbay/shaye_annotations_added_new_extended.csv'
    # download_to_folder = '/home/ubuntu/soundbay/datasets/shaye_data_extended'
    # df2 = pd.read_csv(df2_path)
    # download_files(df2, download_to_folder)





    # main_dir = '/Users/tomernahshon/Documents/soundbay/shaye_data/all_txt_19.7.25'
    main_dir = '/home/ubuntu/soundbay/datasets/shaye_txt'
    process_annotations_files(get_files, process_df, get_folders, check_file_exists, main_dir)
    # #load the saved csv file
    # df = pd.read_csv('shaye_annotations_added_new.csv')
    # print(df.shape)
    # print(f'Unique values in annotations_filename: {df["annotations_filename"].unique()}')
    # df2 = df.dropna()
    # print(f'Number of rows after dropping NaN: {df2.shape[0]}')
    # # df2 = df2[df2['call_length'] < 10]
    # # find the diff in annotations_filename between the original and the processed
    # original_annotations_filenames = set(df['annotations_filename'].unique())
    # processed_annotations_filenames = set(df2['annotations_filename'].unique())
    # diff_annotations_filenames = original_annotations_filenames - processed_annotations_filenames

    # df2.loc[:, 'full_s3_path'] = df2['s3_path'].str.rstrip('/') + '/' + df2['filename']

    # print(f'Number of rows after filtering by call length: {df2.shape[0]}')
    # # save the df2
    # df2.to_csv('shaye_annotations_added_nan_removed.csv', index=False)

    # process dataframe to val and train
    # train_df, val_df = process_val_train_bg(df2_path) 
    # save_path = '/home/ubuntu/soundbay/datasets/shaye_dfs'
    # train_df.to_csv(os.path.join(save_path, 'train.csv'))
    # val_df.to_csv(os.path.join(save_path, 'val.csv'))


        

