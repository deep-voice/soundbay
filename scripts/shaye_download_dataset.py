import os
import sys
import pandas as pd
from tqdm import tqdm


def download_files_from_s3(df, s3_path, local_dir):
    # make dir local_dir
    os.makedirs(local_dir, exist_ok=True)
    df['full_path_filename'] = df['s3_path'] 
    for irow in tqdm(df.iterrows()):
        filename = irow[1]['full_path_filename']
        base_filename = os.path.basename(filename)
        # stripped_filename = filename.split('/')[-1]
        # Make sure stripped filename does't exist in local_dir
        if not os.path.exists(os.path.join(local_dir, base_filename)):
            execute_str = f'aws s3 cp "{filename}" {local_dir}'
            # print(f'Downloading {filename} to {local_dir}')
            os.system(execute_str)
        else:
            print(f'File {base_filename} already exists in {local_dir}')

def compare_two_dfs(df1, df2):
    # Compare two DataFrames and return the differences
    # diff_df = pd.concat([df1, df2]).drop_duplicates(keep=False)
    # df1['filename']
    # df1['filename'].apply(lambda x: x[:-4])
    # df1_squeezed = df1[['filename', 'label', 'call_length']].copy()
    # df2_squeezed = df2[['s3_path', 'label', 'call_length']].copy()
    # df2_squeezed['filename'] = df2_squeezed['s3_path'].apply(lambda x: x.split('/')[-1])
    # df2_squeezed = df2_squeezed.drop(columns=['s3_path'])

    # #filter out label=0 on both dfs
    # df1_squeezed = df1_squeezed[df1_squeezed['label'] != 0]
    # df2_squeezed = df2_squeezed[df2_squeezed['label'] != 0]
    # #check if values of df1_squeezed are present in df2_squuezed across the columns 'filename' 'label' and 'call_length'
    # diff_df = df2_squeezed[~df2_squeezed.isin(df1_squeezed)].dropna()
    # split to train an test
    df1['filename'] = df1['filename'].apply(lambda x: x[:-4])
    # cast filename column as str
    df1['filename'] = df1['filename'].astype(str)
    # Split train and val based on 'label' column and avoid leakage using the 'filename' column
    from sklearn.model_selection import train_test_split
    train, val = train_test_split(df1, test_size=0.2, stratify=df1['label'], random_state=42)
    # concat train and val by adding columns 'split_type' and 'split_type' with values 'train' and 'val' respectively
    train['split_type'] = 'train'
    val['split_type'] = 'val'
    train_val = pd.concat([train, val])
    # save train_val to csv /datadrive/soundbay_backup/soundbay/shaye_annotations_15_9_24_processed.csv
    from sklearn.model_selection import GroupShuffleSplit
    # Split using GroupShuffleSplit
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)

    # The 'groups' argument is based on the 'filename' column
    train_idx, test_idx = next(splitter.split(train_val, groups=train_val['filename']))

    # Create train and test DataFrames
    train_df = train_val.iloc[train_idx]
    test_df = train_val.iloc[test_idx]
    # count number of unique values in 'label' column in each df
    train_c = train_df['label'].value_counts()
    test_c = test_df['label'].value_counts()
    # count filename in each df
    train_filename = train_df['filename'].count()
    test_filename = test_df['filename'].count()
    # check that unique filename in test_df are not in train_df
    assert len(set(train_df['filename']).intersection(set(test_df['filename']))) == 0
    # concat train_df and test_df
    train_val = pd.concat([train_df, test_df])
    # reset index
    train_val.reset_index(drop=True, inplace=True)


    train_val.to_csv('/datadrive/soundbay_backup/soundbay/shaye_annotations_15_9_24_processed.csv', index=False)


    return diff_df



if __name__ == '__main__':
    csv_path = '/datadrive/soundbay_backup/soundbay/shaye_annotations_22_10_24.csv'
    s3_path = 's3://deepvoice-user-uploads/shayetudor@gmail.com/dropbox/cods/'
    output_dir = '/datadrive/shaye_train_data/shaye_cods_data_15_9_24/'
    df = pd.read_csv(csv_path)
    download_files_from_s3(df, s3_path, output_dir)
#     df1_path = '/datadrive/soundbay_backup/soundbay/shaye_annotations_15_9_24.csv'
#     df2_path = '/datadrive/shaye_train_data/shaye_annotations_train_val_noleak_prod.csv'
#     df1 = pd.read_csv(df1_path)
#     df2 = pd.read_csv(df2_path)
#     diff_df = compare_two_dfs(df1, df2)
