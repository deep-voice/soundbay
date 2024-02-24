import os
import sys
import pandas as pd


def download_files_from_s3(df, s3_path, local_dir):
    # make dir local_dir
    os.makedirs(local_dir, exist_ok=True)
    for irow in df.iterrows():
        filename = irow[1]['filename']
        execute_str = f'aws s3 cp "{s3_path}{filename}.wav" {local_dir}'
        print(execute_str)
        os.system(execute_str)




if __name__ == '__main__':
    csv_path = '/home/azureuser/soundbay/datasets/shaye_dataset/shaye_annotations_train_val.csv'
    s3_path = 's3://deepvoice-user-uploads/##/dropbox/cods/'
    output_dir = '/home/azureuser/soundbay/datasets/shaye_dataset/data/'
    df = pd.read_csv(csv_path)
    download_files_from_s3(df, s3_path, output_dir)
