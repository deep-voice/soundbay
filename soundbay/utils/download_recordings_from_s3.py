#!/usr/bin/python
from pydoc import describe
import pandas as pd
import subprocess
import numpy as np
from tqdm import tqdm
import sys
if __name__=="__main__":
    _, csv_path, download_path = sys.argv
    df = pd.read_csv(csv_path)
    print(f'size of the dataframe is {df.shape}')
    print(df.columns)
    unique_files, freq = np.unique(df['filename'], return_counts=True)
    # print(unique_files, freq)
    iterator = tqdm(list(unique_files), desc='Downloading files from s3...')
    for ifile in iterator:
        run_cmd = f'aws s3 cp {ifile} {download_path}'
        subprocess.run(run_cmd, shell=True)
    print(csv_path, download_path)
