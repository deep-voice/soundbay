#!/usr/bin/python
import pandas as pd
import numpy as np
import sys
if __name__=="__main__":
    _, csv_path, download_path = sys.argv
    df = pd.read_csv(csv_path)
    print(f'size of the dataframe is {df.shape}')
    print(df.columns)
    unique_files, freq = np.unique(df['filename'], return_counts=True)
    print(unique_files, freq)
    print(csv_path, download_path)
