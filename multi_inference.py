import pandas as pd
import os
import glob
from tqdm import tqdm

list_untagged = glob.glob('/Volumes/Elements/DATA_Racheli/1/*.[Ww][Aa][Vv]')

models = ["lgteb9qj"]
ths = [0.7]


if __name__ == "__main__":
    for (model,th) in zip(models,ths):
        for f in tqdm(list_untagged):
            os.system(f'sh run_infer.sh {f} {model} {th}')

