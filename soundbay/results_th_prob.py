import pandas as pd
import os
import numpy as np

from soundbay.results_analysis import inference_csv_to_raven

BASE_PATH = "/Users/shai/personal/deepvoice/Fish_Rach/Results/res_jul10"
THs = [0.15, 0.25, 0.35]
SEQ_LEN = 0.5
FILES = ["Inference_results-2024-07-13_00-47-43-54nytj6c-2up_20231114_0026_to_0035.csv",
         "Inference_results-2024-07-13_00-47-53-54nytj6c-2up_20231114_1800_to_1809.csv"]

for threshold in THs:
    for file in FILES:
        file = os.path.join(BASE_PATH, file)
        results_df = pd.read_csv(file)
        results_df.columns = ["channel", 0, 1]
        df = inference_csv_to_raven(results_df, results_df.shape[1]-1, SEQ_LEN, 1, threshold, 'call',
                               channel=1, max_freq=2048)

        df = df.sort_values('Begin Time (s)')
        df.to_csv(os.path.join(f"{file.split('.csv')[0]}_th{threshold}.txt"), index=False, sep='\t')