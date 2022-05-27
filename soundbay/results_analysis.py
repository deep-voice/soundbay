import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import os
import pandas
import datetime
from scipy.special import softmax
from soundbay.utils.logging import Logger


def analysis_logging(results_df,num_classes):
    """
    Logs the results of the analysis to a wandb.

    Args:
        results_df: The dataframe containing the results of the analysis.
        gt_metadata: The ground truth metadata.

    Returns:
        None

    """
    columns_array = results_df.columns[-int(num_classes):]
    results_array = np.array(results_df[columns_array])
    metrics_dict = Logger.get_metrics_dict(results_df["label"].values.tolist(),
                                       np.argmax(results_array, axis=1).tolist(),
                                       results_array)
    print(metrics_dict)

def inference_csv_to_raven(probsdataframe: pd.DataFrame, seq_length: float, selected_class: str,threshold: float = 0.5, class_name: str = "call") -> pd.DataFrame:
    # transforms the probability dataframe to a raven format with class_1 predictions as the annotated bounding boxes
    len_dataset = probsdataframe.shape[0]  # number of segments in wav
    seq_len = seq_length
    all_begin_times = np.arange(0, len_dataset * seq_len, seq_len)

    if_positive = probsdataframe[selected_class] > threshold  # check if the probability is above the threshold
    begin_times = all_begin_times[if_positive]
    # begin_times = np.round(all_begin_times[if_positive], decimals=1) #rounded to avoid float errors (e.g. 24.00000000000001)

    end_times = np.round(begin_times+seq_len, decimals=1)
    if end_times[-1] > round(len_dataset*seq_len,1):
        end_times[-1] = round(len_dataset*seq_len,1) #cut off last bbox if exceeding eof
    low_freq = np.zeros_like(begin_times)
    high_freq = np.ones_like(begin_times)*20000 #just tall enough bounding box
    view = ['Spectrogram 1']*len(begin_times)
    selection = np.arange(1,len(begin_times)+1)
    annotation = [class_name]*len(begin_times)
    channel = np.ones_like(begin_times).astype(int)
    bboxes = {'Selection': selection, 'View': view, 'Channel': channel,
              'Begin Time (s)': begin_times, 'End Time (s)': end_times,
              'Low Freq (Hz)': low_freq, 'High Freq (Hz)': high_freq, 'Annotation': annotation}
    annotations_df = pandas.DataFrame(data = bboxes)




    return annotations_df

def analysis_main() -> None:
    """
    The main function for running an analysis on a model inference results for required Dataset.

    """
    # configurations:

    workdir = Path(os.getcwd())

    output_dirpath = workdir.parent.absolute() / "outputs"
    output_dirpath.mkdir(exist_ok=True)
    inference_csv_name = "Inference_results-2022-05-03_17-01-10-best_margin0_5-220302_0054"
    inference_results_path = output_dirpath / Path(inference_csv_name + ".csv")
    num_classes = 2
    threshold = 1/num_classes  # threshold for the classifier in the raven results
    results_df = pd.read_csv(inference_results_path)
    name_col = str(1) # selected class for raven results
    # go through columns and find the one containing the selected class
    column_list = results_df.columns
    selected_class_column = [s for s in column_list if name_col in s][0]
    seq_length = results_df["call_length"].unique()[0]  # extract call length from inference results

    save_raven = True

    analysis_logging(results_df,num_classes)

    if save_raven:
        thresholdtext = int(threshold*10)
        raven_filename = f"raven_annotations-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{inference_csv_name[37:]}-thresh0{thresholdtext}.csv"
        raven_output_file = output_dirpath / raven_filename
        relevant_columns = results_df.columns[-int(num_classes):]
        relevant_columns_df = results_df[relevant_columns]
        raven_df = inference_csv_to_raven(relevant_columns_df, seq_length, selected_class_column, threshold=threshold, class_name=name_col)
        raven_df.to_csv(index=False, path_or_buf=raven_output_file, sep="\t")


    print("Finished analysis.")


if __name__ == "__main__":
    analysis_main()