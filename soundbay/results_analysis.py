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

def inference_csv_to_raven(probsdataframe: pd.DataFrame, num_classes, seq_length: float, selected_class: str,threshold: float = 0.5, class_name: str = "call") -> pd.DataFrame:
    """ Converts a csv file containing the inference results to a raven csv file.
        Args: probsdataframe: a pandas dataframe containing the inference results.
                      num_classes: the number of classes in the dataset.
                      seq_length: the length of the sequence.
                      selected_class: the class to be selected.
                      threshold: the threshold to be used for the selection.
                      class_name: the name of the class for which the raven csv file is generated.

        Returns: a pandas dataframe containing the raven csv file.
    """

    relevant_columns = probsdataframe.columns[-int(num_classes):]
    relevant_columns_df = probsdataframe[relevant_columns]
    len_dataset = relevant_columns_df.shape[0]  # number of segments in wav
    seq_len = seq_length
    if_positive = probsdataframe[selected_class] > threshold  # check if the probability is above the threshold
    if "begin_time" in probsdataframe.columns: #if the dataframe has metadata
        all_begin_times = probsdataframe["begin_time"].values
    else: #if the dataframe came from a file with no ground truth
        all_begin_times = np.arange(0, len_dataset * seq_len, seq_len)

    begin_times = all_begin_times[if_positive]  # get the begin times of the positive segments
    end_times = np.round(begin_times+seq_len, decimals=3)
    if end_times[-1] > round(len_dataset*seq_len,1):
        end_times[-1] = round(len_dataset*seq_len,1)  # cut off last bbox if exceeding eof

    # create columns for raven format
    low_freq = np.zeros_like(begin_times)
    high_freq = np.ones_like(begin_times)*20000  # just tall enough bounding box
    view = ['Spectrogram 1']*len(begin_times)
    selection = np.arange(1,len(begin_times)+1)
    annotation = [class_name]*len(begin_times)
    channel = np.ones_like(begin_times).astype(int)
    bboxes = {'Selection': selection, 'View': view, 'Channel': channel,
              'Begin Time (s)': begin_times, 'End Time (s)': end_times,
              'Low Freq (Hz)': low_freq, 'High Freq (Hz)': high_freq, 'Annotation': annotation}

    annotations_df = pandas.DataFrame(data = bboxes)  # create dataframe

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
    seq_length_default = 0.2  # relevant only if analyzing an inference single file (not a dataset)
    seq_length = results_df["call_length"].unique()[0] if "call_length" in results_df.columns else seq_length_default  # extract call length from inference results

    save_raven = True
    if "label" in results_df.columns:
        # log analysis metrics
        analysis_logging(results_df,num_classes)

    # create raven results
    if save_raven:
        thresholdtext = int(threshold*10)
        raven_filename = f"raven_annotations-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{inference_csv_name[37:]}-thresh0{thresholdtext}.csv"
        raven_output_file = output_dirpath / raven_filename

        raven_df = inference_csv_to_raven(results_df, num_classes, seq_length, selected_class= selected_class_column, threshold=threshold, class_name=name_col)
        raven_df.to_csv(index=False, path_or_buf=raven_output_file, sep="\t")


    print("Finished analysis.")


if __name__ == "__main__":
    analysis_main()