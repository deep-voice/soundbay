import pandas as pd
import numpy as np
from pathlib import Path
import pandas
import datetime
from soundbay.utils.logging import Logger
import argparse


def make_parser():
    parser = argparse.ArgumentParser("Results Analysis")

    parser.add_argument("--num_classes", default=2, help="number of classes for analysis")
    parser.add_argument("--filedir", default="../outputs", help="directory for inference file")
    parser.add_argument("--filename", default="", help="csv file of inference results for analysis")
    parser.add_argument("--selected_class", default="1", help = "selected class, will be annotated raven file")
    parser.add_argument("--save_raven", default=True, help ="whether or not to create a raven file")
    parser.add_argument("--threshold", default=0.5, type=float, help="threshold for the classifier in the raven results")

    return parser

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


def inference_csv_to_raven(results_df: pd.DataFrame, num_classes, seq_len: float, selected_class: str,
                           threshold: float = 0.5, class_name: str = "call",
                           max_freq: float = 20_000) -> pd.DataFrame:
    """ Converts a csv file containing the inference results to a raven csv file.
        Args: probsdataframe: a pandas dataframe cosntaining the inference results.
                      num_classes: the number of classes in the dataset.
                      seq_length: the length of the sequence.
                      selected_class: the class to be selected.
                      threshold: the threshold to be used for the selection.
                      class_name: the name of the class for which the raven csv file is generated.

        Returns: a pandas dataframe containing the raven csv file.
    """
    results_list = []
    for channel, df in results_df.groupby("channel"):
        relevant_columns = df.columns[-int(num_classes):]
        relevant_columns_df = df[relevant_columns]
        len_dataset = relevant_columns_df.shape[0]  # number of segments in wav
        if_positive = df[selected_class] > threshold  # check if the probability is above the threshold
        if "begin_time" in df.columns: #if the dataframe has metadata
            all_begin_times = df["begin_time"].values
        else: #if the dataframe came from a file with no ground truth
            all_begin_times = np.arange(0, len_dataset * seq_len, seq_len)

        begin_times = all_begin_times[if_positive]  # get the begin times of the positive segments
        class_probabilities = df[selected_class][if_positive].values  # get the probabilities of the positive segments
        end_times = np.round(begin_times+seq_len, decimals=3)
        if len(end_times >= 1):
            if end_times[-1] > round(len_dataset*seq_len,1):
                end_times[-1] = round(len_dataset*seq_len,1)  # cut off last bbox if exceeding eof

        # create columns for raven format
        low_freq = np.zeros_like(begin_times)
        high_freq = np.ones_like(begin_times)*max_freq
        view = ['Spectrogram 1']*len(begin_times)
        selection = np.arange(1,len(begin_times)+1)
        annotation = [f'{class_name}, {probability:.3f}' for probability in class_probabilities]
        channel = np.ones_like(begin_times).astype(int) * channel
        bboxes = {'Selection': selection, 'View': view, 'Channel': channel,
                  'Begin Time (s)': begin_times, 'End Time (s)': end_times,
                  'Low Freq (Hz)': low_freq, 'High Freq (Hz)': high_freq, 'Annotation': annotation}
        annotations_df = pandas.DataFrame(data=bboxes)  # create dataframe
        results_list.append(annotations_df)

    annotations_df = pd.concat(results_list, axis=0).sort_values('Begin Time (s)')
    annotations_df['Selection'] = np.arange(1, len(annotations_df) + 1)

    return annotations_df


def analysis_main() -> None:
    """
    The main function for running an analysis on a model inference results for required Dataset.

    """
    # configurations:
    args = make_parser().parse_args()
    assert args.filename, "filename argument is empty, you can't run analysis on nothing"
    output_dirpath = Path(args.filedir) #workdir.parent.absolute() / "outputs"
    output_dirpath.mkdir(exist_ok=True)
    save_raven = args.save_raven
    inference_csv_name = args.filename
    inference_results_path = output_dirpath / Path(inference_csv_name + ".csv")
    num_classes = int(args.num_classes)
    # threshold = 1/num_classes  # threshold for the classifier in the raven results
    threshold = args.threshold
    results_df = pd.read_csv(inference_results_path)
    name_col = args.selected_class  # selected class for raven results

    # go through columns and find the one containing the selected class
    column_list = results_df.columns
    selected_class_column = [s for s in column_list if name_col in s][0]
    seq_length_default = 0.2  # relevant only if analyzing an inference single file (not a dataset)
    seq_length = results_df["call_length"].unique()[0] if "call_length" in results_df.columns else seq_length_default  # extract call length from inference results

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