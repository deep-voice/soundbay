import numpy as np
import sys
import pandas as pd
from tqdm import tqdm
import pathlib
from pytz import UTC
from datetime import datetime
pandas_version = pd.__version__
assert pandas_version == '2.0.3', f'Pandas version {pandas_version} is not supported, please change to 2.0.3'

joining_dict = {'bma': 'bmabz',
                'bmb': 'bmabz',
                'bmz': 'bmabz',
                'bmd': 'd',
                'bpd': 'd',
                'bp20': 'bp',
                'bp20plus': 'bp'}


def compute_confusion_matrix(ground_truth, predictions, all_classes):
    """
    Compute the confusion matrix per given ground_truth and predictions.

    :param ground_truth: pd.DataFrame of the ground truth annotations
    :param predictions: pd.DataFrame of the predicted annotations
    :param all_classes: list of all the classes in the dataset
    :return: confusion matrix including tp, fp, fn, recall, precision
    """
    conf_matrix = pd.DataFrame(columns=['tp', 'fp', 'fn'], index=ground_truth.annotation.unique())
    for class_id in all_classes:
        ground_truth_class = ground_truth.loc[ground_truth.annotation == class_id]
        class_predictions = predictions.loc[predictions.annotation == class_id]
        conf_matrix.loc[class_id, 'tp'] = ground_truth_class['detected'].sum()
        conf_matrix.loc[class_id, 'fp'] = len(class_predictions) - class_predictions['correct'].sum()
        conf_matrix.loc[class_id, 'fn'] = len(ground_truth_class) - ground_truth_class['detected'].sum()

    conf_matrix['recall'] = conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'])
    conf_matrix['precision'] = conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fp'])

    return conf_matrix


def compute_confusion_matrix_per_dataset(ground_truth, predictions, all_classes):
    """
    Yields the confusion matrix per dataset

    :param ground_truth: pd.DataFrame of the ground truth annotations
    :param predictions: pd.DataFrame of the predicted annotations
    :param all_classes: list of all the classes in the dataset
    :return: str, pd.DataFrame  (dataset_name, confusion_matrix)
    """
    for dataset_name, ground_truth_dataset in ground_truth.groupby('dataset'):
        predictions_dataset = predictions.loc[predictions.dataset == dataset_name]
        conf_matrix_dataset = compute_confusion_matrix(ground_truth_dataset, predictions_dataset, all_classes)
        yield dataset_name, conf_matrix_dataset


def join_annotations_if_dir(path_to_annotations):
    """
    Join all the annotations of one directory, and return as pandas DataFrame

    :param path_to_annotations: pathlib.Path pointing to the folder with all the csv prediction files to evaluate
    or only one of them (either folder or file are good)
    :return: pandas.DataFrame with all the annotations combined
    """
    if path_to_annotations.is_dir():
        annotations_list = []
        for annotations_path in path_to_annotations.glob('*.csv'):
            annotations = pd.read_csv(annotations_path, parse_dates=['start_datetime', 'end_datetime'])
            annotations_list.append(annotations)
        total_annotations = pd.concat(annotations_list, ignore_index=True)
    else:
        total_annotations = pd.read_csv(path_to_annotations, parse_dates=['start_datetime', 'end_datetime'])

    return total_annotations


def run(predictions_path, ground_truth_path, iou_threshold=0.3):
    """
    Run the evaluation protocol. Prints the obtained results per dataset and in total

    :param predictions_path: str or pathlib.Path pointing to the folder with all the csv prediction files to evaluate
    or only one of them (either folder or file are good)
    :param ground_truth_path: str or pathlib.Path pointing to the folder with all the csv ground truth files to evaluate
    or only one of them (either folder or file are good)
    :param iou_threshold: float, 0 to 1 for the IOU threshold to consider for evaluation
    :return: None
    """
    if type(predictions_path) is str:
        predictions_path = pathlib.Path(predictions_path)
    if type(ground_truth_path) is str:
        ground_truth_path = pathlib.Path(ground_truth_path)
    ground_truth = join_annotations_if_dir(ground_truth_path)
    # predictions = pd.read_csv(predictions_path)

    # predictions = refactor_dv_predictions(ground_truth, predictions)

    predictions = join_annotations_if_dir(predictions_path)

    ground_truth = ground_truth.replace(joining_dict)
    predictions = predictions.replace(joining_dict)
    ground_truth['detected'] = 0
    predictions['correct'] = 0

    for (dataset_name, wav_path_name), wav_predictions in tqdm(predictions.groupby(['dataset', 'filename']),
                                               total=len(predictions.filename.unique())):
        ground_truth_wav = ground_truth.loc[ground_truth['filename'] == wav_path_name]
        for class_id, class_predictions in wav_predictions.groupby('annotation'):
            ground_truth_wav_class = ground_truth_wav.loc[ground_truth_wav['annotation'] == class_id]
            ground_truth_not_detected = ground_truth_wav_class.loc[ground_truth_wav_class.detected == 0]

            if ground_truth_not_detected.empty:
                continue
            for i, row in class_predictions.iterrows():
                # For each row, compute the minimum end and maximum start with all the ground truths
                min_end = np.minimum(row['end_datetime'], ground_truth_not_detected['end_datetime'])
                max_start = np.maximum(row['start_datetime'], ground_truth_not_detected['start_datetime'])
                inter = (min_end - max_start).dt.total_seconds().clip(0)
                union = (row['end_datetime'] - row['start_datetime']).total_seconds() + (
                    (ground_truth_not_detected['end_datetime'] - ground_truth_not_detected['start_datetime']).dt.total_seconds()) - inter
                iou = inter / union

                # Save the maximum iou for that prediction
                if iou.max() > iou_threshold:
                    predictions.loc[i, 'correct'] = 1
                    ground_truth_index = ground_truth_not_detected.iloc[iou.argmax()].name
                    ground_truth.loc[ground_truth_index, 'detected'] = 1

    all_classes = ground_truth.annotation.unique()
    for dataset_name, conf_matrix_dataset in compute_confusion_matrix_per_dataset(ground_truth, predictions, all_classes):
        print(f'Results dataset {dataset_name}')
        print(conf_matrix_dataset)

    conf_matrix = compute_confusion_matrix(ground_truth, predictions, all_classes)
    print('Final results')
    print(conf_matrix)

def refactor_dv_predictions(ground_truth, predictions, mock=True):
    if mock:
        # This is a mockup of the predictions for the sake of format compliance with soundbay repo output
        predictions['dataset'] = ground_truth['dataset'].iloc[0]
        predictions['filename'] = ground_truth['filename']
        predictions = predictions.dropna()
        predictions = predictions.drop(columns=['Downsweeps', 'Tones', 'Squeaks', 'Clicks'])
        predictions['Upsweeps'] = predictions['Upsweeps'] > 0.5
        indices_true = predictions['Upsweeps'].index[predictions['Upsweeps'] == True].tolist()
        predictions['annotation'] = np.where(predictions['Upsweeps'] == True, 'bma', 'bmb')




    # The format of the filenmae is 2015-04-19T06-00-00_000.wav
    predictions['start_date'] = predictions['filename'].str.split('_').str[0].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%dT%H-%M-%S').replace(tzinfo=UTC)
    )

    # Find sequences of consecutive indices
    sequences = []
    current_sequence = [indices_true[0]]
    for i in range(1, len(indices_true)):
        if indices_true[i] == indices_true[i - 1] + 1:
            current_sequence.append(indices_true[i])
        else:
            sequences.append(current_sequence)
            current_sequence = [indices_true[i]]
    sequences.append(current_sequence)
    tuples_of_begin_and_end = [(i[0], i[-1]) for i in sequences]


    # Create a new DataFrame with the start and end times
    new_data = []
    for start, end in tuples_of_begin_and_end:
        start_time = predictions['begin_time'].iloc[start]
        end_time = predictions['end_time'].iloc[end]
        start_date = predictions['start_date'].iloc[start]
        end_date = start_date
        start_time = pd.to_datetime(start_date) + pd.to_timedelta(start_time, unit='s')
        end_time = pd.to_datetime(end_date) + pd.to_timedelta(end_time, unit='s')
        new_data.append({'start_datetime': start_time, 'end_datetime': end_time, 'annotation': predictions['annotation'].iloc[start]})
    temp_df = pd.DataFrame(new_data)
    predictions['start_datetime'] = temp_df['start_datetime']
    predictions['end_datetime'] = temp_df['end_datetime']
    predictions = predictions.dropna()
    predictions = predictions
    return predictions

def get_args(args):
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the predictions against the ground truth')
    parser.add_argument('--predictions_csv_path', type=str, help='Path to the predictions csv file or folder')
    parser.add_argument('--ground_truth_csv_path', type=str, help='Path to the ground truth csv file or folder')
    return parser.parse_args(args)


if __name__ == '__main__':
    # predictions_csv_path = pathlib.Path(input('Where are the predictions in csv format?'))
    # ground_truth_csv_path = pathlib.Path(input('Where are the ground truth in csv format?'))
    args = sys.argv[1:] 
    args = get_args(args)
    run(args.predictions_csv_path, args.ground_truth_csv_path)