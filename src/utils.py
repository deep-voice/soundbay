import librosa
import numpy as np
import torch
import soundfile as sf
import wandb
from sklearn import metrics
from unittest.mock import Mock
from omegaconf import OmegaConf
import boto3
from pathlib import Path
from typing import Union
from tqdm import tqdm
import pandas as pd
import re


def load_audio(filepath: str, sr: int, max_val=0.9):
    """
    load audio signal from file using librosa load function, and normalize the vector
    Args:
        filepath: location of audio file directory
        sr: sampling rate
        max_val: maximum value for normalization

    Returns:

    """
    audio, sr = librosa.core.load(filepath, sr)
    return norm_audio(audio, max_val)


def norm_audio(audio, max_val=0.9):
    """
    This function normalizes audio segment to a specified value range
    Args:
        audio: signal vector to be normalized
        max_val: maximum value

    Returns:
        normalized audio vector
    """
    if max_val:
        if len(audio.shape) == 2:
            audio = audio / audio.abs().max(dim=1)[0].view(-1, 1) * max_val
        elif len(audio.shape) == 1:
            audio = audio / np.max([audio.max(), abs(audio.min())]) * max_val
        else:
            raise Exception('accepts only dims == 1 or 2!')
    return audio


def write_torch_audio(filename, audio_tensor, sr):
    audio_tensor = audio_tensor.detach().squeeze().cpu().numpy()
    sf.write(filename, audio_tensor, sr)


class LossMeter(object):
    """
    A class for managing the losses for all the epochs
    """
    def __init__(self, name):
        """
        __init__ function initializes the loss meter
        Input:
            name: the name of the meter - string
        """
        self.name = name
        self.losses = []

    def reset(self):
        self.losses = []

    def add(self, val):
        self.losses.append(val)

    def summarize_epoch(self):
        if self.losses:
            return np.mean(self.losses)
        else:
            return 0

    def sum(self):
        return sum(self.losses)


class Logger:
    """
    A class for computing performance metrics, logging them and displaying them throughout the train
    """
    def __init__(self,
                 log_writer=Mock()
                 ):
        """
        __init__ initializes the logger and all the associated arrays and variables
        Input:
            log_writer: such as wandb, tensorboard etc.
        """
        self.log_writer = log_writer
        self.loss_meter_train, self.loss_meter_val = {}, {}
        self.loss_meter_keys = ['loss']
        self.init_losses_meter()
        self.pred_list = []
        self.pred_proba_list = []
        self.label_list = []
        self.metrics_dict = {'accuracy': [], 'f1score': [], 'precision': [], 'recall': [], 'auc': []}

    def log(self, log_num: int, flag: str):
        """logging losses using writer"""
        for key in self.loss_meter_keys:
            if flag == 'train':
                self.log_writer.log({f"{key}_train":
                                           self.loss_meter_train[key].summarize_epoch()}, step=log_num)
            elif flag == 'val':
                self.log_writer.log({f"{key}_val":
                                           self.loss_meter_val[key].summarize_epoch()}, step=log_num)

    def init_losses_meter(self):
        for key in self.loss_meter_keys:
            self.loss_meter_train[key] = LossMeter(key)
            self.loss_meter_val[key] = LossMeter(key)

    def reset_losses(self):
        for key in self.loss_meter_keys:
            self.loss_meter_train[key].reset()
            self.loss_meter_val[key].reset()

    def update_losses(self, loss, flag):
        losses = [loss]
        for key, current_loss in zip(self.loss_meter_keys, losses):
            if flag == 'train':
                self.loss_meter_train[key].add(current_loss.data.cpu().numpy().mean())
            elif flag == 'val':
                self.loss_meter_val[key].add(current_loss.data.cpu().numpy().mean())
            else:
                raise ValueError('accept train or flag only!')

    def update_predictions(self, pred_tuple):
        """update prediction and label list from current batch/iteration"""
        _, predicted = torch.max(pred_tuple[0].data, 1)
        label = pred_tuple[1].data
        self.pred_list += predicted.cpu().numpy().tolist()  # add current batch prediction to full epoch pred list
        self.pred_proba_list.append(torch.softmax(pred_tuple[0].data, 1).cpu().numpy())
        self.label_list += label.cpu().numpy().tolist()

    def calc_metrics(self, epoch):
        """calculates metrics, saves to tensorboard log & flush prediction list"""
        pred_proba_array = np.concatenate(self.pred_proba_list)
        self.metrics_dict = self.get_metrics_dict(self.label_list, self.pred_list, pred_proba_array)
        self.log_writer.log({'Accuracy': self.metrics_dict['accuracy']}, step=epoch)
        self.log_writer.log({'f1score': self.metrics_dict['f1score']}, step=epoch)
        self.log_writer.log({'precision': self.metrics_dict['precision']}, step=epoch)
        self.log_writer.log({'recall': self.metrics_dict['recall']}, step=epoch)
        self.log_writer.log({'auc': self.metrics_dict['auc']}, step=epoch)
        self.log_writer.log(
            {'ROC Curve': wandb.plot.roc_curve(self.label_list, pred_proba_array, labels=['Noise', 'Call'])},
            step=epoch
        )
        self.log_writer.log(
            {'PR Curve': wandb.plot.pr_curve(self.label_list, pred_proba_array, labels=['Noise', 'Call'])},
            step=epoch
        )
        self.pred_list = []  # flush
        self.label_list = []
        self.pred_proba_list = []

    @staticmethod
    def get_metrics_dict(label_list, pred_list, pred_proba_array):
        """calculate the metrics comparing the predictions to the ground-truth labels, and return them in dict format"""
        accuracy = metrics.accuracy_score(label_list, pred_list)  # calculate accuracy using sklearn.metrics
        f1score = metrics.f1_score(label_list, pred_list)
        precision = metrics.precision_score(label_list, pred_list)
        recall = metrics.recall_score(label_list, pred_list)
        auc = metrics.roc_auc_score(label_list, pred_proba_array[:, 1])
        metrics_dict = {'accuracy': accuracy, 'f1score': f1score, 'precision': precision, 'recall': recall, 'auc': auc}
        return metrics_dict


class LibrosaMelSpectrogram:
    """
    Defining and computing a mel-spectrogram transform using pre-defined parameters
    Input:
        init args:
            sr: sample rate
            n_mels: number of mel frequency bins
            fmin: minimum frequency
        call args:
            sample: audio segment vector
    Output:
        melspectrogram: numpy array with size n_mels*t (signal length)
    """
    def __init__(self, sr, n_mels, fmin):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_mels*20
        self.hop_length = int(sr / n_mels)
        self.fmin = fmin
        self.fmax = sr//2

    def __call__(self,  sample):
        slice_len = 1
        sample_numpy = sample.numpy().ravel() # convert to numpy and flatten
        melspectrogram = librosa.feature.melspectrogram(sample_numpy, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft,
                                                        hop_length=self.hop_length, fmin=self.fmin, fmax=self.fmax)
        return melspectrogram


class LibrosaPcen:
    """
    Defining and computing pcen (Per-Channel Energy Normalization) transform using pre-defined parameters
    Input:
        init args:
            sr: sample rate
            n_mels: number of mel-frequency bins
            fmin: minimum frequency
        call args:
            sample: audio segment vector
    Output:
         pcen: per-channel energy normalized version of signal - torch tensor
    """
    def __init__(self, sr, n_mels, fmin):
        self.sr = sr
        self.n_mels = n_mels
        self.slice_len = 1
        self.hop_length = int(self.sr / (self.n_mels / self.slice_len))
        self.fmin = fmin
        self.fmax = sr//2


    # sample,
    def __call__(self,  sample, gain=0.6, bias=0.1, power=0.2, time_constant=0.4, eps=1e-9):
        hop_length = int(self.sr / (self.n_mels / self.slice_len))
        pcen_librosa = librosa.core.pcen(sample, sr=self.sr, hop_length=self.hop_length, gain=gain, bias=bias, power=power,
                                         time_constant=time_constant, eps=eps)
        pcen_librosa = np.expand_dims(pcen_librosa, 0)  # expand dims to greyscale


        return torch.from_numpy(pcen_librosa).float()


class App:
    '''
    Class to be used as a global params and states handler across the project
    '''
    class _App:
        def __init__(self, args):
            self.args = args
            self.states = {}

    @classmethod
    def init(cls, args):
        App.inner = App._App(args)

    def __getattr__(self, item):
        return getattr(self.inner, item)


app = App()


def merge_with_checkpoint(run_args, checkpoint_args):
    """
    Merge into current args the needed arguments from checkpoint
    Right now we select the specific modules needed, can make it more generic if we'll see the need for it
    Input:
        run_args: dict_config of run args
        checkpoint_args: dict_config of checkpoint args
    Output:
        run_args: updated dict_config of run args
    """

    OmegaConf.set_struct(run_args, False)
    run_args.model = checkpoint_args.model
    run_args.data.test_dataset.preprocessors = checkpoint_args.data.train_dataset.preprocessors
    run_args.data.sample_rate = checkpoint_args.data.sample_rate
    OmegaConf.set_struct(run_args, True)
    return run_args


def walk(input_path):
    """
    helper function to yield folder's file content
    Input:
        input_path: the path of the folder
    Output:
        generator of files in directory tree
    """
    for p in Path(input_path).iterdir():
        if p.is_dir():
            yield from walk(p)
            continue
        yield p.resolve()


def upload_experiment_to_s3(experiment_id: str,
                            dir_path: Union[Path,str],
                            bucket_name: str,
                            include_parent: bool = True):
    """
    Uploads the experiment folder to s3 bucket
    Input:
        experiment_id: id of the experiment, taken usually from wandb logger
        dir_path: path to the experiment directory
        bucket_name: name of the desired bucket path
        include_parent: flag to include the parent of the experiment folder while saving to s3
    """
    dir_path = Path(dir_path)
    assert dir_path.is_dir(), 'should upload experiments as directories to s3!'
    object_global = f'{experiment_id}/{dir_path.parent.name}/{dir_path.name}' if include_parent \
        else f'{experiment_id}/{dir_path.name}'
    current_global = str(dir_path.resolve())
    upload_files = list(walk(dir_path))
    s3_client = boto3.client('s3')
    for upload_file in tqdm(upload_files):
        upload_file = str(upload_file)
        s3_client.upload_file(upload_file, bucket_name, upload_file.replace(current_global, object_global))


# <<<<<<< feature/EDA_script
def raven_to_df_annotations(annotations_path: str,
                             recording_path: str,
                             positive_tag_names: list = ['w','sc']):

    """
    Takes annotation files (selection table) created in Raven and turns it to a compatible annotations csv.
    """
    #create dataframe
    annotations=Path(annotations_path)
    recording=Path(recording_path)
    filelist = list(annotations.glob('*selections.txt')) #ignore irrelevent files
    metadata = []
    for file in filelist:
        dfTemp = pd.read_csv(file, sep="\t")
        dfTemp['filename'] = re.search("\.Table.1.selections", file.as_posix()).group()
        dfTemp['FirstCallTime'] = np.amin(dfTemp['Begin Time (s)'])
        dfTemp['LastCallTime'] = np.amax(dfTemp['End Time (s)'])
        metadata.append(dfTemp)

    metadata = pd.concat(metadata)
    metadata.rename(columns={'Begin Time (s)': 'begin_time', 'End Time (s)': 'end_time'}, inplace=True)
    print('Number of Labels:', metadata.shape[0])
    for tag in positive_tag_names:
        metadata['Annotation'] = metadata['Annotation'].replace(np.nan, tag, regex=True)
    #add recording length to dataframe
    wav_filelist = list(recording.glob('*.wav'))
    wav_filedict = {re.search("\.Table.1.selections", file.as_posix()).group(): {'path': file} for file in wav_filelist
                    if re.search("\.Table.1.selections", file.as_posix())}
    for key, value in wav_filedict.items():
        record_length = sf.info(value['path']).duration
        value.update({'length': record_length})
        wav_filedict[key] = value
    annotation_lengths = []
    for filename_txt in metadata['filename']:
        annotation_lengths.append(wav_filedict[filename_txt]['length'])
    metadata['TotalRecordLength'] = annotation_lengths
    # filter metadata to contain only desired call types
    filters = positive_tag_names
    metadata_calls = metadata[metadata.Annotation.isin(filters)]
    unique_files, idx = np.unique(metadata['filename'], return_index=True)
    # Find true length of call sequences ( get rid of over lapping-sequences)
    non_overlap_all = non_overlap_df(metadata)
    non_overlap_calls = non_overlap_df(metadata_calls)
    # label background segments
    bg_segments = []
    for file in (unique_files):
        file_df = non_overlap_all[non_overlap_all['filename'] == file]
        begin = np.array(file_df['begin_time'])
        end = np.array(file_df['end_time'])
        for item in end:
            next_beginning = begin[begin > item]
            if next_beginning.size == 0:
                break
            next_beginning = np.min(next_beginning)
            bg_segments.append([item, next_beginning, file])
    bg_segments = pd.DataFrame(bg_segments, columns=['begin_time', 'end_time', 'filename'])
    bg_segments['call_length'] = bg_segments['end_time'] - bg_segments['begin_time']
    bg_segments.sort_values(by=['call_length'])
    # add labels: 0 for background and 1 for call. TODO: modify if there are different call types
    bg_segments['label'] = np.zeros(bg_segments.shape[0], dtype=int)
    non_overlap_calls['label'] = np.ones(non_overlap_calls.shape[0], dtype=int)
    # combine to a csv
    combined_annotations = pd.concat([bg_segments, non_overlap_calls])
    return combined_annotations


def annotations_df_to_csv(annotations_dataset, dataset_name: str = 'recordings_2018_filtered'):
    filename = 'combined_annotations_' + dataset_name + '.csv'
    annotations_dataset.to_csv(filename, index=False)


def merge_calls(sorted_list):
    merged = [sorted_list[0]]
    for higher in sorted_list[1:]:
        lower = merged[-1]
        # test for intersection between lower and higher:
        # we know via sorting that lower[0] <= higher[0]
        if higher[0] <= lower[1]:
            upper_bound = max(lower[1], higher[1])
            merged[-1] = (lower[0], upper_bound)  # replace by merged interval
        else:
            merged.append(higher)
    return merged


def non_overlap_df(input_df):
    unique_files, idx = np.unique(input_df['filename'], return_index=True)
    non_overlap = []
    for file in unique_files:
        file_df = input_df[input_df['filename'] == file]

        begin = np.array(file_df['begin_time'])
        end = np.array(file_df['end_time'])
        begin_end = np.transpose(np.array((begin, end)))
        sorted_by_lower_bound = sorted(begin_end, key=lambda tup: tup[0])

        merged = merge_calls(sorted_by_lower_bound)

        p = pd.DataFrame({'begin_time': np.array(merged)[:, 0], 'end_time': np.array(merged)[:, 1], 'filename': file})
        non_overlap.append(p)

    non_overlap = pd.concat(non_overlap)
    non_overlap['call_length'] = non_overlap['end_time'] - non_overlap['begin_time']
    return non_overlap

  
def merge_with_checkpoint(run_args, checkpoint_args):
    """
    Merge into current args the needed arguments from checkpoint
    Right now we select the specific modules needed, can make it more generic if we'll see the need for it
    Input:
        run_args: dict_config of run args
        checkpoint_args: dict_config of checkpoint args
    Output:
        run_args: updated dict_config of run args
    """

    OmegaConf.set_struct(run_args, False)
    run_args.model = checkpoint_args.model
    run_args.data.test_dataset.preprocessors = checkpoint_args.data.train_dataset.preprocessors
    run_args.data.test_dataset.seq_length = checkpoint_args.data.train_dataset.seq_length
    run_args.data.sample_rate = checkpoint_args.data.sample_rate
    OmegaConf.set_struct(run_args, True)
    return run_args
