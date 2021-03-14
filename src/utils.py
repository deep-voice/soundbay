import librosa
import numpy as np
import torch
import torchaudio
import soundfile as sf
from sklearn import metrics
from unittest.mock import Mock


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
        self.label_list = []
        self.metrics_dict = {'accuracy': [], 'f1score': [], 'precision': [], 'recall': []}

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
        self.label_list += label.cpu().numpy().tolist()

    def calc_metrics(self, epoch):
        """calculates metrics, saves to tensorboard log & flush prediction list"""
        self.metrics_dict = self.get_metrics_dict(self.label_list, self.pred_list)
        self.log_writer.log({'Accuracy': self.metrics_dict['accuracy']}, step=epoch)
        self.log_writer.log({'f1score': self.metrics_dict['f1score']}, step=epoch)
        self.log_writer.log({'precision': self.metrics_dict['precision']}, step=epoch)
        self.log_writer.log({'recall': self.metrics_dict['recall']}, step=epoch)
        self.pred_list = []  # flush
        self.label_list = []

    @staticmethod
    def get_metrics_dict(label_list, pred_list):
        """calculate the metrics comparing the predictions to the ground-truth labels, and return them in dict format"""
        accuracy = metrics.accuracy_score(label_list, pred_list)  # calculate accuracy using sklearn.metrics
        f1score = metrics.f1_score(label_list, pred_list)
        precision = metrics.precision_score(label_list, pred_list)
        recall = metrics.recall_score(label_list, pred_list)
        metrics_dict = {'accuracy': accuracy, 'f1score': f1score, 'precision': precision, 'recall': recall}
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