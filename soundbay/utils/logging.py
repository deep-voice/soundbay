import wandb
from sklearn import metrics
from unittest.mock import Mock
import collections
import torch
import numpy as np
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
from typing import Union, List, Optional

from soundbay.utils.metrics import MetricsCalculator

matplotlib.rc('figure', max_open_warning = 0)

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections


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
                 log_writer=Mock(),
                 debug_mode=False,
                 artifacts_upload_limit=64
                 ):
        """
        __init__ initializes the logger and all the associated arrays and variables
        Input:
            log_writer: such as wandb, tensorboard etc.
        """
        self.log_writer = log_writer
        self.loss_meter_train, self.loss_meter_val, self.loss_meter_train_as_val = {}, {}, {}
        self.loss_meter_keys = ['loss']
        self.init_losses_meter()
        self.pred_list = []
        self.pred_proba_list = []
        self.label_list = []
        self.upload_artifacts_limit = artifacts_upload_limit
        self.metrics_dict = {}
        self.debug_mode = debug_mode

    def log(self, log_num: int, flag: str):
        """logging losses using writer"""
        for key in self.loss_meter_keys:
            if flag == 'train':
                self.log_writer.log({f"Losses/{key}_train":
                                     self.loss_meter_train[key].summarize_epoch()}, step=log_num)
            elif flag == 'val':
                self.log_writer.log({f"Losses/{key}_val":
                                     self.loss_meter_val[key].summarize_epoch()}, step=log_num)
            elif flag == 'train_as_val':
                self.log_writer.log({f"Losses/{key}_train_as_val": self.loss_meter_train_as_val[key].summarize_epoch()},
                                    step=log_num)

    def init_losses_meter(self):
        for key in self.loss_meter_keys:
            self.loss_meter_train[key] = LossMeter(key)
            self.loss_meter_val[key] = LossMeter(key)
            self.loss_meter_train_as_val[key] = LossMeter(key)

    def reset_losses(self):
        for key in self.loss_meter_keys:
            self.loss_meter_train[key].reset()
            self.loss_meter_val[key].reset()
            self.loss_meter_train_as_val[key].reset()


    def update_losses(self, loss, flag):
        losses = [loss]
        for key, current_loss in zip(self.loss_meter_keys, losses):
            if flag == 'train':
                self.loss_meter_train[key].add(current_loss.data.cpu().numpy().mean())
            elif flag == 'val':
                self.loss_meter_val[key].add(current_loss.data.cpu().numpy().mean())
            elif flag == 'train_as_val':
                self.loss_meter_train_as_val[key].add(current_loss.data.cpu().numpy().mean())

            else:
                raise ValueError('accept train or flag only!')

    def update_predictions(self, pred: np.ndarray, proba: np.ndarray, label: np.ndarray):
        """update prediction and label list from current batch/iteration"""
        self.pred_list += pred.tolist()  # add current batch prediction to full epoch pred list
        self.pred_proba_list.append(proba)
        self.label_list += label.tolist()

    def calc_metrics(self, epoch: int, label_type: str, mode: str = 'train', label_names: Optional[List[str]] = None):
        """calculates metrics, saves to tensorboard log & flush prediction list"""
        self.metrics_dict = MetricsCalculator(
            label_list=self.label_list,
            pred_list=self.pred_list,
            pred_proba_list=self.pred_proba_list,
            label_type=label_type).calc_all_metrics()

        pred_proba_array = np.concatenate(self.pred_proba_list)
        for metric, value in self.metrics_dict['global'].items():
            self.log_writer.log({f'Global Metrics {mode}/{metric}': value}, step=epoch)

        if label_names is None:
            label_names = ['Noise'] + [f'Call_{i}' for i in range(1, len(self.metrics_dict['calls']) + 1)]

        for class_id in self.metrics_dict['calls']:
            for metric in self.metrics_dict['calls'][class_id]:
                self.log_writer.log({f'Call Metrics {mode}/{metric}_{label_names[class_id]}':
                                    self.metrics_dict['calls'][class_id][metric]}, step=epoch)

        if (not self.debug_mode) and isinstance(self.label_list[0], int) == 1:
            self.log_writer.log(
                {f'{mode}_charts/ROC Curve': wandb.plot.roc_curve(self.label_list, pred_proba_array,
                                                                  labels=label_names)},
                step=epoch
            )
            self.log_writer.log(
                {f'{mode}_charts/PR Curve': wandb.plot.pr_curve(self.label_list, pred_proba_array, labels=label_names)},
                step=epoch
            )
            wandb.log({f'{mode}_charts/conf_mat': wandb.plot.confusion_matrix(probs=None, y_true=self.label_list,
                                                                              preds=self.pred_list,
                                                                              class_names=label_names)},
                                                                              step=epoch, commit=False)
        self.pred_list = []  # flush
        self.label_list = []
        self.pred_proba_list = []

    def upload_artifacts(self, audio: torch.Tensor, label: torch.Tensor, raw_wav: torch.Tensor, meta: dict, sample_rate: int=16000, flag: str='train', data_sample_rate: int = 16000):
        """upload algorithm artifacts to W&B during training session"""
        volume = 50
        matplotlib.use('Agg')
        idx = meta['idx'].detach().cpu().numpy()
        meta['begin_time'] = meta['begin_time'].detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        if audio.shape[0] > self.upload_artifacts_limit:
            audio = audio[:self.upload_artifacts_limit,...]
            label = label[:self.upload_artifacts_limit,...]
            raw_wav = raw_wav[:self.upload_artifacts_limit,...]
            idx = idx[:self.upload_artifacts_limit,...]

        # Original wavs batch

        artifact_wav = torch.squeeze(raw_wav).detach().cpu().numpy()
        artifact_wav = artifact_wav / np.expand_dims(np.abs(artifact_wav).max(axis=1) + 1e-8, 1) * 0.5  # gain -6dB
        list_of_wavs_objects = [wandb.Audio(data_or_path=wav, caption=f'{flag}_label{lab}_i{ind}_{round(b_t/data_sample_rate,2)}sec_{f_n}', sample_rate=sample_rate) for wav, ind, lab, b_t, f_n in zip(artifact_wav,idx, label, meta['begin_time'], meta['org_file'])]
        log_wavs = {f'First batch {flag} original wavs': list_of_wavs_objects}

        # Spectrograms batch
        if audio.dim() >= 4: # In case that spectrogram preprocessing was not applied the dimension is 3.
            artifact_spec = torch.squeeze(audio).detach().cpu().numpy()
            specs = []
            for artifact_id in range(artifact_spec.shape[0]):
                ax = plt.subplots(nrows=1, ncols=1)
                specs.append(librosa.display.specshow(artifact_spec[artifact_id,...], ax=ax[1]))
                plt.close('all')
                del ax
            list_of_specs_objects = [wandb.Image(data_or_path=spec, caption=f'{flag}_label{lab}_i{ind}_{round(b_t/data_sample_rate,2)}sec_{f_n}') for spec, ind, lab, b_t, f_n in zip(specs,idx, label, meta['begin_time'], meta['org_file'])]
            log_specs = {f'First batch {flag} augmented spectrogram\'s': list_of_specs_objects}
            # Upload spectrograms to W&B
            wandb.log(log_specs, commit=False)

        # Upload WAVs to W&B
        wandb.log(log_wavs, commit=False)

def get_experiment_name(args) -> Union[str, None]:
    if args.experiment.name:
        experiment_name = args.experiment.name
    elif args.experiment.run_id and args.experiment.group_name:
        experiment_name = f'{args.experiment.group_name}-{args.experiment.run_id}'
    elif args.experiment.group_name:
        experiment_name = f'{args.experiment.group_name}-{wandb.util.generate_id()}'
    else:
        experiment_name = None
    return experiment_name


def flatten(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collectionsAbc.MutableMapping):
            #items.extend(flatten(v, new_key, sep=sep).items())
            items.update(flatten(v, new_key, sep=sep))
        else:
            items.update({new_key: v})
    return dict(items)
