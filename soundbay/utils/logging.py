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
        self.loss_meter_train, self.loss_meter_val = {}, {}
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

    def calc_metrics(self, epoch: int, mode: str = 'train', label_names: Optional[List[str]] = None):
        """calculates metrics, saves to tensorboard log & flush prediction list"""
        pred_proba_array = np.concatenate(self.pred_proba_list)
        self.metrics_dict = self.get_metrics_dict(self.label_list, self.pred_list, pred_proba_array)
        for metric, value in self.metrics_dict['global'].items():
            self.log_writer.log({f'Global Metrics {mode}/{metric}': value}, step=epoch)

        if label_names is None:
            label_names = ['Noise'] + [f'Call_{i}' for i in range(1, len(self.metrics_dict['calls']) + 1)]

        for class_id in self.metrics_dict['calls']:
            for metric in self.metrics_dict['calls'][class_id]:
                self.log_writer.log({f'Call Metrics {mode}/{metric}_{label_names[class_id]}':
                                    self.metrics_dict['calls'][class_id][metric]}, step=epoch)

        if not self.debug_mode:
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

    def upload_artifacts(self, audio: torch.Tensor, label: torch.Tensor, raw_wav: torch.Tensor, idx: torch.Tensor, sample_rate: int=16000, flag: str='train'):
        """upload algorithm artifacts to W&B during training session"""
        volume = 50
        matplotlib.use('Agg')
        idx = idx.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        if audio.shape[0] > self.upload_artifacts_limit:
            audio = audio[:self.upload_artifacts_limit,...]
            label = label[:self.upload_artifacts_limit,...]
            raw_wav = raw_wav[:self.upload_artifacts_limit,...]
            idx = idx[:self.upload_artifacts_limit,...]

        # Original wavs batch

        artifact_wav = torch.squeeze(raw_wav).detach().cpu().numpy()
        artifact_wav = artifact_wav / np.expand_dims(np.abs(artifact_wav).max(axis=1) + 1e-8, 1) * 0.5  # gain -6dB
        list_of_wavs_objects = [wandb.Audio(data_or_path=wav, caption=f'label_{lab}_{ind}_train', sample_rate=sample_rate) for wav, ind, lab in zip(artifact_wav,idx, label)]

        # Spectrograms batch
        artifact_spec = torch.squeeze(audio).detach().cpu().numpy()
        axes = [plt.subplots(nrows=1, ncols=1) for _ in range(artifact_spec.shape[0])]
        specs = [librosa.display.specshow(artifact_spec[x,...], ax=axes[x][1]) for x in range(artifact_spec.shape[0])]
        list_of_specs_objects = [wandb.Image(data_or_path=spec, caption=f'label_{lab}_{ind}_train') for spec, ind, lab in zip(specs,idx, label)]
        log_wavs = {f'First batch {flag} original wavs': list_of_wavs_objects}
        log_specs = {f'First batch {flag} augmented spectrograms': list_of_specs_objects}

        # Upload to W&B
        wandb.log(log_wavs, commit=False)
        wandb.log(log_specs, commit=False)

        # Clear figures
        plt.figure().clear()
        return

    @staticmethod
    def get_metrics_dict(label_list: Union[list, np.ndarray], pred_list: Union[list, np.ndarray],
                         pred_proba_array: np.ndarray):
        """calculate the metrics comparing the predictions to the ground-truth labels, and return them in dict format"""

        label_list = np.array(label_list)
        pred_list = np.array(pred_list)
        pred_proba_array = np.array(pred_proba_array)

        metrics_dict = {
            'global': {'accuracy': metrics.accuracy_score(label_list, pred_list),
                       'call_average_precision_macro': np.nanmean([metrics.average_precision_score(
                           label_list == i, pred_proba_array[:, i]) for i in range(1, pred_proba_array.shape[1])]),
                       'bg_average_precision': metrics.average_precision_score(label_list == 0, pred_proba_array[:, 0]),
                       'call_f1_macro': metrics.f1_score(label_list, pred_list, average='macro',
                                                         labels=list(range(1, pred_proba_array.shape[1]))),
                       'bg_f1': metrics.f1_score(label_list == 0, pred_list == 0),
                       'bg_precision': metrics.precision_score(label_list == 0, pred_list == 0),
                       'bg_recall': metrics.recall_score(label_list == 0, pred_list == 0),
                       },
            'calls': {}
        }

        def nan_auc(y_true, y_pred):
            try:
                return metrics.roc_auc_score(y_true, y_pred)
            except ValueError:
                return np.nan

        metrics_dict['global']['bg_auc'] = nan_auc(label_list == 0, pred_proba_array[:, 0])

        # Equivalent of 'macro' 'ovr' auc for only the positive classes.
        # nan auc are not counted towards the mean.
        pos_auc_list = [nan_auc(label_list == i, pred_proba_array[:, i]) for i in range(1, pred_proba_array.shape[1])]
        metrics_dict['global']['call_auc_macro'] = np.nanmean(pos_auc_list)

        for class_id in range(1, pred_proba_array.shape[1]):
            metrics_dict['calls'][class_id] = {
                'precision': metrics.precision_score(label_list == class_id, pred_list == class_id),
                'recall': metrics.recall_score(label_list == class_id, pred_list == class_id),
                'f1': metrics.f1_score(label_list == class_id, pred_list == class_id),
            }

        return metrics_dict


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
