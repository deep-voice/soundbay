import os

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
from pathlib import Path

from Code.Metrics import MetricsCalculator, MetricsCalculator2Detection
from torch.utils.tensorboard import SummaryWriter

############################################# TB WRAPPER #############################################

# A simple helper for images to match wandb.Image(tensor)
class Image:
    def __init__(self, data, caption=None):
        self.data = data
        self.caption = caption

class TBWrapper:
    """ a wrapper for tensorboard SummaryWriter to have a unified interface with W&B logger """
    def __init__(self):
        self.writer = None
        self.step_key = "global_step"
        self.watched_model = None
        self.log_freq = 100

    def init(self, project=None, name=None, config=None, dir="./runs", group=None, id=None, resume=False):
        """Mimics wandb.init()"""
        dir = Path(dir)
        # log_path = os.path.join(dir, project or "default", name or "run")
        log_path = dir / (project or "default") / (name or "run")
        log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_path)
        
        # In TB, we log config as Text or Hyperparameters
        if config:
            # Short-cut to log config as a string/table
            config_str = "\n".join([f"{k}: {v}" for k, v in config.items()])
            self.writer.add_text("config", config_str, 0)
        
        print(f"✅ TensorBoard initialized at: {log_path}")
        return self
    
    def watch(self, model, log_freq=100):
        """Mimics wandb.watch()"""
        self.watched_model = model
        self.log_freq = log_freq

    def log(self, data, step):
        """
        Mimics wandb.log()
        Expects a dict: {"loss": 0.5, "accuracy": 0.9}
        """
        if self.writer is None:
            raise RuntimeError("Run .init() before logging!")

        # If step is not provided, we need to track it or TB won't plot correctly
        # Usually, users pass step explicitly in custom loops
        for key, value in data.items():
            if isinstance(value, (int, float, torch.Tensor)) and not isinstance(value, torch.Tensor) or value.ndim == 0:
                self.writer.add_scalar(key, value, step)
            elif isinstance(value, Image): # Handle custom Image wrapper
                self.writer.add_image(key, value.data, step)

        # mimic watch:
        if self.watched_model and step % self.log_freq == 0:
            for name, param in self.watched_model.named_parameters():
                # Log weights:
                self.writer.add_histogram(f"{name}_weights", param.data.cpu().numpy(), step)
                # Log gradients if exist:
                if param.grad is not None:
                    self.writer.add_histogram(f"{name}_grads", param.grad.data.cpu().numpy(), step)
                
    def finish(self):
        """Mimics wandb.finish()"""
        if self.writer:
            self.writer.close()


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

############################################### LOGGERS #############################################

class LoggerAbs:
    def __init__(self, log_writer=Mock(), debug_mode=False, artifacts_upload_limit=64,
                 loss_meter_class=LossMeter):
        self.log_writer = log_writer
        self.loss_meter_train, self.loss_meter_val, self.loss_meter_train_as_val = {}, {}, {}
        self.loss_meter_keys = ['loss']
        self.loss_meter_class = loss_meter_class
        self.init_losses_meter()
        self.pred_list = []
        self.pred_proba_list = []
        self.label_list = []
        self.upload_artifacts_limit = artifacts_upload_limit
        self.metrics_dict = {}
        self.debug_mode = debug_mode

    def log(self, log_num: int | float, flag: str):
        """logging losses using writer"""
        # print(f"Logging {flag} losses at step {log_num}: ", end="")
        for key in self.loss_meter_keys:
            self.log_writer.log({f"Losses/{key}_{flag}":
                        self.loss_meter_train[key].summarize_epoch()}, step=log_num)

    def init_losses_meter(self):
        for key in self.loss_meter_keys:
            self.loss_meter_train[key] = self.loss_meter_class(key)
            self.loss_meter_val[key] = self.loss_meter_class(key)
            self.loss_meter_train_as_val[key] = self.loss_meter_class(key)

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

    def update_predictions(self, pred: np.ndarray, label: np.ndarray):
        raise NotImplementedError("This method should be implemented in the child class")

    def calc_metrics(self, epoch: int, label_type: str, mode: str = 'train', label_names: Optional[List[str]] = None):
        raise NotImplementedError("This method should be implemented in the child class")

    def upload_artifacts(self, spectrograms: torch.Tensor, pred_labels: torch.Tensor, target_labels: torch.Tensor, step: int, save_dir: Path, flag: str='train',
                         seq_length: float = 1.0, sample_rate: int = 16000):
        raise NotImplementedError("This method should be implemented in the child class")

class Detection1DLogger(LoggerAbs):
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
        super().__init__(log_writer, debug_mode, artifacts_upload_limit)

    def update_predictions(self, pred: np.ndarray, label: np.ndarray):
        """update prediction and label list from current batch/iteration"""
        self.pred_list += pred.tolist()  # add current batch prediction to full epoch pred list
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

        if (not self.debug_mode) and label_type == 'single_label':
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

    def upload_artifacts(self, spectrograms: torch.Tensor, pred_labels: torch.Tensor, target_labels: torch.Tensor, step: int, save_dir: Path, flag: str='train'):
        raise NotImplementedError("This method should be implemented in the child class")
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

import torchaudio.transforms as T
from torch import nn

class Detection2DLogger(LoggerAbs):
    def __init__(self, 
                 log_writer=Mock(),
                 debug_mode=False,
                 artifacts_upload_limit=64,
                 loss_meter_class=LossMeter,
                 iou_threshold: float = 0.5,
                 confidence_threshold: float = 0.5,
                 num_classes: int = 10,
                 n_fft: int = 256,
                 hop_length: int = 64,
                 stype: str = 'power',
                 ):
        super().__init__(log_writer, debug_mode, artifacts_upload_limit, loss_meter_class)
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes

        power = 2.0 if stype == 'power' else 1.0
        self.spectrograms_creator = nn.Sequential(
            T.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=None, power=power),
            T.AmplitudeToDB(stype=stype, top_db=80)
        )

    def update_predictions(self, pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor):
        """update prediction and label list from current batch/iteration"""
        self.pred_list += [pred_bboxes]  # add current batch prediction to full epoch pred list
        self.label_list += [target_bboxes]  # add current batch labels to full epoch label list

    def calc_metrics(self, epoch: int, label_type: str, mode: str = 'train', label_names: Optional[List[str]] = None):
        if self.pred_list == [] or self.label_list == []:
            print("No predictions or labels to calculate metrics.")
            return
        self.metrics_dict = MetricsCalculator2Detection(
            target_boxes=torch.cat(self.label_list, dim=0),
            pred_boxes=torch.cat(self.pred_list, dim=0),
            iou_threshold=self.iou_threshold,
            confidence_threshold=self.confidence_threshold,
            n_classes=self.num_classes).calc_all_metrics()
        
        # log global metrics:
        for metric, value in self.metrics_dict['global'].items():
            self.log_writer.log({f'Global Metrics {mode}/{metric}': value}, step=epoch)

        # log per-class metrics:
        for class_id in self.metrics_dict['calls']:
            for metric in self.metrics_dict['calls'][class_id]:
                self.log_writer.log({f'Call Metrics {mode}/{metric}_class{class_id}':
                                    self.metrics_dict['calls'][class_id][metric]}, step=epoch)
        
        # flush:
        self.pred_list = []
        self.label_list = []

    def upload_artifacts(self, spectrograms: torch.Tensor, pred_labels: torch.Tensor, target_labels: torch.Tensor, step: int, save_dir: Path, flag: str='train',
                         seq_length: float = 1.0, sample_rate: int = 16000):
        """ upload spectrogram image of the audio sample with predicted and true bounding boxes drawn"""
        # This is a placeholder implementation. You can expand this to draw bounding boxes on the spectrograms and upload them as images.
        metrics_dict = MetricsCalculator2Detection(
            target_boxes=target_labels.detach().cpu(),
            pred_boxes=pred_labels.detach().cpu(),
            iou_threshold=self.iou_threshold,
            confidence_threshold=self.confidence_threshold,
            n_classes=self.num_classes)
        
        pred_labels = metrics_dict.pred_boxes
        
        for i in range(min(self.upload_artifacts_limit, spectrograms.shape[0])):
            spec = spectrograms[i].detach().cpu().numpy()
            # if start with channel dimension, remove it:
            if spec.shape[0] == 1:
                spec = spec[0][0]

            pred_box = pred_labels[i]
            pred_mask = metrics_dict._get_box_mask(pred_box)
            pred_boxes = pred_box[pred_mask].numpy()
            pred_boxes = pred_boxes[:, :4]  # Assuming the first 4 values are [x, y, w, h]

            target_box = target_labels[i]
            target_mask = metrics_dict._get_box_mask(target_box)
            target_boxes = target_box[target_mask].numpy()
            target_boxes = target_boxes[:, :4]  # Assuming the first 4 values are [x, y, w, h]

            # drow pred_boxes and target_boxes on the spec image (this is a placeholder, you can implement the actual drawing logic)
            plt.figure(figsize=(10, 4))
            # librosa.display.specshow(spec, sr=sample_rate, x_axis='time', y_axis='linear')
            plt.imshow(spec, aspect='auto', origin='lower')
            spec_height, spec_width = spec.shape
            for box in pred_boxes:
                x, y, w, h = box
                x *= spec_width
                y *= spec_height
                w *= spec_width
                h *= spec_height

                rect = plt.Rectangle((x, y), w, h, edgecolor='k', facecolor='none', linewidth=2)
                plt.gca().add_patch(rect)
            for box in target_boxes:
                x, y, w, h = box
                x *= spec_width
                y *= spec_height
                w *= spec_width
                h *= spec_height
                rect = plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none', linewidth=2)
                plt.gca().add_patch(rect)
            plt.title(f'{flag} sample {i} - Black: Predicted, Red: Target')
            plt.axis('off')
            plt.tight_layout()
            # for debug show the image:
            if self.debug_mode:
                plt.show()

            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f'{flag}_step_{step}_sample_{i}.png')
            plt.close()

            # TODO: upload image to self.log_writer (e.g. W&B or TB) if needed, currently we save it to disk for simplicity
            # self.log_writer.log({f'{flag}_charts/Spectrogram_with_BBoxes_sample_{i}': wandb.Image(plt)}, step=step)
            # log the image to W&B:
            # self.log_writer.log({f'{flag}_charts/Spectrogram_with_BBoxes_sample_{i}': wandb.Image(plt)}, step=step)


if __name__ == "__main__":
    # Example usage of the Detection2DLogger
    logger = Detection2DLogger(log_writer=Mock(), debug_mode=False, artifacts_upload_limit=64,
                              iou_threshold=0.5, confidence_threshold=0.5, num_classes=3)
    
    # Simulated predictions and targets for a batch
    pred_boxes = torch.tensor([[[10, 100, 0.25, 15, 1, 0.9, 0.05, 0.8],
                                  [20, 200, 0.5, 25, 2, 0.8, 0.1, 0.6],
                                  [30, 300, 0.25, 35, 3, 0.7, 0.2, 2.0]],
                                 [[15, 150, 0.15, 45, 1, 0.85, 0.1, 0.05],
                                  [25, 250, 0.05, 15, 2, 0.75, 0.15, 0.1],
                                  [35, 350, 0.5, 25, 3, 0.65, 0.25, 0.1]]])
    print(pred_boxes.shape)
    target_boxes = torch.tensor([[[12, 120, 0.25, 15, 1, 5, 7, 1.0],
                                [22, 220, 0.25, 25, 2, 0.7, 0.2, 1.0],
                                [32, 320, 0.25, 35, 3, 0.6, 0.3, 0.1]],
                               [[14, 140, 0.15, 45, 1, 0.9, 0.05, 0.05],
                                [24, 240, 0.25, 15, 2, 0.85, 0.1, 0.05],
                                [34, 340, 0.5, 25, 3, 0.7, 110, 0.1]]])

    logger.update_predictions(pred_bboxes=pred_boxes, target_bboxes=target_boxes)
    logger.calc_metrics(epoch=1, label_type='multi_label', mode='train')
    print(logger.metrics_dict)

    logger.upload_artifacts(spectrograms=torch.randn(2, 1, 128, 128), 
                            pred_labels=pred_boxes, 
                            target_labels=target_boxes, 
                            sample_rate=1000,
                            seq_length=30.0,
                            step=1, 
                            flag='train')