from typing import Union, Generator, Tuple, List
import torch
import torch.utils.data
from tqdm import tqdm
from pathlib import Path
from soundbay.utils.app import app
from soundbay.utils.logging import Logger
from soundbay.utils.post_process import post_process_predictions
import wandb


class Trainer:
    """
    Trainer class -
    concludes the variables related to training a model
    Args:
        train_dataloader(Generator) - defined in main.py
        val_dataloader(Generator) - defined in main.py
        optimizer(torch.optimizer) -
        criterion(torch.o)
        epochs: int,
        logger: Logger,
        output_path: Union[str, Path],
        device: Union[torch.device, None] = torch.device("cpu"),
        scheduler=None,
        checkpoint: str = None,
        debug: bool = False):

    """
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataloader: Generator[Tuple[torch.tensor, torch.tensor], None, None],
                 val_dataloader: Generator[Tuple[torch.tensor, torch.tensor], None, None],
                 train_as_val_dataloader: Generator[Tuple[torch.tensor, torch.tensor], None, None],
                 optimizer: torch.optim,
                 criterion,
                 label_type: str,
                 epochs: int,
                 logger: Logger,
                 output_path: Union[str, Path],
                 device: Union[torch.device, None] = torch.device("cpu"),
                 scheduler=None,
                 checkpoint: str = None,
                 load_optimizer_state: bool = False,
                 label_names: List[str] = None,
                 debug: bool = False,
                 train_as_val_interval: int = 20,
                 proba_threshold: Union[float,None] = None):

        # set parameters for stft loss
        self.model = model
        self.epochs = epochs
        self.logger = logger
        self.criterion = criterion
        self.device = device
        self.epochs_trained = 0
        self.debug = debug
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.verbose = True
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_as_val_dataloader = train_as_val_dataloader
        # every X epochs we evaluate the train set as if it was a validation set
        self.train_as_val_interval = train_as_val_interval
        self.output_path = output_path
        self.label_names = list(label_names) if label_names else None
        self.label_type = label_type
        self.proba_threshold = proba_threshold

        # load checkpoint
        if checkpoint:
            self._load_checkpoint(checkpoint_path=checkpoint, load_optimizer_state=load_optimizer_state)

    def train(self):
        best_loss = float('inf')
        best_macro_f1 = 0
        # Run training script
        iterator = tqdm(range(self.epochs_trained, self.epochs),
                        desc='Running Epochs', leave=True, disable=not self.verbose)
        for epoch in iterator:
            self.logger.reset_losses()
            self.train_epoch(epoch)
            self.epochs_trained += 1
            self.eval_epoch(epoch, 'val')
            if epoch % self.train_as_val_interval == 0:
                self.eval_epoch(epoch, 'train_as_val')
            # save checkpoint
            loss = self.logger.loss_meter_val['loss'].summarize_epoch()
            macro_f1 = self.logger.metrics_dict['global']['call_f1_macro']
            if loss < best_loss:
                best_loss = loss
                self._save_checkpoint("best.pth")
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                self._save_checkpoint("best_macro_f1.pth")
            self._save_checkpoint("last.pth")
            if self.verbose:  # show batch metrics in progress bar
                s = 'epoch: ' + str(epoch) + ', ' + str(self.logger.metrics_dict)
                iterator.set_postfix_str(s)
            if self.debug and epoch > 2:
                break


    def train_epoch(self, epoch):
        self.model.train()
        for it, batch in tqdm(enumerate(self.train_dataloader), desc='train'):
            if it == 3 and self.debug:
                break

            self.model.zero_grad()
            audio, label, raw_wav, meta = batch
            audio, label = audio.to(self.device), label.to(self.device)

            if (it == 0) and (not self.debug) and ((epoch % 5) == 0):
                self.logger.upload_artifacts(audio, label, raw_wav, meta, sample_rate=self.train_dataloader.dataset.sample_rate,
                                             flag='train', data_sample_rate=self.train_dataloader.dataset.data_sample_rate)

            # estimate and calc losses
            estimated_label = self.model(audio)
            if self.label_type == 'multi_label':
                label = label.type_as(estimated_label)
            loss = self.criterion(estimated_label, label)
            loss.backward()
            self.optimizer.step()

            # process the logit predictions:
            predicted_proba, predicted_label = post_process_predictions(estimated_label.data, self.label_type, self.proba_threshold)

            # update losses and log batch

            self.logger.update_losses(loss.detach(), flag='train')
            self.logger.update_predictions(predicted_label, predicted_proba, label.cpu().numpy())

        # logging
        if not app.args.experiment.debug:
            self.logger.calc_metrics(epoch, self.label_type,'train', self.label_names)

        self.logger.log(epoch, 'train')
        if self.scheduler is not None:
            self.scheduler.step()

    def eval_epoch(self, epoch: int, datatset_name: str = None):

        with torch.no_grad():
            self.model.eval()

            # set the desired dataloader for evaluation
            if datatset_name == 'val':
                dataloader = self.val_dataloader
            elif datatset_name == "train_as_val":  # data from the train set, processed as validation set
                dataloader = self.train_as_val_dataloader

            for it, batch in tqdm(enumerate(dataloader), desc=datatset_name):
                if it == 3 and self.debug:
                    break
                audio, label, raw_wav, meta = batch
                audio, label = audio.to(self.device), label.to(self.device)
                if (it == 0) and (not self.debug) and ((epoch % 5) == 0):
                    self.logger.upload_artifacts(audio, label, raw_wav, meta, sample_rate=self.train_dataloader.dataset.sample_rate,
                                                 flag=datatset_name, data_sample_rate=self.train_dataloader.dataset.data_sample_rate)

                # estimate and calc losses
                estimated_label = self.model(audio)
                if self.label_type == 'multi_label':
                    label = label.type_as(estimated_label)
                loss = self.criterion(estimated_label, label)

                # process the logit predictions:
                predicted_proba, predicted_label = post_process_predictions(estimated_label.data, self.label_type, self.proba_threshold)

                # update losses
                self.logger.update_losses(loss.detach(), flag=datatset_name)
                self.logger.update_predictions(predicted_label, predicted_proba, label.cpu().numpy())

            # logging
            if not app.args.experiment.debug:
                self.logger.calc_metrics(epoch, self.label_type, datatset_name, self.label_names)
            self.logger.log(epoch, datatset_name)


    def _save_checkpoint(self, checkpoint_path: Union[str, None]):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        if checkpoint_path is None or app.args.experiment.debug:
            return
        state_dict = {"optimizer": self.optimizer.state_dict(),
                      "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                      "epochs": self.epochs_trained,
                      "model": self.model.state_dict(),
                      "wandb_experiment_id": wandb.run.id if not app.args.experiment.debug else None,
                      "args": app.args
                      }

        torch.save(state_dict, self.output_path / checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: Union[str, None], load_optimizer_state: bool):
        """Load checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
        """
        if checkpoint_path is None:
            return
        print('Loading checkpoint')
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict["model"])
        if load_optimizer_state:
            self.epochs_trained = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state_dict["scheduler"])
