import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from Code.logger import LoggerAbs
from pathlib import Path
import wandb

class TrainerAbs:
    def __init__(self, model: nn.Module, criterion: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 train_loader: DataLoader, 
                 train_as_val_loader: DataLoader,
                 val_loader: DataLoader, 
                 logger: LoggerAbs,
                 n_epochs: int, checkpoint_path: Path,
                 label_names: list[str],
                 sr: int = 16000,
                 seq_length: float = 1.0,
                 log_interval: int = 50,
                 verbose=True, train_as_val_interval: int = 20, 
                 scheduler=None,
                 stopper=None,
                 checkpoint=None, load_optimizer_state=False,
                 debug=False, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.train_as_val_loader = train_as_val_loader
        self.val_loader = val_loader
        self.logger = logger
        self.log_interval = log_interval
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self.label_names = label_names
        self.train_as_val_interval = train_as_val_interval
        self.debug = debug
        self.device = device
        # self.early_stopper = stopper

        self.sr = sr
        self.seq_length = seq_length

        self.trained_batches = 0

        if checkpoint:
            self._load_checkpoint(checkpoint_path=checkpoint, load_optimizer_state=load_optimizer_state)

        self.epochs_trained = 0

    def train(self):
        raise NotImplementedError

    def train_one_epoch(self, epoch):
        raise NotImplementedError
    
    def validate(self, epoch, train_as_val=False):
        raise NotImplementedError
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        # if isinstance(self.logger.log_writer, wandb):#self.logger.is_wandb:
        #     wandb_experiment_id = self.logger.run.id
        # else:
        #     wandb_experiment_id = None
        wandb_experiment_id = 42
        state_dict = {"optimizer": self.optimizer.state_dict(),
                      "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                      "epochs": self.epochs_trained,
                      "model": self.model.state_dict(),
                      "wandb_experiment_id": wandb_experiment_id,
                    #   "args": app.args # TODO: make it work
                      }

        torch.save(state_dict, self.checkpoint_path / checkpoint_name)
    
    def _load_checkpoint(self, checkpoint_path: str, load_optimizer_state: bool):
        """Load checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
        """
        if checkpoint_path is None:
            return
        print('Loading checkpoint')
        state_dict = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        self.model.load_state_dict(state_dict["model"])
        if load_optimizer_state:
            self.epochs_trained = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state_dict["scheduler"])

class Trainer1D(TrainerAbs):
    def train(self):
        pass

    def train_one_epoch(self, epoch):
        pass
    
    def validate(self, epoch, train_as_val=False):
        pass
    
class Trainer2D(TrainerAbs):
    def train(self) -> None:
        best_loss = float('inf')
        best_macro_f1 = 0.0
        best_macro_iou = float('inf')
        for epoch in tqdm(range(self.n_epochs), desc="Running epochs", leave=True, disable=not self.verbose):
            self.logger.reset_losses()
            
            self.train_one_epoch(epoch)
            self.validate(epoch)

            if epoch % self.train_as_val_interval == 0:
                self.validate(epoch, train_as_val=True)

            # save checkpoint if best loss:
            loss = self.logger.loss_meter_val['loss'].summarize_epoch()
            macro_f1 = self.logger.metrics_dict['global']['call_f1_macro']
            macro_iou = self.logger.metrics_dict['global']['call_avg_iou_macro']

            if loss < best_loss:
                best_loss = loss
                self._save_checkpoint("best.pth")
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                self._save_checkpoint("best_macro_f1.pth")
            if macro_iou < best_macro_iou:
                best_macro_iou = macro_iou
                self._save_checkpoint("best_macro_iou.pth")

            self._save_checkpoint("last.pth")

            self.epochs_trained += 1

            # self.early_stopper.update(val_loss, model)
            # if self.early_stopper.early_stop:
            #     break

            if self.debug and epoch >= 2:
                print('Debug mode: stopping after 3 epochs')
                break

    def _get_frac_epoch(self, epoch, multiplier=1000) -> int:
        """Return a fractional epoch number, because that some logger turn the step into int, the fraction is multiplies by multiplier"""
        return int(epoch + self.trained_batches / len(self.train_loader) * multiplier)
    
    def train_one_epoch(self, epoch) -> None:
        self.model.train()
        for batch in tqdm(self.train_loader, desc=f"Training epoch {epoch}", leave=False, disable=not self.verbose):
            features, label = batch
            features, label = features.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(features)
            loss = self.criterion(pred, label)
            loss.backward()
            self.optimizer.step()

            self.logger.update_losses(loss.detach().cpu(), flag='train')
            self.logger.update_predictions(pred.detach().cpu(), label.detach().cpu())

            self.trained_batches += 1
            if self.trained_batches % self.log_interval == 0:
                frac_epoch = self._get_frac_epoch(epoch)
                self.logger.calc_metrics(frac_epoch, mode='train', label_type='multilabel', label_names=self.label_names)
                self.logger.log(frac_epoch, flag='train')
            
        self.logger.calc_metrics(self._get_frac_epoch(epoch), mode='train', label_type='multilabel', label_names=self.label_names)
        self.logger.log(self._get_frac_epoch(epoch), flag='train')

        if self.scheduler is not None:
            self.scheduler.step()

    def upload_artifacts(self, data_loader, flag, frac_epoch):
        # choose 5 random samples and log their spectrograms with pred and true labels:
        rand_indices = np.random.choice(len(data_loader), size=min(5, len(data_loader)), replace=False)
        features_list = []
        target_list = []
        pred_list = []
        for i in rand_indices:
            features, label = data_loader.dataset[i]
            features, label = features.unsqueeze(0).to(self.device), label.unsqueeze(0).to(self.device)
            pred = self.model(features)
            features_list.append(features.cpu())
            target_list.append(label.cpu())
            pred_list.append(pred.cpu())
        self.logger.upload_artifacts(
            spectrograms=torch.stack(features_list),
            pred_labels=torch.stack(pred_list),
            target_labels=torch.stack(target_list),
            step=frac_epoch,
            flag=flag,
            seq_length=self.seq_length,
            sample_rate=self.sr,
            save_dir=self.checkpoint_path / f"artifacts_{flag}"
        )

    def validate(self, epoch, train_as_val=False) -> None:
        with torch.no_grad():
            self.model.eval()

            if train_as_val:
                data_loader = self.train_as_val_loader
                flag = 'train_as_val'
            else: # Validation
                data_loader = self.val_loader
                flag = 'val'

            for batch in tqdm(data_loader, desc=f"Validating epoch {epoch} on {flag} set", leave=False, disable=not self.verbose):
                features, label = batch
                features, label = features.to(self.device), label.to(self.device)

                pred = self.model(features)
                loss = self.criterion(pred, label)

                # log:
                self.logger.update_losses(loss.detach().cpu(), flag=flag)
                self.logger.update_predictions(pred.detach().cpu(), label.detach().cpu())
            
            frac_epoch = self._get_frac_epoch(epoch)
            self.logger.calc_metrics(frac_epoch, mode=flag, label_type='multilabel', label_names=self.label_names)
            self.logger.log(frac_epoch, flag=flag)

            self.upload_artifacts(data_loader, flag=flag, frac_epoch=frac_epoch)