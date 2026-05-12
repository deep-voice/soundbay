from pathlib import Path
import random
import string
import torch

from torch.utils.data import DataLoader

from Code.Config import load_config, save_config
from Code.Trainers import Trainer1D, Trainer2D
from Code.Utils import set_seed, get_dataset, get_model, get_criterion, get_optimizer, get_scheduler, get_sampler
from Code.logger import Mock, TBWrapper, Detection1DLogger, Detection2DLogger
from Code.Losses import DetectionLoss
import wandb

def main(config_path: Path, verbose: bool = True):
    config = load_config(config_path)

    if config.experiment.debug and config.experiment.logger.name == "wandb":
        log_writer = Mock()
        log_writer.run.id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    elif config.experiment.logger.name == "wandb":
        log_writer = wandb
    elif config.experiment.logger.name == "tensorboard":
        log_writer = TBWrapper()
    else:
        raise ValueError(f"Unknown logger: {config.experiment.logger}")

    log_writer.init(
        project = config.experiment.project_name,
        name = config.experiment.run_name,
        group=config.experiment.group_name,
        dir=config.experiment.logger.log_dir,
        id = config.experiment.run_id,
        resume = config.experiment.resume,
    )

    # check for device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("--" * 10, f"Using device: {device}", "--" * 10, flush=True)

    # make sure checkpoint directory exists
    checkpoint_dir = Path(config.experiment.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # save the config file to the checkpoint directory for reproducibility
    save_config(config, checkpoint_dir / "config.yaml")

    # set seed for reproducibility
    set_seed(config.experiment.seed)

    # set datasets:
    train_dataset = get_dataset(config.data.train_dataset, 
                                config.data, 
                                n_classes=config.model.num_classes)
    train_as_val_dataset = get_dataset(config.data.train_dataset, 
                                       config.data, n_classes=config.model.num_classes)
    val_dataset = get_dataset(config.data.val_dataset, 
                              config.data, n_classes=config.model.num_classes)

    sampler = get_sampler(config.data.equalize_data, train_dataset)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=not config.data.equalize_data,
        num_workers=config.data.num_workers,
        sampler=sampler,
        pin_memory=True
    )

    train_as_val_dataloader = DataLoader(
        train_as_val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    model = get_model(config.model, config.data, n_classes=config.model.num_classes, pooling_size=config.model.pooling_size)
    print('*** model has been loaded successfully ***')
    print(f'number of trainable params: {sum([p.numel() for p in model.parameters() if p.requires_grad]):,}')
    model.to(device)

    log_writer.watch(model, log_freq=config.experiment.log_interval)

    criterion = get_criterion(config.model.criterion)
    optimizer = get_optimizer(model.parameters(), config.learning.optimizer)
    scheduler = get_scheduler(optimizer, config.learning.scheduler)

    if config.data.label_type == "1D":
        logger = Detection1DLogger(
            log_writer=log_writer,
            artifacts_upload_limit=config.experiment.logger.artifacts_upload_limit,
        )
        m_trainer = Trainer1D(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_dataloader,
            train_as_val_loader=train_as_val_dataloader,
            val_loader=val_dataloader,
            logger=logger,
            n_epochs=config.learning.epochs,
            checkpoint_path=checkpoint_dir,
            label_names=config.data.label_names,
            verbose=verbose,
            train_as_val_interval=config.experiment.train_as_val_interval,
            # checkpoint=config.experiment.checkpoint,
            # load_optimizer_state=config.experiment.load_optimizer_state,
        )
    elif config.data.label_type == "2D":
        logger = Detection2DLogger(
            log_writer=log_writer,
            artifacts_upload_limit=config.experiment.logger.artifacts_upload_limit,
            confidence_threshold=config.experiment.logger.confidence_threshold,
            iou_threshold=config.experiment.logger.iou_threshold,
            num_classes=config.model.num_classes,
        )

        m_trainer = Trainer2D(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_dataloader,
            train_as_val_loader=train_as_val_dataloader,
            val_loader=val_dataloader,
            logger=logger,
            n_epochs=config.learning.epochs,
            checkpoint_path=checkpoint_dir,
            label_names=config.data.label_names,
            verbose=verbose,
            train_as_val_interval=config.experiment.train_as_val_interval,
            scheduler=scheduler,
            seq_length=config.data.seq_length,
            sr=config.data.wanted_sample_rate
        )
    else:
        raise ValueError(f"Unknown label type: {config.data.label_type}")

    # if config.model.freeze_for_fine_tuning:
    #     model.freeze_layers()

    m_trainer.train()


if __name__ == "__main__":
    config_dir = Path("Config")
    config_paths = [config_dir / "train_20_sec.yaml"]
    for config_path in config_paths:
        print(f"Running experiment with config: {config_path}")
        main(config_path)