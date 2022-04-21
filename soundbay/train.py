
"""
Main training loop
--------------------
This script main.py constitutes the main training loop.
main function is wrapped with hydra @main wrapper which contains all the configuration and variables needed
to run the main training loop (models, data paths,
augmentations, preprocessing etc..) - for more details about hydra package
configuration please refer to https://hydra.cc/

The configuration files are located in ./soundbay/conf folder and it's possible to overwrite specific arguments
using the command line when running main.py (e.g. "main.py experiment.debug=True")

* prior to running this script make sure to define the data paths, annotations and output accordingly
* make sure to install all the packages stated in the requirements.txt file prior to running this script

"""

import torch
from torch.utils.data import DataLoader
import wandb
from functools import partial
from pathlib import Path
import hydra
from hydra.utils import instantiate
import random
from unittest.mock import Mock
import os
import tempfile
from soundbay.utils.app import App
from soundbay.utils.logging import Logger, flatten, get_experiment_name
from soundbay.utils.checkpoint_utils import upload_experiment_to_s3
from soundbay.trainers import Trainer


def modeling(
    trainer,
    device,
    batch_size,
    num_workers,
    train_dataset_args,
    val_dataset_args,
    optimizer_args,
    scheduler_args,
    model_args,
    logger,
    num_unfrozen,
):
    """
    modeling function takes all the variables and parameters defined in the main script
    (either through hydra configuration files or overwritten in the command line
    , instantiates them and starts a training on the relevant model chosen

    input:
    trainer - a Trainer object class instance as defined in trainers.py
    device - device (cpu\ gpu)
    batch_size - int
    num_workers - number of workers
    train_dataset_args - train dataset arguments taken from the configuration files/ overwritten
    val_dataset_args - val dataset arguments taken from the configuration files/ overwritten
    optimizer_args - optimizer  arguments taken from the configuration files/ overwritten
    scheduler_args - scheduler arguments taken from the configuration files/ overwritten
    model_args - model arguments taken from the configuration files/ overwritten
    logger - logger arguments taken from the configuration files/ overwritten

    """
    # Set paths and create dataset
    train_dataset = instantiate(train_dataset_args)
    val_dataset = instantiate(val_dataset_args)

    # Define model and device for training
    model = instantiate(model_args)
    model.to(device)

    # Add model watch to WANDB
    logger.log_writer.watch(model)

    # Define dataloader for training and validation datasets as well as optimizers arguments
    train_dataloader = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
    val_dataloader = DataLoader(
            dataset=val_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

    optimizer = instantiate(optimizer_args, model.parameters())
    scheduler = instantiate(scheduler_args, optimizer)

    # Add the rest of the parameters to trainer instance
    _trainer = trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger
    )

    # Freeze layers if required (num_unfrozen!=0)
    if num_unfrozen:
        model.freeze_layers(num_unfrozen)

    # Commence training

    _trainer.train()

    return


# TODO check how to use hydra without path override
@hydra.main(config_name="runs/main", config_path="conf")
def main(args):

    # Set logger
    _logger = wandb if not args.experiment.debug else Mock()
    experiment_name = get_experiment_name(args)
    _logger.init(project="finding_willy", name=experiment_name, group=args.experiment.group_name,
                 id=args.experiment.run_id)

    # Set device
    if not torch.cuda.is_available():
        print('CPU!!!!!!!!!!!')
        device = torch.device("cpu")
    else:
        print('GPU!!!!!!!!!')
        device = torch.device("cuda")

    # Convert filepaths, convenient if you wish to use relative paths
    working_dirpath = Path(hydra.utils.get_original_cwd())
    output_dirpath = Path.cwd()
    os.chdir(working_dirpath)

    # Define checkpoint
    if args.checkpoint.path:
        checkpoint = working_dirpath / args.checkpoint.path
        assert checkpoint.exists(), 'Checkpoint does not exists!'
    else:
        checkpoint = None

    # Logging
    logger = Logger(_logger, debug_mode=args.experiment.debug, artifacts_upload_limit=args.experiment.artifacts_upload_limit)
    flattenArgs = flatten(args)
    logger.log_writer.config.update(flattenArgs)
    App.init(args)

    # Define criterion
    criterion = instantiate(args.model.criterion)

    # Seed script
    if args.experiment.manual_seed is None:
        args.experiment.manual_seed = random.randint(1, 10000)
    random.seed(args.experiment.manual_seed)
    torch.manual_seed(args.experiment.manual_seed)

    # instantiate Trainer class with parameters "meta" parameters
    trainer_partial = partial(
        Trainer,
        device=device,
        epochs=args.optim.epochs,
        debug=args.experiment.debug,
        criterion=criterion,
        checkpoint=checkpoint,
        output_path=output_dirpath
    )
    # modeling function for training
    modeling(
        trainer=trainer_partial,
        device=device,
        batch_size=args.data.batch_size,
        num_workers=args.data.num_workers,
        train_dataset_args=args.data.train_dataset,
        val_dataset_args=args.data.val_dataset,
        optimizer_args=args.optim.optimizer,
        scheduler_args=args.optim.scheduler,
        model_args=args.model.model,
        logger=logger,
        num_unfrozen=args.optim.num_unfrozen,
    )

    if args.experiment.bucket_name and not args.experiment.debug:
        upload_experiment_to_s3(experiment_id=logger.log_writer.run.id, dir_path=output_dirpath,
                                bucket_name=args.experiment.bucket_name, include_parent=True, logger=logger)
        


if __name__ == "__main__":
    main()
