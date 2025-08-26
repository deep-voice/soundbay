"""
Main training loop with dataclass-based configuration
---------------------------------------------------
This script replaces the Hydra-based training script with a dataclass-based configuration system.
The configuration supports hierarchical overrides with priority:
1. Command line overrides (highest priority)
2. Config file overrides
3. Checkpoint configuration
4. Script defaults (lowest priority)

Usage:
    python train_new.py --config_path path/to/config.yaml --checkpoint_path path/to/checkpoint
    python train_new.py --data.batch_size 32 --experiment.debug False
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb
from functools import partial
from pathlib import Path
import random
import string
from unittest.mock import Mock
from copy import deepcopy
import argparse
import sys

from soundbay.config import load_config, Config
from soundbay.utils.app import App
from soundbay.utils.logging import Logger, flatten, get_experiment_name
from soundbay.utils.checkpoint_utils import upload_experiment_to_s3
from soundbay.trainers import Trainer
from soundbay.conf_dict import models_dict, criterion_dict, datasets_dict, optim_dict, scheduler_dict


def modeling(
    trainer,
    device,
    config: Config,
    logger
):
    """
    Modeling function that instantiates datasets, model, and starts training
    
    Args:
        trainer: Trainer class instance
        device: Device (cpu/gpu)
        config: Configuration object
        logger: Logger instance
    """
    # Set paths and create dataset
    train_dataset = datasets_dict[config.data.train_dataset._target_](
        data_path=config.data.train_dataset.data_path,
        metadata_path=config.data.train_dataset.metadata_path,
        augmentations=config.data.train_dataset.augmentations,
        augmentations_p=config.data.train_dataset.augmentations_p,
        preprocessors=config.data.train_dataset.preprocessors,
        seq_length=config.data.train_dataset.seq_length,
        data_sample_rate=config.data.train_dataset.data_sample_rate,
        sample_rate=config.data.train_dataset.sample_rate,
        margin_ratio=config.data.train_dataset.margin_ratio,
        slice_flag=config.data.train_dataset.slice_flag,
        mode=config.data.train_dataset.mode,
        path_hierarchy=config.data.train_dataset.path_hierarchy,
        label_type=config.data.label_type
    )

    # Train data which is handled as validation data
    train_as_val_dataset = datasets_dict[config.data.train_dataset._target_](
        data_path=config.data.train_dataset.data_path,
        metadata_path=config.data.train_dataset.metadata_path,
        augmentations=config.data.val_dataset.augmentations,
        augmentations_p=config.data.val_dataset.augmentations_p,
        preprocessors=config.data.val_dataset.preprocessors,
        seq_length=config.data.val_dataset.seq_length,
        data_sample_rate=config.data.train_dataset.data_sample_rate,
        sample_rate=config.data.train_dataset.sample_rate,
        margin_ratio=config.data.val_dataset.margin_ratio,
        slice_flag=config.data.val_dataset.slice_flag,
        mode=config.data.val_dataset.mode,
        path_hierarchy=config.data.val_dataset.path_hierarchy,
        label_type=config.data.label_type
    )

    val_dataset = datasets_dict[config.data.val_dataset._target_](
        data_path=config.data.val_dataset.data_path,
        metadata_path=config.data.val_dataset.metadata_path,
        augmentations=config.data.val_dataset.augmentations,
        augmentations_p=config.data.val_dataset.augmentations_p,
        preprocessors=config.data.val_dataset.preprocessors,
        seq_length=config.data.val_dataset.seq_length,
        data_sample_rate=config.data.val_dataset.data_sample_rate,
        sample_rate=config.data.val_dataset.sample_rate,
        margin_ratio=config.data.val_dataset.margin_ratio,
        slice_flag=config.data.val_dataset.slice_flag,
        mode=config.data.val_dataset.mode,
        path_hierarchy=config.data.train_dataset.path_hierarchy,
        label_type=config.data.label_type
    )

    # Define model and device for training
    model_args = dict(config.model.model)
    model = models_dict[model_args.pop('_target_')](**model_args)

    print('*** model has been loaded successfully ***')
    print(f'number of trainable params: {sum([p.numel() for p in model.parameters() if p.requires_grad]):,}')
    model.to(device)

    # Assert number of labels in the dataset and the number of labels in the model
    assert model_args['num_classes'] == train_dataset.num_classes == val_dataset.num_classes, \
        "Num of classes in model and the datasets must be equal, check your configs and your dataset labels!!"

    # Add model watch to WANDB
    logger.log_writer.watch(model)

    # Define dataloader for training and validation datasets
    if config.experiment.equalize_data:
        sampler = WeightedRandomSampler(train_dataset.samples_weight, len(train_dataset))
    else:
        sampler = None
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=sampler,
        shuffle=sampler is None,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    train_as_val_dataloader = DataLoader(
        dataset=train_as_val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    # Define optimizer and scheduler
    optimizer_args = dict(config.optim.optimizer.__dict__)
    optimizer = optim_dict[optimizer_args.pop('_target_')](model.parameters(), **optimizer_args)

    scheduler_args = dict(config.optim.scheduler.__dict__)
    scheduler = scheduler_dict[scheduler_args.pop('_target_')](optimizer, **scheduler_args)

    # Add the rest of the parameters to trainer instance
    _trainer = trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_as_val_dataloader=train_as_val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger
    )

    # Freeze layers if required
    if config.optim.freeze_layers_for_finetune:
        model.freeze_layers()

    # Commence training
    _trainer.train()

    return


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train with dataclass-based configuration')
    parser.add_argument('--config_path', type=str, help='Path to YAML configuration file')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint directory')
    parser.add_argument('--overrides', type=str, help='JSON string of configuration overrides')
    
    # Parse known args to get our specific arguments
    args, remaining_args = parser.parse_known_args()
    
    # Load configuration with proper priority
    config = load_config(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        overrides=json.loads(args.overrides) if args.overrides else None,
        cmdline_args=remaining_args
    )
    
    # Set logger
    if config.experiment.debug:
        _logger = Mock()
        _logger.run.id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    else:
        _logger = wandb

    experiment_name = get_experiment_name(config)
    _logger.init(
        project=config.experiment.project,
        name=experiment_name,
        group=config.experiment.group_name,
        id=config.experiment.run_id,
        resume=config.experiment.checkpoint.resume
    )

    # Set device
    if not torch.cuda.is_available():
        print('CPU!!!!!!!!!!!')
        device = torch.device("cpu")
    else:
        print('GPU!!!!!!!!!')
        device = torch.device("cuda")

    # Set up output directory
    working_dirpath = Path.cwd()
    output_dirpath = working_dirpath / f'../checkpoints/{_logger.run.id}'
    output_dirpath.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    from soundbay.config import ConfigManager
    config_manager = ConfigManager(config)
    config_manager.save(output_dirpath / 'args.yaml')

    # Define checkpoint
    if config.experiment.checkpoint.path:
        checkpoint = working_dirpath / config.experiment.checkpoint.path
        assert checkpoint.exists(), 'Checkpoint does not exist!'
    else:
        checkpoint = None

    # Logging
    logger = Logger(_logger, debug_mode=config.experiment.debug, 
                   artifacts_upload_limit=config.experiment.artifacts_upload_limit)
    flattenArgs = flatten(config)
    logger.log_writer.config.update(flattenArgs)
    App.init(config)

    # Define criterion
    if config.model.criterion['_target_'] in ['torch.nn.CrossEntropyLoss', 'torch.nn.BCEWithLogitsLoss']:
        criterion = criterion_dict[config.model.criterion['_target_']]

    # Seed script
    if config.experiment.manual_seed is None:
        config.experiment.manual_seed = random.randint(1, 10000)
    random.seed(config.experiment.manual_seed)
    torch.manual_seed(config.experiment.manual_seed)

    # Finetune
    if config.optim.freeze_layers_for_finetune is None:
        config.optim.freeze_layers_for_finetune = False
    if config.optim.freeze_layers_for_finetune:
        print('The model is in finetune mode!')

    # Instantiate Trainer class with parameters
    trainer_partial = partial(
        Trainer,
        device=device,
        epochs=config.optim.epochs,
        debug=config.experiment.debug,
        criterion=criterion,
        checkpoint=checkpoint,
        output_path=output_dirpath,
        load_optimizer_state=config.experiment.checkpoint.load_optimizer_state,
        label_names=config.data.label_names,
        label_type=config.data.label_type,
        proba_threshold=config.data.proba_threshold,
    )
    
    # Start modeling and training
    modeling(
        trainer=trainer_partial,
        device=device,
        config=config,
        logger=logger
    )

    # Upload to S3 if specified
    if config.experiment.bucket_name and not config.experiment.debug:
        upload_experiment_to_s3(
            experiment_id=logger.log_writer.run.id,
            dir_path=output_dirpath,
            bucket_name=config.experiment.bucket_name,
            include_parent=True,
            logger=logger
        )


if __name__ == "__main__":
    import json
    main()
