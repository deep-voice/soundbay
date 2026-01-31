
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
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb
from pathlib import Path
from omegaconf import OmegaConf 
import random
from unittest.mock import Mock
from soundbay.utils.app import app
from soundbay.utils.misc import instantiate_from_string
from soundbay.utils.logging import Logger, get_experiment_name
from soundbay.utils.checkpoint_utils import upload_experiment_to_s3
from soundbay.trainers import Trainer
from soundbay.conf_dict import models_dict, criterion_dict, datasets_dict
from soundbay.preprocessing import Preprocessor
from soundbay.augmentations import Augmentor
import string
import click
from typing import Optional
from soundbay.config import create_training_config
from dataclasses import asdict


def modeling(
    device,
    logger,
    output_path,
    checkpoint,
):
    """
    """
    # Set paths and create dataset
    train_dataset_args = app.args.data.train_dataset
    val_dataset_args = app.args.data.val_dataset
    model_args = app.args.model
    optimizer_args = app.args.optim.optimizer
    scheduler_args = app.args.optim.scheduler
    batch_size = app.args.data.batch_size
    num_workers = app.args.data.num_workers

    # create preprocessors
    preprocessor = Preprocessor(
        audio_representation=app.args.data.audio_representation,
        normalization=app.args.data.normalization,
        resize=app.args.data.resize,
        size=app.args.data.size,
        sample_rate=app.args.data.sample_rate,
        min_freq=app.args.data.min_freq,
        n_fft=app.args.data.n_fft,
        hop_length=app.args.data.hop_length,
        n_mels=app.args.data.n_mels
    )

    # create augmentor
    augmentor = Augmentor(
        pitch_shift_p=train_dataset_args.augmentations_config.pitch_shift_p,
        time_stretch_p=train_dataset_args.augmentations_config.time_stretch_p,
        time_masking_p=train_dataset_args.augmentations_config.time_masking_p,
        frequency_masking_p=train_dataset_args.augmentations_config.frequency_masking_p,
        min_semitones=train_dataset_args.augmentations_config.min_semitones,
        max_semitones=train_dataset_args.augmentations_config.max_semitones,
        min_rate=train_dataset_args.augmentations_config.min_rate,
        max_rate=train_dataset_args.augmentations_config.max_rate,
        min_band_part=train_dataset_args.augmentations_config.min_band_part,
        max_band_part=train_dataset_args.augmentations_config.max_band_part,
        min_bandwidth_fraction=train_dataset_args.augmentations_config.min_bandwidth_fraction,
        max_bandwidth_fraction=train_dataset_args.augmentations_config.max_bandwidth_fraction,
        add_multichannel_background_noise_p=train_dataset_args.augmentations_config.add_multichannel_background_noise_p,
        min_snr_in_db=train_dataset_args.augmentations_config.min_snr_in_db,
        max_snr_in_db=train_dataset_args.augmentations_config.max_snr_in_db,
        lru_cache_size=train_dataset_args.augmentations_config.lru_cache_size,
        sounds_path=train_dataset_args.augmentations_config.sounds_path,
        min_center_freq=app.args.data.min_freq
    )

    train_dataset = datasets_dict[train_dataset_args.module_name](
        data_path = train_dataset_args.data_path,
        metadata_path=train_dataset_args.metadata_path, 
        augmentor=augmentor,
        augmentations_p=train_dataset_args.augmentations_p,
        preprocessor=preprocessor,
        seq_length=app.args.data.seq_length, 
        data_sample_rate=app.args.data.data_sample_rate,
        sample_rate=app.args.data.sample_rate, 
        margin_ratio=train_dataset_args.margin_ratio,
        slice_flag=train_dataset_args.slice_flag, 
        mode=train_dataset_args.mode,
        path_hierarchy=train_dataset_args.path_hierarchy,
        label_type=app.args.data.label_type,
    )

    # train data which is handled as validation data
    train_as_val_dataset = datasets_dict[train_dataset_args.module_name](
        data_path=train_dataset_args.data_path,
        metadata_path=train_dataset_args.metadata_path, 
        augmentor=augmentor,
        augmentations_p=val_dataset_args.augmentations_p,
        preprocessor=preprocessor,
        seq_length=app.args.data.seq_length, 
        data_sample_rate=app.args.data.data_sample_rate,
        sample_rate=app.args.data.sample_rate, 
        margin_ratio=val_dataset_args.margin_ratio,
        slice_flag=val_dataset_args.slice_flag, 
        mode=val_dataset_args.mode,
        path_hierarchy=val_dataset_args.path_hierarchy,
        label_type=app.args.data.label_type,
    )

    val_dataset = datasets_dict[val_dataset_args.module_name](
        data_path = val_dataset_args.data_path,
        metadata_path=val_dataset_args.metadata_path, 
        augmentor=None,
        augmentations_p=val_dataset_args.augmentations_p,
        preprocessor=preprocessor,
        seq_length=app.args.data.seq_length, 
        data_sample_rate=app.args.data.data_sample_rate,
        sample_rate=app.args.data.sample_rate, 
        margin_ratio=val_dataset_args.margin_ratio,
        slice_flag=val_dataset_args.slice_flag, 
        mode=val_dataset_args.mode,
        path_hierarchy=train_dataset_args.path_hierarchy,
        label_type=app.args.data.label_type
    )

    # Define model and device for training
    model = models_dict[model_args.module_name](num_classes=model_args.num_classes, **model_args.model_params)

    print('*** model has been loaded successfully ***')
    print(f'number of trainable params: {sum([p.numel() for p in model.parameters() if p.requires_grad]):,}')
    model.to(device)

    # Assert number of labels in the dataset and the number of labels in the model
    assert model_args.num_classes == train_dataset.num_classes == val_dataset.num_classes, \
    "Num of classes in model and the datasets must be equal, check your configs and your dataset labels!!"

    # Add model watch to WANDB
    logger.log_writer.watch(model)

    # Define dataloader for training and validation datasets as well as optimizers arguments
    if app.args.experiment.equalize_data:
        sampler = WeightedRandomSampler(train_dataset.samples_weight, len(train_dataset)) 
    else:
        sampler = None
    train_dataloader = DataLoader(
            dataset=train_dataset,
            sampler=sampler,
            shuffle=sampler is None,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
    val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    train_as_val_dataloader = DataLoader(
            dataset=train_as_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    optimizer = instantiate_from_string(optimizer_args.module_name, model.parameters(), **optimizer_args.params)
    scheduler = instantiate_from_string(scheduler_args.module_name, optimizer, **scheduler_args.params)
    criterion = criterion_dict[model_args.criterion]
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_as_val_dataloader=train_as_val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        logger=logger,
        device=device,
        epochs=app.args.optim.epochs,
        debug=app.args.experiment.debug,
        checkpoint=checkpoint,
        output_path=output_path,
        load_optimizer_state=app.args.experiment.checkpoint.load_optimizer_state,
        label_names=app.args.data.label_names,
        label_type=app.args.data.label_type,
        proba_threshold=app.args.data.proba_threshold,
    )

    # Freeze layers if required
    if app.args.optim.freeze_layers_for_finetune:
        model.freeze_layers()

    # Commence training
    trainer.train()

    return


@click.command()
@click.option("--config", type=Optional[str], help="Path to configuration file")
@click.argument("overrides", nargs=-1)
def train(config: Optional[str], overrides: list[str]) -> None:
    
    args = create_training_config(config_path=config, overrides=overrides)
    app.init(args)
    # Set logger
    if args.experiment.debug:
        _logger = Mock()
        _logger.run.id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    else:
        _logger = wandb

    experiment_name = get_experiment_name(args)
    _logger.init(
        project=args.experiment.project, 
        name=experiment_name,
        group=args.experiment.group_name,
        id=args.experiment.run_id, 
        resume=args.experiment.checkpoint.resume
    )

    # Set device
    if not torch.cuda.is_available():
        print('CPU!!!!!!!!!!!')
        device = torch.device("cpu")
    else:
        print('GPU!!!!!!!!!')
        device = torch.device("cuda")

    # Convert filepaths, convenient if you wish to use relative paths
    working_dirpath = Path(__file__).parent.parent
    output_path = working_dirpath / 'checkpoints' / f'{_logger.run.id}'
    output_path.mkdir(parents=True)
    OmegaConf.save(args, output_path / 'args.yaml', resolve=False)  # we prefer to save the referenced version,

    # Define checkpoint
    if args.experiment.checkpoint.path:
        checkpoint = working_dirpath / args.experiment.checkpoint.path
        assert checkpoint.exists(), 'Checkpoint does not exists!'
    else:
        checkpoint = None

    # Logging
    logger = Logger(_logger, debug_mode=args.experiment.debug, artifacts_upload_limit=args.experiment.artifacts_upload_limit)
    logger.log_writer.config.update(asdict(args))

    # Seed script
    if args.experiment.manual_seed is None:
        args.experiment.manual_seed = random.randint(1, 10000)
    random.seed(args.experiment.manual_seed)
    torch.manual_seed(args.experiment.manual_seed)

    modeling(
        device=device,
        logger=logger,
        output_path=output_path,
        checkpoint=checkpoint,
    )

    if args.experiment.bucket_name and not args.experiment.debug:
        upload_experiment_to_s3(experiment_id=logger.log_writer.run.id, dir_path=output_path,
                                bucket_name=args.experiment.bucket_name, include_parent=True, logger=logger)
        

if __name__ == "__main__":
    train()
