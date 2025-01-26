
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
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf 
from soundbay.utils.conf_validator import Config
import hydra
import random
from unittest.mock import Mock
from copy import deepcopy
from soundbay.utils.app import App
from soundbay.utils.logging import Logger, flatten, get_experiment_name
from soundbay.utils.checkpoint_utils import upload_experiment_to_s3
from soundbay.trainers import Trainer
from soundbay.conf_dict import models_dict, criterion_dict, datasets_dict, optim_dict, scheduler_dict
import string


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
    freeze_layers_for_finetune,
    equalize_data,
    label_type
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
    equalize_data - Boolean argument for data equalization - given frequency of each class`

    """
    # Set paths and create dataset

    train_dataset = datasets_dict[train_dataset_args['_target_']](data_path = train_dataset_args['data_path'],
    metadata_path=train_dataset_args['metadata_path'], augmentations=train_dataset_args['augmentations'],
    augmentations_p=train_dataset_args['augmentations_p'],
    preprocessors=train_dataset_args['preprocessors'],
    seq_length=train_dataset_args['seq_length'], data_sample_rate=train_dataset_args['data_sample_rate'],
    sample_rate=train_dataset_args['sample_rate'], margin_ratio=train_dataset_args['margin_ratio'],
    slice_flag=train_dataset_args['slice_flag'], mode=train_dataset_args['mode'],
    path_hierarchy=train_dataset_args['path_hierarchy'],
    label_type=label_type
    )

    # train data which is handled as validation data
    train_as_val_dataset = datasets_dict[train_dataset_args['_target_']](data_path=train_dataset_args['data_path'],
    metadata_path=train_dataset_args['metadata_path'], augmentations=val_dataset_args['augmentations'],
    augmentations_p=val_dataset_args['augmentations_p'],
    preprocessors=val_dataset_args['preprocessors'],
    seq_length=val_dataset_args['seq_length'], data_sample_rate=train_dataset_args['data_sample_rate'],
    sample_rate=train_dataset_args['sample_rate'], margin_ratio=val_dataset_args['margin_ratio'],
    slice_flag=val_dataset_args['slice_flag'], mode=val_dataset_args['mode'],
    path_hierarchy=val_dataset_args['path_hierarchy'],
    label_type=label_type
    )

    val_dataset = datasets_dict[val_dataset_args['_target_']](data_path = val_dataset_args['data_path'],
    metadata_path=val_dataset_args['metadata_path'], augmentations=val_dataset_args['augmentations'],
    augmentations_p=val_dataset_args['augmentations_p'],
    preprocessors=val_dataset_args['preprocessors'],
    seq_length=val_dataset_args['seq_length'], data_sample_rate=val_dataset_args['data_sample_rate'],
    sample_rate=val_dataset_args['sample_rate'], margin_ratio=val_dataset_args['margin_ratio'],
    slice_flag=val_dataset_args['slice_flag'], mode=val_dataset_args['mode'],
    path_hierarchy=train_dataset_args['path_hierarchy'],
    label_type=label_type
    )

    # Define model and device for training
    model_args = dict(model_args)
    model = models_dict[model_args.pop('_target_')](**model_args)

    print('*** model has been loaded successfully ***')
    print(f'number of trainable params: {sum([p.numel() for p in model.parameters() if p.requires_grad]):,}')
    model.to(device)

    # Assert number of labels in the dataset and the number of labels in the model
    assert model_args['num_classes'] == train_dataset.num_classes == val_dataset.num_classes, \
    "Num of classes in model and the datasets must be equal, check your configs and your dataset labels!!"

    # Add model watch to WANDB
    logger.log_writer.watch(model)

    # Define dataloader for training and validation datasets as well as optimizers arguments
    if equalize_data:
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

    optimizer_args = dict(optimizer_args)
    optimizer = optim_dict[optimizer_args.pop('_target_')](model.parameters(), **optimizer_args)

    scheduler = scheduler_dict[scheduler_args._target_](optimizer, gamma=scheduler_args['gamma'])

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

    # Freeze layers if required (optim.freeze_layers_for_finetune==True)
    if freeze_layers_for_finetune:
        model.freeze_layers()

    # Commence training

    _trainer.train()

    return


# TODO check how to use hydra without path override
@hydra.main(config_name="/runs/main", config_path="conf", version_base='1.2')
def main(validate_args) -> None:
    
    args = deepcopy(validate_args)
    OmegaConf.resolve(validate_args)
    Config(**validate_args)
    # Set logger
    if args.experiment.debug:
        _logger = Mock()
        _logger.run.id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    else:
        _logger = wandb

    experiment_name = get_experiment_name(args)
    _logger.init(project="finding_willy", name=experiment_name, group=args.experiment.group_name,
                 id=args.experiment.run_id, resume=args.experiment.checkpoint.resume)

    # Set device
    if not torch.cuda.is_available():
        print('CPU!!!!!!!!!!!')
        device = torch.device("cpu")
    else:
        print('GPU!!!!!!!!!')
        device = torch.device("cuda")

    # Convert filepaths, convenient if you wish to use relative paths
    hydra_dirpath = Path(hydra.utils.get_original_cwd())
    working_dirpath = Path.cwd()
    assert working_dirpath == hydra_dirpath, "hydra is doing funky stuff with the paths again, check it out"
    output_dirpath = working_dirpath / f'../checkpoints/{_logger.run.id}'
    output_dirpath.mkdir(parents=True)
    OmegaConf.save(args, output_dirpath / 'args.yaml', resolve=False)  # we prefer to save the referenced version,
    # we can always resolve once we load the conf again

    # Define checkpoint
    if args.experiment.checkpoint.path:
        checkpoint = working_dirpath / args.experiment.checkpoint.path
        assert checkpoint.exists(), 'Checkpoint does not exists!'
    else:
        checkpoint = None

    # Logging
    logger = Logger(_logger, debug_mode=args.experiment.debug, artifacts_upload_limit=args.experiment.artifacts_upload_limit)
    flattenArgs = flatten(args)
    logger.log_writer.config.update(flattenArgs)
    App.init(args)

    # Define criterion
    # criterion = instantiate(args.model.criterion)
    if args.model.criterion._target_ in ['torch.nn.CrossEntropyLoss', 'torch.nn.BCEWithLogitsLoss']:
        criterion = criterion_dict[args.model.criterion._target_]
    
        # criterion = torch.nn.CrossEntropyLoss()

    # Seed script
    if args.experiment.manual_seed is None:
        args.experiment.manual_seed = random.randint(1, 10000)
    random.seed(args.experiment.manual_seed)
    torch.manual_seed(args.experiment.manual_seed)


    # extra asserts
    assert args.data.max_freq == args.data.sample_rate // 2, "max_freq must be equal to sample_rate // 2"

    # Finetune
    if args.optim.freeze_layers_for_finetune is None:
        args.optim.freeze_layers_for_finetune = False
    if args.optim.freeze_layers_for_finetune:
        print('The model is in finetune mode!')

    # proba threshold (for multi-label classification, using "get" for backward compatibility)
    classification_proba_threshold = args.data.get('proba_threshold', None)

    # label type, using "get" for backward compatibility
    label_type = args.data.get('label_type', 'single_label')

    # instantiate Trainer class with parameters "meta" parameters
    trainer_partial = partial(
        Trainer,
        device=device,
        epochs=args.optim.epochs,
        debug=args.experiment.debug,
        criterion=criterion,
        checkpoint=checkpoint,
        output_path=output_dirpath,
        load_optimizer_state=args.experiment.checkpoint.load_optimizer_state,
        label_names=args.data.label_names,
        label_type=label_type,
        proba_threshold=classification_proba_threshold,
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
        freeze_layers_for_finetune=args.optim.freeze_layers_for_finetune,
        equalize_data=args.experiment.equalize_data,
        label_type=label_type
    )

    if args.experiment.bucket_name and not args.experiment.debug:
        upload_experiment_to_s3(experiment_id=logger.log_writer.run.id, dir_path=output_dirpath,
                                bucket_name=args.experiment.bucket_name, include_parent=True, logger=logger)
        

if __name__ == "__main__":
    main()
