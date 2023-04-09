import copy
import os
import torch
import wandb
from soundbay.utils.logging import Logger
from soundbay.inference import predict
from soundbay.trainers import Trainer
from pathlib import Path
from soundbay.utils.app import App
from omegaconf import DictConfig


class VariablesChangeException(Exception):
    pass


def check_variable_change(model_before, model_after, vars_change=True, device=torch.device('cpu')):
    params = [np for np in model_before.named_parameters() if np[1].requires_grad]
    initial_params = [np for np in model_after.named_parameters() if np[1].requires_grad]

    # check if variables have changed
    for (_, p0), (name, p1) in zip(initial_params, params):
        try:
            if vars_change:
                assert not torch.equal(p0.to(device), p1.to(device))
            else:
                assert torch.equal(p0.to(device), p1.to(device))
        except AssertionError:
            raise VariablesChangeException(  # error message
                "{var_name} {msg}".format(
                    var_name=name,
                    msg='did not change!' if vars_change else 'changed!'
                )
            )


def test_trainer(model, optimizer, scheduler, train_data_loader, criterion):

    wandb.init(project=None, mode='disabled')
    args = DictConfig({'experiment': {'debug': False}})
    App.init(args)
    output_dirpath = Path.cwd()

    # basic training run of model
    pre_training_model = copy.deepcopy(model)
    logger = Logger(debug_mode=True)
    trainer = Trainer(
        model=model,
        train_dataloader=train_data_loader,
        val_dataloader=train_data_loader,
        train_as_val_dataloader=train_data_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=2,
        logger=logger,
        debug=True,
        criterion=criterion,
        output_path=output_dirpath
    )
    trainer.train()
    check_variable_change(pre_training_model, model)
    assert trainer.epochs_trained == 2

    # check that can load from checkpoint
    pre_training_model = copy.deepcopy(model)
    trainer = Trainer(
        model=model,
        train_dataloader=train_data_loader,
        val_dataloader=train_data_loader,
        train_as_val_dataloader=train_data_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=4,
        logger=logger,
        debug=True,
        criterion=criterion,
        checkpoint='last.pth',
        output_path=output_dirpath
    )
    trainer.train()
    os.remove('last.pth')
    os.remove('best.pth')
    check_variable_change(pre_training_model, model)
    assert trainer.epochs_trained == 4


def test_inference(model, inference_data_loader):
    y = predict(model, inference_data_loader)
    assert y.sum() != 0
    predict(model, inference_data_loader, threshold=0.7, selected_class_idx=1)
