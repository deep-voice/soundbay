from typing import Generator, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from scipy.special import expit
import hydra
from pathlib import Path
import os
import pandas
import datetime
from omegaconf import OmegaConf

from soundbay.results_analysis import inference_csv_to_raven
from soundbay.utils.logging import Logger
from soundbay.utils.checkpoint_utils import merge_with_checkpoint
from soundbay.conf_dict import models_dict, datasets_dict


def predict_proba(model: torch.nn.Module, data_loader: DataLoader,
                  device: torch.device = torch.device('cpu'),
                  selected_class_idx: Union[None, int] = None,
                  proba_norm_func: str = 'softmax'
                  ) -> np.ndarray:
    """
    calculates the predicted probability to belong to a class for all the samples in the dataset given a specific model
    Input:
        model: the wanted trained model for the inference
        data_loader: dataloader class, containing the dataset location, metadata, batch size etc.
        device: cpu or gpu - torch.device()
        selected_class_idx: the wanted class for prediction. must be bound by the number of classes in the model

    Output:
        proba: the vector of the predictions normalized to probabilities.

    """
    assert proba_norm_func in ['softmax', 'sigmoid'], 'proba_norm_func must be softmax or sigmoid'
    all_predictions = []
    with torch.no_grad():
        model.eval()
        for audio in tqdm(data_loader):
            audio = audio.to(device)

            predicted_probability = model(audio).cpu().numpy()
            if selected_class_idx is None:
                all_predictions.extend(predicted_probability)
            else:
                if selected_class_idx in list(range(predicted_probability.shape[1])):
                    return predicted_probability[:, selected_class_idx]
                else:
                    raise ValueError(f'selected class index {selected_class_idx} not in output dimensions')
        if proba_norm_func == 'softmax':
            proba = softmax(all_predictions, 1)
        elif proba_norm_func == 'sigmoid':
            proba = expit(all_predictions)
        else:
            raise ValueError('proba_norm_func must be softmax or sigmoid')
        return proba


def load_model(model_params, checkpoint_state_dict):
    """
    load_model receives model params and state dict, instantiating a model and loading trained parameters.
    Input:
        model_params: config arguments of model object
        checkpoint_state_dict: dict including the train parameters to be loaded to the model
    Output:
        model: nn.Module object of the model
    """

    model_params = OmegaConf.to_container(model_params) 
    model = models_dict[model_params.pop('_target_')](**model_params)
    model.load_state_dict(checkpoint_state_dict)
    return model


def infer_with_metadata(
        device,
        batch_size,
        dataset_args,
        model_args,
        checkpoint_state_dict,
        output_path,
        model_name,
        proba_norm_func
):
    """
        This functions takes the ClassifierDataset dataset and produces the model prediction to a file
        Input:
            device: cpu/gpu
            batch_size: the number of samples the model will infer at once
            dataset_args: the required arguments for the dataset class
            model_path: directory for the wanted trained model
            output_path: directory to save the prediction file
        """
    # set paths and create dataset
    test_dataset = datasets_dict[dataset_args['_target_']](data_path = dataset_args['data_path'],
    metadata_path=dataset_args['metadata_path'], augmentations=dataset_args['augmentations'],
    augmentations_p=dataset_args['augmentations_p'],
    preprocessors=dataset_args['preprocessors'],
    seq_length=dataset_args['seq_length'], data_sample_rate=dataset_args['data_sample_rate'],
    sample_rate=dataset_args['sample_rate'], 
    mode=dataset_args['mode'], slice_flag=dataset_args['slice_flag'], path_hierarchy=dataset_args['path_hierarchy'],
    label_type='single_label' if proba_norm_func == 'softmax' else 'multi_label'
    )

    # load model
    model = load_model(model_args, checkpoint_state_dict).to(device)

    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=0,
                                 pin_memory=False)

    # predict
    predict_prob = predict_proba(model, test_dataloader, device, None, proba_norm_func)

    results_df = pandas.DataFrame(predict_prob)  # add class names
    if hasattr(test_dataset, 'metadata'):
        concat_dataset = pandas.concat([test_dataset.metadata, results_df],
                                       axis=1)  # TODO: make sure metadata column order matches the prediction df order
        metrics_dict = Logger.get_metrics_dict(concat_dataset["label"].values.tolist(),
                                               np.argmax(predict_prob, axis=1).tolist(),
                                               results_df.values)
        print(metrics_dict)
    else:
        concat_dataset = results_df
        print("Notice: The dataset has no ground truth labels")

    # save file
    dataset_name = Path(test_dataset.metadata_path).stem
    filename = f"Inference_results-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{model_name}-{dataset_name}.csv"
    output_file = output_path / filename
    concat_dataset.to_csv(index=False, path_or_buf=output_file)

    # save raven file

    return


def infer_without_metadata(
        device,
        batch_size,
        dataset_args,
        model_args,
        checkpoint_state_dict,
        output_path,
        model_name,
        save_raven,
        threshold,
        label_names,
        raven_max_freq,
        proba_norm_func
):
    """
        This functions takes the InferenceDataset dataset and produces the model prediction to a file, by iterating
        on all channels in the file and concatenating the results.
        Input:
            device: cpu/gpu
            batch_size: the number of samples the model will infer at once
            dataset_args: the required arguments for the dataset class
            model_path: directory for the wanted trained model
            output_path: directory to save the prediction file
    """
    # load model
    model = load_model(model_args, checkpoint_state_dict).to(device)
    all_raven_list = []
    dataset_args = dict(dataset_args)
    dataset_type = dataset_args.pop('_target_')
    test_dataset = datasets_dict[dataset_type](**dataset_args)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=0,
                                 pin_memory=False)

    # predict
    predict_prob = predict_proba(model, test_dataloader, device, None, proba_norm_func)
    label_names = ['Noise'] + [f'Call_{i}' for i in
                               range(1, predict_prob.shape[1] + 1)] if label_names is None else label_names

    results_df = pandas.DataFrame(predict_prob, columns=label_names)

    concat_dataset = pandas.concat([test_dataset.metadata, results_df], axis=1)
    # create raven file
    raven_max_freq = dataset_args['sample_rate'] // 2 if raven_max_freq is None else raven_max_freq
    if save_raven:
        for file, df in concat_dataset.groupby('filename'):
            file_raven_lists = []
            for i in range(1, predict_prob.shape[1]):
                file_raven_lists.append(
                                       inference_csv_to_raven(results_df=df,
                                                              num_classes=predict_prob.shape[1],
                                                              seq_len=dataset_args['seq_length'],
                                                              selected_class=label_names[i],
                                                              threshold=threshold,
                                                              class_name=label_names[i],
                                                              max_freq=raven_max_freq)
                                   )
            whole_file_df = pd.concat(file_raven_lists, axis=0).sort_values('Begin Time (s)')
            whole_file_df['Selection'] = np.arange(1, len(whole_file_df) + 1)
            all_raven_list.append((file, whole_file_df))

    #save file
    dataset_name = Path(test_dataset.metadata_path).stem
    filename = f"Inference_results-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{model_name}-{dataset_name}.csv"
    output_file = output_path / filename
    concat_dataset = concat_dataset.sort_values(by=['filename', 'begin_time'])
    concat_dataset.to_csv(index=False, path_or_buf=output_file)

    # Save raven file
    if save_raven:
        if Path(test_dataset.metadata_path).is_dir():
            output_path = output_path / dataset_name
            output_path.mkdir(exist_ok=True)
        for filename, raven_out_df in all_raven_list:
            save_raven_file(filename, raven_out_df, output_path, model_name)

    return


def save_raven_file(filename, raven_out_df, output_path, model_name):
    raven_filename = f"{filename.stem}-Raven-inference_results-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{model_name}.txt"
    raven_output_file = output_path / raven_filename
    raven_out_df.to_csv(index=False, path_or_buf=raven_output_file, sep='\t')


def inference_to_file(
    device,
    batch_size,
    dataset_args,
    model_args,
    checkpoint_state_dict,
    output_path,
    model_name,
    save_raven,
    threshold,
    label_names,
    raven_max_freq,
    proba_norm_func
):
    """
    This functions takes the dataset and produces the model prediction to a file
    Input:
        device: cpu/gpu
        batch_size: the number of samples the model will infer at once
        dataset_args: the required arguments for the dataset class
        model_path: directory for the wanted trained model
        output_path: directory to save the prediction file
    """
    if dataset_args._target_.endswith('ClassifierDataset') or dataset_args._target_.endswith('NoBackGroundDataset'):
        infer_with_metadata(device,
                         batch_size,
                         dataset_args,
                         model_args,
                         checkpoint_state_dict,
                         output_path,
                         model_name,
                         proba_norm_func
                         )
    elif dataset_args._target_.endswith('InferenceDataset'):
        infer_without_metadata(device,
                          batch_size,
                          dataset_args,
                          model_args,
                          checkpoint_state_dict,
                          output_path,
                          model_name,
                          save_raven,
                          threshold,
                          label_names,
                          raven_max_freq,
                          proba_norm_func)
    else:
        raise ValueError('Only ClassifierDataset or InferenceDataset allowed in inference')


@hydra.main(config_name="/runs/main_inference.yaml", config_path="conf", version_base='1.2')
def inference_main(args) -> None:
    """
    The main function for running predictions using a trained model on a wanted dataset. The arguments are inserted
    using hydra from a configuration file.
    Input:
        args: from conf file (for instance: main_inference.yaml)
    """
    # set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    working_dirpath = Path(hydra.utils.get_original_cwd())
    os.chdir(working_dirpath)
    output_dirpath = working_dirpath.parent.absolute() / "outputs"
    output_dirpath.mkdir(exist_ok=True)

    ckpt_dict = torch.load(args.experiment.checkpoint.path, map_location=torch.device('cpu'))
    ckpt_args = ckpt_dict['args']
    args = merge_with_checkpoint(args, ckpt_args)
    ckpt = ckpt_dict['model']

    inference_to_file(
        device=device,
        batch_size=args.data.batch_size,
        dataset_args=args.data.test_dataset,
        model_args=args.model.model,
        checkpoint_state_dict=ckpt,
        output_path=output_dirpath,
        model_name=Path(args.experiment.checkpoint.path).parent.stem,
        save_raven=args.experiment.save_raven,
        threshold=args.experiment.threshold,
        label_names=args.data.label_names,
        raven_max_freq=args.experiment.raven_max_freq,
        proba_norm_func=args.data.get('proba_norm_func', 'softmax') # using "get" for backward compatibility
    )
    print("Finished inference")


if __name__ == "__main__":
    inference_main()
