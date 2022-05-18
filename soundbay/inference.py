from typing import Generator, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
import hydra
from pathlib import Path
import os
import pandas
import datetime
from hydra.utils import instantiate
from soundbay.utils.logging import Logger, flatten, get_experiment_name
from soundbay.utils.checkpoint_utils import merge_with_checkpoint
import wandb
from unittest.mock import Mock

def predict_proba(model: torch.nn.Module, data_loader: DataLoader,
                  device: torch.device = torch.device('cpu'),
                  selected_class_idx: Union[None, int] = None,
                  ) -> np.ndarray:
    """
    calculates the predicted probability to belong to a class for all the samples in the dataset given a specific model
    Input:
        model: the wanted trained model for the inference
        data_loader: dataloader class, containing the dataset location, metadata, batch size etc.
        device: cpu or gpu - torch.device()
        selected_class_idx: the wanted class for prediction. must be bound by the number of classes in the model

    Output:
        softmax_activation: the vector of the predictions of all the samples after a softmax function

    """
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
        softmax_activation = softmax(all_predictions, 1)
        return softmax_activation


def predict(model: torch.nn.Module, data_loader: Generator[torch.tensor, None, None],
            device: torch.device = torch.device('cpu'),
            threshold: Union[float, None] = None, selected_class_idx: int = 1
            ) -> np.ndarray:
    """
   calculates the predicted probability to belong to a class for all the samples in the dataset given a specific model
    Input:
        model: the wanted trained model for the inference
        data_loader: dataloader class, containing the dataset location, metadata, batch size etc.
        device: cpu or gpu - torch.device()
        threshold: a number between 0 and 1 to decide if the classification is positive
        selected_class_idx: the wanted class for prediction. must be bound by the number of classes in the model

    Output:
        prediction: binary prediction given the threshold, the model and the wanted class
    """
    if threshold is None:
        predicted_probability = predict_proba(model, data_loader, device)
        return predicted_probability
    else:
        predicted_probability = predict_proba(model, data_loader, device,
                                              selected_class_idx=selected_class_idx).reshape((-1, 1))
        return (predicted_probability > threshold).reshape((-1, 1))


def load_model(model_params, checkpoint_state_dict):
    """
    load_model receives model params and state dict, instantiating a model and loading trained parameters.
    Input:
        model_params: config arguments of model object
        checkpoint_state_dict: dict including the train parameters to be loaded to the model
    Output:
        model: nn.Module object of the model
    """

    model = instantiate(model_params)
    model.load_state_dict(checkpoint_state_dict)
    return model


def inference_to_file(
    args,
    device,
    batch_size,
    dataset_args,
    model_args,
    checkpoint_state_dict,
    output_path,
    model_name,
    save_raven,
    threshold
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
    # set paths and create dataset
    test_dataset = instantiate(dataset_args)

    # load model
    model = load_model(model_args, checkpoint_state_dict).to(device)

    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=0,
                                 pin_memory=False)

    # predict
    predict_prob = predict_proba(model, test_dataloader, device, None)
    results_df = pandas.DataFrame(predict_prob, columns=['class0_prob', 'class1_prob'])

    if hasattr(test_dataset, 'metadata'):
        concat_dataset = pandas.concat([test_dataset.metadata, results_df], axis=1)
        metrics_dict = Logger.get_metrics_dict(concat_dataset["label"].values.tolist(),
                                               np.array(concat_dataset["class1_prob"].values.tolist()) >= 0.5,
                                               results_df.values)
        print(metrics_dict)

        print('Logging metrics to wandb...')
        _logger = wandb if not args.experiment.debug else Mock()
        experiment_name = get_experiment_name(args)
        _logger.init(project="soundbay-inference", name=experiment_name, group=args.experiment.group_name,
                     id=args.experiment.run_id)
        logger = Logger(_logger, debug_mode=args.experiment.debug,
                        artifacts_upload_limit=args.experiment.artifacts_upload_limit)
        flattenArgs = flatten(args)
        logger.log_writer.config.update(flattenArgs)
        logger.log_writer.config.update(metrics_dict)
        label_names = ('Noise', 'Call')
        logger.log_writer.log(
            {f'test_charts/PR Curve': wandb.plot.pr_curve(concat_dataset["label"].values.tolist(),
                                                          np.array([concat_dataset["class0_prob"].values.tolist(),
                                                                   concat_dataset["class1_prob"].values.tolist()]).T,
                                                          labels=label_names)}, step=1)
        # logger.log_writer.log(
        #     {f'test_charts/ROC Curve': wandb.plot.roc_curve(concat_dataset["label"].values.tolist(),
        #     np.array(concat_dataset["class1_prob"].values.tolist()), labels=label_names)}, step=1)
        wandb.log({f'test_charts/conf_mat': wandb.plot.confusion_matrix(probs=None, y_true=concat_dataset["label"].values.tolist(),
                                                                          preds=np.array(concat_dataset["class1_prob"].values.tolist()) >= 0.5,
                                                                          class_names=label_names)},
                  step=1, commit=False)

    else:
        concat_dataset = results_df
        print("Notice: The dataset has no ground truth labels")

    # save file
    dataset_name = Path(test_dataset.metadata_path).stem
    filename = f"Inference_results-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{model_name}-{dataset_name}.csv"
    output_file = output_path / filename
    concat_dataset.to_csv(index=False, path_or_buf=output_file)
    # save raven file
    if save_raven:
        thresholdtext = int(threshold*10)
        raven_filename = f"raven_annotations-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{model_name}-{dataset_name}-thresh0{thresholdtext}.csv"
        raven_output_file = output_path / raven_filename
        raven_df = inference_csv_to_raven(results_df, test_dataset.seq_length, threshold=threshold)
        raven_df.to_csv(index=False, path_or_buf=raven_output_file, sep="\t")

    return

def inference_csv_to_raven(probsdataframe: pd.DataFrame, seq_length: float, threshold: float = 0.5):
    # transforms the probability dataframe to a raven format with class_1 predictions as the annotated bounding boxes
    len_dataset = probsdataframe.shape[0]  # number of segments in wav
    seq_len = seq_length
    all_begin_times = np.arange(0, len_dataset * seq_len, seq_len)

    if_positive = probsdataframe['class1_prob'] > threshold  # check if the probability is above the threshold
    begin_times = all_begin_times[if_positive]
    # begin_times = np.round(all_begin_times[if_positive], decimals=1) #rounded to avoid float errors (e.g. 24.00000000000001)

    end_times = np.round(begin_times+seq_len, decimals=1)
    if end_times[-1] > round(len_dataset*seq_len,1):
        end_times[-1] = round(len_dataset*seq_len,1) #cut off last bbox if exceeding eof
    low_freq = np.zeros_like(begin_times)
    high_freq = np.ones_like(begin_times)*20000 #just tall enough bounding box
    view = ['Spectrogram 1']*len(begin_times)
    selection = np.arange(1,len(begin_times)+1)
    channel = np.ones_like(begin_times).astype(int)
    bboxes = {'Selection': selection, 'View': view, 'Channel': channel,
              'Begin Time (s)': begin_times, 'End Time (s)': end_times,
              'Low Freq (Hz)': low_freq, 'High Freq (Hz)': high_freq}
    annotations_df = pandas.DataFrame(data = bboxes)




    return annotations_df

@hydra.main(config_name="runs/main_inference", config_path="conf")
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
        args=args,
        device=device,
        batch_size=args.data.batch_size,
        dataset_args=args.data.test_dataset,
        model_args=args.model.model,
        checkpoint_state_dict=ckpt,
        output_path=output_dirpath,
        model_name=Path(args.experiment.checkpoint.path).stem,
        save_raven=args.experiment.save_raven,
        threshold=args.experiment.threshold
    )
    print("Finished inference")


if __name__ == "__main__":
    inference_main()
