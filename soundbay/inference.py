"""
Inference script
----------------
This script runs model inference on audio files and produces predictions.
The configuration is handled via dataclasses and can be overridden via command line.

Usage:
    python inference.py --checkpoint /path/to/model.pth --file /path/to/audio.wav
    python inference.py --config config.yaml --checkpoint /path/to/model.pth
    python inference.py --checkpoint /path/to/model.pth data.data_sample_rate=2000
"""

from typing import Union, Literal, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from scipy.special import expit
from pathlib import Path
import pandas
import datetime
import click

from soundbay.results_analysis import inference_csv_to_raven
from soundbay.utils.checkpoint_utils import merge_inference_config_with_checkpoint
from soundbay.conf_dict import models_dict, datasets_dict
from soundbay.config import InferenceConfig, create_inference_config
from soundbay.preprocessing import Preprocessor


def predict_proba(model: torch.nn.Module, data_loader: DataLoader,
                  device: torch.device = torch.device('cpu'),
                  selected_class_idx: Union[None, int] = None,
                  proba_norm_func: Literal["softmax", "sigmoid"] = 'softmax'
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


def load_model(model_config, checkpoint_state_dict):
    """
    load_model receives model config and state dict, instantiating a model and loading trained parameters.
    Input:
        model_config: InferenceModelConfig with module_name, num_classes, model_params
        checkpoint_state_dict: dict including the train parameters to be loaded to the model
    Output:
        model: nn.Module object of the model
    """
    model = models_dict[model_config.module_name](
        num_classes=model_config.num_classes,
        **model_config.model_params
    )
    model.load_state_dict(checkpoint_state_dict)
    return model


def infer_without_metadata(
        device: torch.device,
        batch_size: int,
        test_dataset,
        model: torch.nn.Module,
        output_path: Path,
        model_name: str,
        save_raven: bool,
        threshold: float,
        label_names: list,
        raven_max_freq: Optional[int],
        proba_norm_func: str,
        seq_length: float,
        sample_rate: int
):
    """
    This functions takes the InferenceDataset dataset and produces the model prediction to a file.
    
    Input:
        device: cpu/gpu
        batch_size: the number of samples the model will infer at once
        test_dataset: InferenceDataset instance
        model: loaded model for inference
        output_path: directory to save the prediction file
        model_name: name identifier for the model (used in output filenames)
        save_raven: whether to save Raven-compatible annotation file
        threshold: probability threshold for positive predictions in Raven output
        label_names: list of class names
        raven_max_freq: maximum frequency for Raven annotations
        proba_norm_func: 'softmax' or 'sigmoid' for probability normalization
        seq_length: length of audio segments in seconds
        sample_rate: audio sample rate
    """
    all_raven_list = []
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=0,
                                 pin_memory=False)

    # predict
    predict_prob = predict_proba(model, test_dataloader, device, None, proba_norm_func)
    
    # Set default label names if not provided
    if label_names is None:
        label_names = ['Noise'] + [f'Call_{i}' for i in range(1, predict_prob.shape[1])]

    results_df = pandas.DataFrame(predict_prob, columns=label_names)

    concat_dataset = pandas.concat([test_dataset.metadata, results_df], axis=1)
    
    # create raven file
    raven_max_freq = sample_rate // 2 if raven_max_freq is None else raven_max_freq
    if save_raven:
        for file, df in concat_dataset.groupby('filename'):
            file_raven_lists = []
            for i in range(1, predict_prob.shape[1]):
                file_raven_lists.append(
                    inference_csv_to_raven(
                        results_df=df,
                        num_classes=predict_prob.shape[1],
                        seq_len=seq_length,
                        selected_class=label_names[i],
                        threshold=threshold,
                        class_name=label_names[i],
                        max_freq=raven_max_freq
                    )
                )
            whole_file_df = pd.concat(file_raven_lists, axis=0).sort_values('Begin Time (s)')
            whole_file_df['Selection'] = np.arange(1, len(whole_file_df) + 1)
            all_raven_list.append((file, whole_file_df))

    # save file
    dataset_name = Path(test_dataset.metadata_path).stem
    filename = f"Inference_results-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{model_name}-{dataset_name}.csv"
    output_file = output_path / filename
    concat_dataset = concat_dataset.sort_values(by=['filename', 'begin_time'])
    concat_dataset.to_csv(index=False, path_or_buf=output_file)

    # Save raven file
    if save_raven:
        if Path(test_dataset.metadata_path).is_dir():
            raven_output_path = output_path / dataset_name
            raven_output_path.mkdir(exist_ok=True)
        else:
            raven_output_path = output_path
        for filename, raven_out_df in all_raven_list:
            save_raven_file(filename, raven_out_df, raven_output_path, model_name)

    return output_file


def save_raven_file(filename, raven_out_df, output_path, model_name):
    """Save inference results in Raven-compatible format."""
    raven_filename = f"{filename.stem}-Raven-inference_results-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{model_name}.txt"
    raven_output_file = output_path / raven_filename
    raven_out_df.to_csv(index=False, path_or_buf=raven_output_file, sep='\t')


def run_inference(
    args: InferenceConfig,
    checkpoint_state_dict: dict,
    output_path: Path,
    model_name: str
) -> Path:
    """
    Run inference with the given configuration and checkpoint.
    
    Input:
        args: InferenceConfig with all settings
        checkpoint_state_dict: model weights from checkpoint
        output_path: directory to save outputs
        model_name: identifier for the model
    
    Output:
        Path to the output CSV file
    """
    # Set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    
    # Create preprocessor
    preprocessor = Preprocessor(
        audio_representation=args.data.audio_representation,
        normalization=args.data.normalization,
        resize=args.data.resize,
        size=args.data.size,
        sample_rate=args.data.sample_rate,
        min_freq=args.data.min_freq,
        n_fft=args.data.n_fft,
        hop_length=args.data.hop_length,
        n_mels=args.data.n_mels
    )
    
    # Create dataset
    test_dataset = datasets_dict[args.data.test_dataset.module_name](
        file_path=args.data.test_dataset.file_path,
        preprocessor=preprocessor,
        seq_length=args.data.seq_length,
        data_sample_rate=args.data.data_sample_rate,
        sample_rate=args.data.sample_rate,
        overlap=args.data.test_dataset.overlap
    )
    
    # Load model
    model = load_model(args.model, checkpoint_state_dict).to(device)
    
    # Determine probability normalization function
    default_norm_func = 'softmax' if args.data.label_type == 'single_label' else 'sigmoid'
    
    # Run inference
    output_file = infer_without_metadata(
        device=device,
        batch_size=args.data.batch_size,
        test_dataset=test_dataset,
        model=model,
        output_path=output_path,
        model_name=model_name,
        save_raven=args.experiment.save_raven,
        threshold=args.experiment.threshold,
        label_names=args.data.label_names,
        raven_max_freq=args.experiment.raven_max_freq,
        proba_norm_func=default_norm_func,
        seq_length=args.data.seq_length,
        sample_rate=args.data.sample_rate
    )
    
    return output_file


@click.command()
@click.option("--config", type=Optional[str], default=None, help="Path to configuration YAML file")
@click.option("--checkpoint", required=True, help="Path to model checkpoint file")
@click.option("--file", "file_path", default=None, help="Path to audio file or directory for inference")
@click.option("--output", "output_path", default=None, help="Directory to save output files (default: ./outputs)")
@click.option("--save-raven/--no-save-raven", default=False, help="Save Raven-compatible annotation file")
@click.argument("overrides", nargs=-1)
def inference_main(
    config: Optional[str],
    checkpoint: str,
    file_path: Optional[str],
    output_path: Optional[str],
    save_raven: bool,
    overrides: tuple
) -> None:
    """
    Run model inference on audio files.
    
    The checkpoint path is required and contains the trained model weights.
    Configuration can be provided via a YAML file (--config) and/or command-line overrides.
    
    Examples:
        python inference.py --checkpoint model.pth --file audio.wav
        python inference.py --checkpoint model.pth --file audio.wav --save-raven
        python inference.py --config config.yaml --checkpoint model.pth
        python inference.py --checkpoint model.pth data.data_sample_rate=2000
    """
    # Build overrides list
    override_list = list(overrides)
    if file_path:
        override_list.append(f"data.test_dataset.file_path={file_path}")
    if save_raven:
        override_list.append("experiment.save_raven=true")
    
    # Create base config
    args = create_inference_config(config_path=config, overrides=override_list if override_list else None)
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    ckpt_args = ckpt_dict['args']
    ckpt_weights = ckpt_dict['model']
    
    # Merge config with checkpoint args
    args = merge_inference_config_with_checkpoint(args, ckpt_args)
    
    # Set output path
    if output_path:
        output_dirpath = Path(output_path)
    else:
        working_dirpath = Path(__file__).parent.parent
        output_dirpath = working_dirpath / "outputs"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    
    # Get model name from checkpoint path
    model_name = checkpoint_path.parent.stem
    
    # Run inference
    output_file = run_inference(
        args=args,
        checkpoint_state_dict=ckpt_weights,
        output_path=output_dirpath,
        model_name=model_name
    )
    
    print(f"Finished inference. Results saved to: {output_file}")


if __name__ == "__main__":
    inference_main()
