"""
Evaluate script
---------------
This script evaluates a trained model on labeled datasets (ClassifierDataset/NoBackGroundDataset)
and calculates performance metrics.

Usage:
    python -m soundbay.evaluate --checkpoint model.pth --metadata annotations.csv --data-path audio/
    python -m soundbay.evaluate --config config.yaml --checkpoint model.pth
    python -m soundbay.evaluate --checkpoint model.pth data.data_sample_rate=2000
"""

from typing import Optional, Literal, Dict, Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from scipy.special import expit
from pathlib import Path
import datetime
import click

from soundbay.utils.checkpoint_utils import merge_evaluate_config_with_checkpoint
from soundbay.conf_dict import models_dict, datasets_dict
from soundbay.config import EvaluateConfig, create_evaluate_config
from soundbay.preprocessing import Preprocessor
from soundbay.utils.metrics import MetricsCalculator


def predict_proba(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device = torch.device('cpu'),
    proba_norm_func: Literal["softmax", "sigmoid"] = 'softmax'
) -> np.ndarray:
    """
    Calculate predicted probabilities for all samples in the dataset.
    
    Input:
        model: trained model for inference
        data_loader: DataLoader containing the dataset
        device: cpu or gpu
        proba_norm_func: 'softmax' or 'sigmoid' for probability normalization
    
    Output:
        proba: array of predicted probabilities
    """
    assert proba_norm_func in ['softmax', 'sigmoid'], 'proba_norm_func must be softmax or sigmoid'
    all_predictions = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(data_loader):
            # ClassifierDataset returns (audio, label, audio_raw, metadata_dict)
            audio = batch[0].to(device)
            predicted_probability = model(audio).cpu().numpy()
            all_predictions.extend(predicted_probability)
    
    if proba_norm_func == 'softmax':
        proba = softmax(all_predictions, 1)
    elif proba_norm_func == 'sigmoid':
        proba = expit(all_predictions)
    else:
        raise ValueError('proba_norm_func must be softmax or sigmoid')
    
    return proba


def load_model(model_config, checkpoint_state_dict):
    """
    Load model with trained weights.
    
    Input:
        model_config: model configuration with module_name, num_classes, model_params
        checkpoint_state_dict: trained model weights
    Output:
        model: nn.Module with loaded weights
    """
    model = models_dict[model_config.module_name](
        num_classes=model_config.num_classes,
        **model_config.model_params
    )
    model.load_state_dict(checkpoint_state_dict)
    return model


def run_evaluation(
    args: EvaluateConfig,
    checkpoint_state_dict: dict,
    output_path: Path,
    model_name: str
) -> tuple[Path, Dict[str, Any]]:
    """
    Run evaluation with the given configuration and checkpoint.
    
    Input:
        args: EvaluateConfig with all settings
        checkpoint_state_dict: model weights from checkpoint
        output_path: directory to save outputs
        model_name: identifier for the model
    
    Output:
        Tuple of (output CSV path, metrics dictionary)
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
        data_path=args.data.test_dataset.data_path,
        metadata_path=args.data.test_dataset.metadata_path,
        augmentor=None,  # No augmentation for evaluation
        augmentations_p=args.data.test_dataset.augmentations_p,
        preprocessor=preprocessor,
        seq_length=args.data.seq_length,
        data_sample_rate=args.data.data_sample_rate,
        sample_rate=args.data.sample_rate,
        margin_ratio=args.data.test_dataset.margin_ratio,
        slice_flag=args.data.test_dataset.slice_flag,
        mode=args.data.test_dataset.mode,
        path_hierarchy=args.data.test_dataset.path_hierarchy,
        label_type=args.data.label_type
    )
    
    # Create data loader
    test_dataloader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=args.data.batch_size,
        num_workers=args.data.num_workers,
        pin_memory=True
    )
    
    # Load model
    model = load_model(args.model, checkpoint_state_dict).to(device)
    
    # Determine probability normalization function
    proba_norm_func = 'softmax' if args.data.label_type == 'single_label' else 'sigmoid'
    
    # Run prediction
    predict_prob = predict_proba(model, test_dataloader, device, proba_norm_func)
    
    # Get label names
    label_names = args.data.label_names
    if label_names is None:
        label_names = [f'Class_{i}' for i in range(predict_prob.shape[1])]
    
    # Create results dataframe with probabilities
    results_df = pd.DataFrame(predict_prob, columns=label_names)
    
    # Merge with metadata (ground truth labels)
    metrics_dict = {}
    if hasattr(test_dataset, 'metadata'):
        concat_dataset = pd.concat([test_dataset.metadata.reset_index(drop=True), results_df], axis=1)
        
        # Calculate metrics
        metrics_dict = MetricsCalculator(
            label_list=concat_dataset["label"].values.tolist(),
            pred_list=np.argmax(predict_prob, axis=1).tolist(),
            pred_proba_list=predict_prob,
            label_type=args.data.label_type
        ).calc_all_metrics()
        
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        for metric_name, metric_value in metrics_dict.items():
            if isinstance(metric_value, float):
                print(f"  {metric_name}: {metric_value:.4f}")
            else:
                print(f"  {metric_name}: {metric_value}")
        print("="*60 + "\n")
    else:
        concat_dataset = results_df
        print("Notice: The dataset has no ground truth labels")
    
    # Save results
    dataset_name = Path(test_dataset.metadata_path).stem
    filename = f"Evaluate_results-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{model_name}-{dataset_name}.csv"
    output_file = output_path / filename
    concat_dataset.to_csv(index=False, path_or_buf=output_file)
    
    return output_file, metrics_dict


@click.command()
@click.option("--config", type=str, default="soundbay/conf/runs/main.yaml", help="Path to configuration YAML file")
@click.option("--checkpoint", required=True, help="Path to model checkpoint file")
@click.option("--metadata", "metadata_path", default=None, help="Path to metadata CSV file with labels")
@click.option("--data-path", "data_path", default=None, help="Path to directory containing audio files")
@click.option("--output", "output_path", default=None, help="Directory to save output files (default: ./outputs)")
@click.argument("overrides", nargs=-1)
def evaluate_main(
    config: Optional[str],
    checkpoint: str,
    metadata_path: Optional[str],
    data_path: Optional[str],
    output_path: Optional[str],
    overrides: tuple
) -> None:
    """
    Evaluate a trained model on labeled data.
    
    The checkpoint path is required and contains the trained model weights.
    Configuration can be provided via a YAML file (--config) and/or command-line overrides.
    
    Examples:
        python -m soundbay.evaluate --checkpoint model.pth --metadata annotations.csv --data-path audio/
        python -m soundbay.evaluate --config config.yaml --checkpoint model.pth
        python -m soundbay.evaluate --checkpoint model.pth data.data_sample_rate=2000
    """
    # Build overrides list
    override_list = list(overrides)
    if metadata_path:
        override_list.append(f"data.test_dataset.metadata_path={metadata_path}")
    if data_path:
        override_list.append(f"data.test_dataset.data_path={data_path}")
    
    # Create base config
    args = create_evaluate_config(config_path=config, overrides=override_list if override_list else None)
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    ckpt_args = ckpt_dict['args']
    ckpt_weights = ckpt_dict['model']
    
    # Merge config with checkpoint args
    args = merge_evaluate_config_with_checkpoint(args, ckpt_args)
    
    # Set output path
    if output_path:
        output_dirpath = Path(output_path)
    else:
        working_dirpath = Path(__file__).parent.parent
        output_dirpath = working_dirpath / "outputs"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    
    # Get model name from checkpoint path
    model_name = checkpoint_path.parent.stem
    
    # Run evaluation
    output_file, metrics = run_evaluation(
        args=args,
        checkpoint_state_dict=ckpt_weights,
        output_path=output_dirpath,
        model_name=model_name
    )
    
    print(f"Finished evaluation. Results saved to: {output_file}")


if __name__ == "__main__":
    evaluate_main()
