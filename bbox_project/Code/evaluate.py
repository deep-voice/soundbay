import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from Code.Metrics import MetricsCalculator2Detection
from Code.data import MultiCallDetectionDataset
from Code.inference import get_model_and_config
from Code.Config import Config, DatasetConfig
from Code.Utils import get_dataset, get_model

def get_dataloader(dataset_config, data_config, n_classes, 
                   batch_size=32, shuffle=False):
    dataset = get_dataset(dataset_config, data_config, n_classes)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    return dataloader

def evaluate_model(dataloader: DataLoader, model: torch.nn.Module, metrics_type: str, device=torch.device('cuda'), conf_threshold=0.5, iou_threshold=0.5):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing Model"):
            features, label = batch
            features, label = features.to(device), label.to(device)

            pred = model(features)
            all_preds.append(pred.cpu())
            all_targets.append(label.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if metrics_type == "2D":
        metrics_calc = MetricsCalculator2Detection(all_targets, all_preds, confidence_threshold=conf_threshold, iou_threshold=iou_threshold)
        metrics = metrics_calc.calc_all_metrics()
    else:
        raise NotImplementedError(f"Metrics type {metrics_type} not implemented yet.")

    print("Test Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value}")

    return metrics

if __name__ == "__main__":
    from pprint import pprint

    audio_dir = Path('/mnt/d/DeepVoice/soundbay/datasets/fannie_project/')
    val_csv = Path('/mnt/d/DeepVoice/soundbay/datasets/fannie_project/val_updated.csv')
    test_csv = Path('/mnt/d/DeepVoice/soundbay/datasets/fannie_project/test_updated.csv')
    
    print("--- Evaluating models on test set: ---")
    for checkpoint_dir in [
        Path("Checkpoints/2d_detector_15_sec_seq_length"),
        Path("Checkpoints/2d_detector_20_sec_seq_length_second_run"),
        Path("Checkpoints/fourth_2d_run")
        ]:
        print(f"\nEvaluating model from checkpoint: {checkpoint_dir}")
    # chechpoint_dir = Path("Checkpoints/fourth_2d_run")
        model, config = get_model_and_config(checkpoint_dir / "last.pth", checkpoint_dir / "config.yaml")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        test_dataset = DatasetConfig(
            data_path=audio_dir,
            mode="test",
            metadata_path=test_csv,
            margin_ratio = 0.05,
            add_random_margin=False
        )

        dataloader = get_dataloader(test_dataset, config.data, config.model.num_classes, batch_size=64, shuffle=False)
        metrics = evaluate_model(dataloader, model, 
                                metrics_type="2D", device=device,
                                conf_threshold=0.95, iou_threshold=0.05)
        
        pprint(metrics)
