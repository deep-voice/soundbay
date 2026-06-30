"""YOLO training wrapper for humpback call detection."""
import argparse
import yaml
from ultralytics import YOLO


def train(dataset_yaml: str, config_yaml: str, resume: str = None):
    """Train YOLOv11 model on humpback spectrogram dataset."""
    with open(config_yaml) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.pop("model", "yolo11m.pt")

    if resume:
        model = YOLO(resume)
        model.train(resume=True)
    else:
        model = YOLO(model_name)
        model.train(data=dataset_yaml, **cfg)


def main():
    parser = argparse.ArgumentParser(description="Train YOLO humpback detector")
    parser.add_argument("--dataset", required=True, help="Path to dataset YAML")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(args.dataset, args.config, args.resume)


if __name__ == "__main__":
    main()
