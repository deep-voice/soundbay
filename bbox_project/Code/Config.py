from dataclasses import dataclass, asdict
import dacite
import yaml
from pathlib import Path

# @dataclass
# class AugmentationsConfig:
#     time_masking: bool
#     frequency_masking: bool
#     time_stretching: bool
#     pitch_shifting: bool

@dataclass
class DatasetConfig:
    data_path: Path
    mode: str
    metadata_path: Path
    # augmentation_p: float
    # augmentations: AugmentationsConfig = None # start with no augmentations, we can add this later if needed
    # preprocessors: PreprocessorsConfig # I don't think we need this...
    margin_ratio: float
    add_random_margin: bool = True
    # seq_length: float

@dataclass
class DataConfig:
    label_names: list[str]
    batch_size: int
    wanted_sample_rate: int
    data_sample_rate: int
    n_fft: int
    hop_length: int
    label_type: str
    max_overlap_labels: int
    num_workers: int
    seq_length: float
    train_dataset: DatasetConfig
    val_dataset: DatasetConfig
    equalize_data: bool = True

@dataclass
class LossConfig:
    name: str
    lambda_coord: float
    lambda_obj: float
    lambda_noobj: float
    lambda_class: float
    
@dataclass
class ModelConfig:
    model_name: str
    num_classes: int
    criterion: LossConfig
    freeze_for_fine_tuning: bool = False
    initialize_detector: bool = False
    neg_bias: float = -2.0
    pooling_size: tuple[int, int] = (4, 8)
    
@dataclass
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float
    
@dataclass
class SchedulerConfig:
    name: str
    step_size: int
    gamma: float

@dataclass
class LearningConfig:
    epochs: int
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    freeze_layers_for_finetuning: bool = False
    
@dataclass
class LoggerConfig:
    name: str
    log_dir: Path
    confidence_threshold: float
    iou_threshold: float
    artifacts_upload_limit: int
    
@dataclass
class ExperimentConfig:
    project_name: str
    run_name: str
    run_id: str
    group_name: str
    resume: bool
    checkpoint_dir: Path
    logger: LoggerConfig
    debug: bool = False
    seed: int = 42
    log_interval: int = 10
    train_as_val_interval: int = 3

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    learning: LearningConfig
    experiment: ExperimentConfig
    
def load_config(config_path: Path):
    with open(config_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    conversion = {Path: lambda x: Path(x)}
    config = dacite.from_dict(data_class=Config, data=data, config=dacite.Config(type_hooks=conversion))
    return config

def stringify_paths(obj):
        if isinstance(obj, dict):
            return {k: stringify_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [stringify_paths(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

def save_config(config: Config, save_path: Path):
    config_dict = asdict(config)
    config_dict = stringify_paths(config_dict)
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    from pathlib import Path
    config_path = Path("Config/") / "test_config.yaml"
    config = load_config(config_path)
    print(config)