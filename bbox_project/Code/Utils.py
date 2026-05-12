import numpy as np
import random
import torch
from torch.utils.data import WeightedRandomSampler
from Code.data import MultiCallDetectionDataset
from Code.Config import ModelConfig, OptimizerConfig, SchedulerConfig, DatasetConfig, DataConfig, LossConfig
from Code.Models import FlexibleTinyDetector, initialize_detector, DeepSpectrogramDetector, GlobalDetectorLongerTime
from Code.Losses import DetectionLoss
from tqdm import tqdm
    

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataset(dataset_config: DatasetConfig, 
                data_config: DataConfig, 
                n_classes: int):
    # train will be used if augmentations would be added later for train as val dataset, but for now it doesn't matter since we don't have augmentations
    return MultiCallDetectionDataset(
        data_path=dataset_config.data_path,
        metadata_path=dataset_config.metadata_path,
        augmentations=None,     # we can add this later if needed
        augmentations_p=0.0,    # we can add this later if needed
        preprocessors=[],       # we can add this later if needed
        seq_length=data_config.seq_length,
        n_classes=n_classes,
        orig_sample_rate=data_config.data_sample_rate,
        wanted_sample_rate=data_config.wanted_sample_rate,
        max_overlap_labels=data_config.max_overlap_labels,
        margin_ratio=dataset_config.margin_ratio,
        add_random_margin=dataset_config.add_random_margin,
        n_fft=data_config.n_fft,
        hop_length=data_config.hop_length,
        label_type=data_config.label_type,
    )

def get_model(model_config: ModelConfig, data_config: DataConfig, 
              n_classes: int, pooling_size: tuple[int, int] = (4, 8), verbose_weight=False):
    if model_config.model_name == "flexible_tiny_detector":
        model = FlexibleTinyDetector(max_boxes=data_config.max_overlap_labels, n_classes=n_classes, pooling_size=model_config.pooling_size)
    elif model_config.model_name == "tiny_detector":
        model = DeepSpectrogramDetector(max_boxes=data_config.max_overlap_labels, n_classes=n_classes, pooling_size=model_config.pooling_size)
        model.init_weights(neg_bias=model_config.neg_bias)
    elif model_config.model_name == "global_detector_longer_time":
        model = GlobalDetectorLongerTime(max_boxes=data_config.max_overlap_labels, n_classes=n_classes, pooling_size=model_config.pooling_size)
        model.init_weights(neg_bias=model_config.neg_bias)
    else:
        raise NotImplementedError(f"Model {model_config.model_name} not implemented yet.")
    

    if verbose_weight:
        for name, param in model.named_parameters():
            if 'weight' in name:
                print(f"{name:30} | mean: {param.data.mean():.4f} | std: {param.data.std():.4f}")

    if model_config.initialize_detector:
        initialize_detector(model, neg_bias=model_config.neg_bias)
    return model

def get_criterion(criterion_config: LossConfig):
    # if criterion_name == "mse":
    #     return torch.nn.MSELoss()
    # elif criterion_name == "bce_with_logits":
    #     return torch.nn.BCEWithLogitsLoss()
    # else:
    #     raise NotImplementedError(f"Criterion {criterion_name} not implemented yet.")
    if criterion_config.name == "detection_loss":
        return DetectionLoss(
            lambda_coord=criterion_config.lambda_coord, lambda_obj=criterion_config.lambda_obj, lambda_noobj=criterion_config.lambda_noobj, lambda_class=criterion_config.lambda_class)
    else:
        raise NotImplementedError(f"Criterion {criterion_config.name} not implemented yet.")

def get_optimizer(model_parameters, optimizer_config):
    if optimizer_config.name == "adam":
        return torch.optim.Adam(model_parameters, lr=optimizer_config.learning_rate, weight_decay=optimizer_config.weight_decay)
    elif optimizer_config.name == "sgd":
        return torch.optim.SGD(model_parameters, lr=optimizer_config.learning_rate, weight_decay=optimizer_config.weight_decay)
    elif optimizer_config.name == "adamw":
        return torch.optim.AdamW(model_parameters, lr=optimizer_config.learning_rate, weight_decay=optimizer_config.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {optimizer_config.name} not implemented yet.")

def get_scheduler(optimizer, scheduler_config):
    if scheduler_config.name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_config.step_size, gamma=scheduler_config.gamma)
    elif scheduler_config.name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_config.step_size, eta_min=1e-6, last_epoch=-1)
    else:
        raise NotImplementedError(f"Scheduler {scheduler_config.name} not implemented yet.")

def get_weighted_sampler(dataset):
    print("Pre-scanning dataset labels to balance classes...")
    all_sample_priority_classes = []

    # We only iterate over the labels to save time
    all_labels = dataset.get_all_labels()
    for label_tensor in tqdm(dataset.get_all_labels(), desc="Scanning dataset for class distribution"):
        # We use dataset.labels if your class stores them in a list, 
        # otherwise we call the dataset index.
        # This assumes dataset[i] returns (image, label_tensor)
        # _, label_tensor = dataset[i] 
        
        # label_tensor shape: [4, 7] -> (x, y, w, h, class0, class1, conf)
        confidences = label_tensor[:, 6]
        class_0_cols = label_tensor[:, 4]
        class_1_cols = label_tensor[:, 5]

        # Priority Logic:
        if torch.any((confidences == 1) & (class_1_cols == 1)):
            # If the image contains the 'Rare' class at all
            priority = 1 
        elif torch.any((confidences == 1) & (class_0_cols == 1)):
            # If it only contains the 'Common' class
            priority = 0
        else:
            # It's an empty background sample
            priority = 2
            
        all_sample_priority_classes.append(priority)

    # Convert to tensor for math
    priority_tensor = torch.tensor(all_sample_priority_classes)
    
    # Calculate counts for each of the 3 types
    class_counts = torch.bincount(priority_tensor, minlength=3)
    print(f"Detected counts: Rare: {class_counts[0]}, Common: {class_counts[1]}, Empty: {class_counts[2]}")

    # Calculate weights (Inverse of frequency)
    # Add a small epsilon to avoid division by zero if a class is missing
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    
    # Map the weights back to every individual sample
    sample_weights = class_weights[priority_tensor]

    # Create the sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True # Crucial: allows rare samples to be picked multiple times per epoch
    )
    
    return sampler

def get_sampler(equalize_data: bool, train_dataset):
    if equalize_data:
        # sample_weights = create_sampler(train_dataset)
        # sampler = WeightedRandomSampler(
        #     weights=sample_weights, 
        #     num_samples=len(sample_weights), 
        #     replacement=True
        # )
        sampler = get_weighted_sampler(train_dataset)
    else:
        sampler = None
    return sampler