from typing import Union, Dict, Any
from collections.abc import Mapping
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from tqdm import tqdm

from soundbay.config import InferenceConfig, EvaluateConfig


def walk(input_path):
    """
    helper function to yield folder's file content
    Input:
        input_path: the path of the folder
    Output:
        generator of files in directory tree
    """
    for p in Path(input_path).iterdir():
        if p.is_dir():
            yield from walk(p)
            continue
        yield p.resolve()


def upload_experiment_to_s3(experiment_id: str,
                            dir_path: Path,
                            bucket_name: str,
                            include_parent: bool = True, logger=None):
    """
    Uploads the experiment folder to s3 bucket
    Input:
        experiment_id: id of the experiment, taken usually from wandb logger
        dir_path: path to the experiment directory
        bucket_name: name of the desired bucket path
        include_parent: flag to include the parent of the experiment folder while saving to s3
    """
    import boto3  # Import here to make it optional for inference
    
    assert dir_path.is_dir(), 'should upload experiments as directories to s3!'
    object_global = experiment_id
    current_global = str(dir_path.resolve())
    upload_files = list(walk(dir_path))
    s3_client = boto3.client('s3')
    for upload_file in tqdm(upload_files):
        upload_file = str(upload_file)
        s3_client.upload_file(upload_file, bucket_name, upload_file.replace(current_global, object_global))

    if logger is not None:
        print(f'experiment {logger.log_writer.run.id} has been successfully uploaded to {bucket_name} bucket')


def merge_with_checkpoint(run_args, checkpoint_args):
    """
    Merge into current args the needed arguments from checkpoint
    Right now we select the specific modules needed, can make it more generic if we'll see the need for it
    Input:
        run_args: dict_config of run args
        checkpoint_args: dict_config of checkpoint args
    Output:
        run_args: updated dict_config of run args
    """

    OmegaConf.set_struct(run_args, False)
    run_args.model = OmegaConf.to_container(checkpoint_args.model, resolve=True)
    run_args.data.test_dataset.preprocessors = OmegaConf.to_container(checkpoint_args.data.train_dataset.preprocessors, resolve=True)
    run_args.data.test_dataset.seq_length = checkpoint_args.data.train_dataset.seq_length
    run_args.data.test_dataset.sample_rate = checkpoint_args.data.train_dataset.sample_rate
    run_args.data.sample_rate = checkpoint_args.data.sample_rate
    run_args.data.n_fft = checkpoint_args.data.n_fft
    run_args.data.hop_length = checkpoint_args.data.hop_length
    run_args.data.label_type = checkpoint_args.data.get('label_type', 'single_label')  # using "get" for backward compatibility
    min_freq = checkpoint_args.data.get('min_freq', None)
    if min_freq is None:
        min_freq = checkpoint_args.data.min_freq_filtering
    run_args.data.min_freq = min_freq
    run_args.data.label_names = checkpoint_args.data.label_names
    OmegaConf.set_struct(run_args, True)
    return run_args


def _extract_model_module_name(model_target: str) -> str:
    """Extract model module name from _target_ string (e.g., 'models.EfficientNet2D' -> 'EfficientNet2D')"""
    return model_target.split('.')[-1]


def _get_checkpoint_value(ckpt_args: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Safely get a nested value from checkpoint args dict using dot notation.
    E.g., _get_checkpoint_value(args, 'data.train_dataset.seq_length', 1.0)
    """
    keys = key_path.split('.')
    value = ckpt_args
    try:
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                value = getattr(value, key, None)
            if value is None:
                return default
        return value
    except (KeyError, AttributeError, TypeError):
        return default


def merge_inference_config_with_checkpoint(
    inference_config: InferenceConfig,
    ckpt_args: Dict[str, Any]
) -> InferenceConfig:
    """
    Merge inference config with checkpoint args, extracting model and preprocessing settings.
    
    This function updates the inference config with values from the checkpoint that are
    necessary for inference (model architecture, preprocessing parameters, etc.).
    
    Input:
        inference_config: InferenceConfig dataclass with user-provided settings
        ckpt_args: Dictionary of checkpoint args (from ckpt_dict['args'])
    Output:
        Updated InferenceConfig with checkpoint values merged in
    """
    # Convert to OmegaConf for easier merging
    config_dict = OmegaConf.structured(inference_config)
    OmegaConf.set_struct(config_dict, False)
    
    # -------------------------------------------------------------------------
    # Extract model configuration
    # -------------------------------------------------------------------------
    model_config = _get_checkpoint_value(ckpt_args, 'model.model', {})
    if isinstance(model_config, (dict, DictConfig, Mapping)):
        model_target = model_config.get('_target_', '')
        module_name = _extract_model_module_name(model_target)
        num_classes = model_config.get('num_classes', 2)
        
        # Extract model params (everything except _target_ and num_classes)
        model_params = {k: v for k, v in model_config.items() 
                       if k not in ('_target_', 'num_classes')}
        
        config_dict.model.module_name = module_name
        config_dict.model.num_classes = num_classes
        config_dict.model.model_params = model_params
    
    # -------------------------------------------------------------------------
    # Extract data/preprocessing configuration
    # -------------------------------------------------------------------------
    # Sample rate (use checkpoint's training sample rate)
    sample_rate = _get_checkpoint_value(ckpt_args, 'data.sample_rate')
    if sample_rate is not None:
        config_dict.data.sample_rate = sample_rate
    
    # FFT parameters
    n_fft = _get_checkpoint_value(ckpt_args, 'data.n_fft')
    if n_fft is not None:
        config_dict.data.n_fft = n_fft
    
    hop_length = _get_checkpoint_value(ckpt_args, 'data.hop_length')
    if hop_length is not None:
        config_dict.data.hop_length = hop_length
    
    # Min frequency - handle backward compatibility
    min_freq = _get_checkpoint_value(ckpt_args, 'data.min_freq')
    if min_freq is None:
        min_freq = _get_checkpoint_value(ckpt_args, 'data.min_freq_filtering', 0)
    config_dict.data.min_freq = min_freq
    
    # n_mels - try to extract from preprocessors if available
    n_mels = _get_checkpoint_value(ckpt_args, '_preprocessors.mel_spectrogram.n_mels')
    if n_mels is not None:
        config_dict.data.n_mels = n_mels
    
    # Seq length from train dataset
    seq_length = _get_checkpoint_value(ckpt_args, 'data.train_dataset.seq_length')
    if seq_length is not None:
        config_dict.data.seq_length = float(seq_length)
    
    # Label configuration
    label_names = _get_checkpoint_value(ckpt_args, 'data.label_names')
    if label_names is not None:
        config_dict.data.label_names = list(label_names)
    
    label_type = _get_checkpoint_value(ckpt_args, 'data.label_type', 'single_label')
    config_dict.data.label_type = label_type
    
    OmegaConf.set_struct(config_dict, True)
    
    # Convert back to InferenceConfig dataclass
    return OmegaConf.to_object(config_dict)


def merge_evaluate_config_with_checkpoint(
    evaluate_config: EvaluateConfig,
    ckpt_args: Dict[str, Any]
) -> EvaluateConfig:
    """
    Merge evaluate config with checkpoint args, extracting model and preprocessing settings.
    
    This function updates the evaluate config with values from the checkpoint that are
    necessary for evaluation (model architecture, preprocessing parameters, etc.).
    
    Input:
        evaluate_config: EvaluateConfig dataclass with user-provided settings
        ckpt_args: Dictionary of checkpoint args (from ckpt_dict['args'])
    Output:
        Updated EvaluateConfig with checkpoint values merged in
    """
    # Convert to OmegaConf for easier merging
    config_dict = OmegaConf.structured(evaluate_config)
    OmegaConf.set_struct(config_dict, False)
    
    # -------------------------------------------------------------------------
    # Extract model configuration
    # -------------------------------------------------------------------------
    model_config = _get_checkpoint_value(ckpt_args, 'model.model', {})
    if isinstance(model_config, (dict, DictConfig, Mapping)):
        model_target = model_config.get('_target_', '')
        module_name = _extract_model_module_name(model_target)
        num_classes = model_config.get('num_classes', 2)
        
        # Extract model params (everything except _target_ and num_classes)
        model_params = {k: v for k, v in model_config.items() 
                       if k not in ('_target_', 'num_classes')}
        
        config_dict.model.module_name = module_name
        config_dict.model.num_classes = num_classes
        config_dict.model.model_params = model_params
    
    # -------------------------------------------------------------------------
    # Extract data/preprocessing configuration
    # -------------------------------------------------------------------------
    # Sample rate (use checkpoint's training sample rate)
    sample_rate = _get_checkpoint_value(ckpt_args, 'data.sample_rate')
    if sample_rate is not None:
        config_dict.data.sample_rate = sample_rate
    
    # FFT parameters
    n_fft = _get_checkpoint_value(ckpt_args, 'data.n_fft')
    if n_fft is not None:
        config_dict.data.n_fft = n_fft
    
    hop_length = _get_checkpoint_value(ckpt_args, 'data.hop_length')
    if hop_length is not None:
        config_dict.data.hop_length = hop_length
    
    # Min frequency - handle backward compatibility
    min_freq = _get_checkpoint_value(ckpt_args, 'data.min_freq')
    if min_freq is None:
        min_freq = _get_checkpoint_value(ckpt_args, 'data.min_freq_filtering', 0)
    config_dict.data.min_freq = min_freq
    
    # n_mels - try to extract from preprocessors if available
    n_mels = _get_checkpoint_value(ckpt_args, '_preprocessors.mel_spectrogram.n_mels')
    if n_mels is not None:
        config_dict.data.n_mels = n_mels
    
    # Seq length from train dataset
    seq_length = _get_checkpoint_value(ckpt_args, 'data.train_dataset.seq_length')
    if seq_length is not None:
        config_dict.data.seq_length = float(seq_length)
    
    # Label configuration
    label_names = _get_checkpoint_value(ckpt_args, 'data.label_names')
    if label_names is not None:
        config_dict.data.label_names = list(label_names)
    
    label_type = _get_checkpoint_value(ckpt_args, 'data.label_type', 'single_label')
    config_dict.data.label_type = label_type
    
    OmegaConf.set_struct(config_dict, True)
    
    # Convert back to EvaluateConfig dataclass
    return OmegaConf.to_object(config_dict)
