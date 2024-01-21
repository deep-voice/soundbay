from typing import Union
import boto3
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm


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
    run_args.data.sample_rate = checkpoint_args.data.sample_rate
    run_args.data.n_fft = checkpoint_args.data.n_fft
    run_args.data.hop_length = checkpoint_args.data.hop_length
    run_args.data.min_freq = checkpoint_args.data.min_freq
    run_args.data.max_freq = checkpoint_args.data.max_freq
    OmegaConf.set_struct(run_args, True)
    return run_args
