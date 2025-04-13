import os
import subprocess
import logging
from typing import List, Optional, Generator, Dict, Iterable, Union
import pandas as pd
import boto3
from box_sdk_gen import BoxClient, BoxCCGAuth, CCGConfig
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import glob
from pathlib import Path

from soundbay.inference import inference_main
from file_logger import FileLogger
from subprocess import check_call


class S3Client:
    def __init__(self, bucket: str, prefix: str = ""):
        self.s3_client = boto3.client('s3')
        self.bucket = bucket
        self.prefix = prefix
        self.logger = logging.getLogger(__name__)

    def get_files_map(self, files_list: List[str] = None) -> dict:
        """Get files from S3 bucket."""
        self.logger.info("Getting files from S3")
        files_map = {}
        
        # List objects in the bucket with the given prefix
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                file_name = os.path.basename(obj['Key'])
                if file_name.endswith('.sud'):  # Only include .sud files
                    files_map[file_name] = obj['Key']
        
        if files_list is not None:
            files_map = {k: v for k, v in files_map.items() if k in files_list}
        return files_map

    def download_file(self, file_key: str, file_name: str, output_dir: Path) -> None:
        """Download a single file from S3."""
        output_path = output_dir / file_name
        self.s3_client.download_file(self.bucket, file_key, str(output_path))


class BoxLocalClient:
    def __init__(self, client_id: str, client_secret: str, folder_id: str, user_id:str):
        self.client = self._setup_client(client_id, client_secret, user_id)
        self.folder_id = folder_id
        self.logger = logging.getLogger(__name__)

    def _setup_client(self,
                      client_id: str,
                      client_secret: str,
                      user_id: str) -> BoxClient:
        """Initialize Box API client."""
        ccg_config = CCGConfig(
            client_id=client_id,
            client_secret=client_secret,
            user_id=user_id
        )
        auth = BoxCCGAuth(config=ccg_config)

        return BoxClient(auth=auth)

    def get_files_map(self, folder_id: str, files_list: List[str]) -> dict:
        """Get files ids from Box folder."""
        self.logger.info("Getting files ids from Box")
        total_count = self.client.folders.get_folder_items(folder_id).to_dict()['total_count']
        files_map = {item['name']: item['id'] for i in range(0, total_count, min(total_count, 100)) for item in
                self.client.folders.get_folder_items(folder_id, offset=i).to_dict()['entries']
                 }
        if files_list is not None:
            files_map = {k:v for k,v in files_map.items() if k in files_list}
        return files_map

    def download_file(self, file_id: str, file_name: str, output_dir: Path) -> None:
        """Download a single file from Box."""
        output_path = output_dir / file_name
        with open(output_path, 'wb') as f:
            self.client.downloads.download_file_to_output_stream(file_id, f)


class Sud2WavConverter:
    def __init__(self,
                 sud_folder: Path,
                 wav_folder: Path,
                 wav_backup_folder: Path,
                 sud2wav_path: Optional[str] = None,
                 s3_bucket: Optional[str] = None):
        self.sud_folder = sud_folder
        self.wav_folder = wav_folder
        self.wav_backup_folder = wav_backup_folder
        self.s3_bucket = s3_bucket
        self.sud2wav_path = sud2wav_path or os.path.expanduser("~") + '/SUD2WAV'
        self.logger = logging.getLogger(__name__)

    def convert_to_wav(self, s3_save: bool) -> None:
        """Convert .sud files to .wav using Docker container."""
        with open("/tmp/output.log", "a") as output:
            self.logger.info("Extracting SUD files")
            cmd = (
                f"docker build -t converter:latest -f {self.sud2wav_path}/images/gh_action_files/Dockerfile . && "
                f"docker run -t --rm -v {self.sud_folder}:/workspace/input_sud_files "
                f"-v {self.wav_folder}:/workspace/output_wav_files converter ./scripts/convert_files.sh"
            )
            subprocess.check_call(cmd, shell=True, stdout=output, stderr=output, cwd=self.sud2wav_path)
            self.logger.info("Finished extracting SUD files")


class ProcessingPipeline:
    def __init__(self,
                 source_client: Union[BoxLocalClient, S3Client],
                 converter: Sud2WavConverter,
                 file_logger: FileLogger,
                 cfg: DictConfig):
        self.cfg = cfg
        self.source_client = source_client
        self.converter = converter
        self.file_logger = file_logger
        self.chunk_size = cfg.pipeline.chunk_size
        self.box_chunk_size = cfg.pipeline.box_chunk_size
        self.sud_folder = converter.sud_folder
        self.wav_folder = converter.wav_folder
        self.wav_backup_folder = converter.wav_backup_folder
        self.clean_wav = cfg.pipeline.delete_wav_files
        self.resample_rate = cfg.pipeline.resample_rate
        self.logger = logging.getLogger(__name__)
        self.s3_bucket = cfg.pipeline.s3_bucket
        self.s3_save = cfg.pipeline.s3_save

    def _clean_directory(self, directory: Path, files_list = None) -> None:
        """Safely clean a directory."""
        self.logger.info(f"Cleaning directory {directory}")
        try:
            if files_list is None:
                remove_list = directory.glob("*")
            else:
                remove_list = [directory / file for file in files_list]
            for file in remove_list:
                file.unlink()
        except Exception as e:
            self.logger.error(f"Failed to clean directory {directory}: {str(e)}")

    def _move_directory(self, directory: Path, target_dir: Path, files_list = None) -> None:
        """Safely clean a directory."""
        self.logger.info(f"move directory {directory} to {target_dir}")
        try:
            if files_list is None:
                remove_list = directory.glob("*")
            else:
                remove_list = [directory / file for file in files_list]
            for file in remove_list:
                file.rename(target_dir / Path(file.name))
        except Exception as e:
            self.logger.error(f"Failed to clean directory {directory}: {str(e)}")

    def _move_chunk_files(self, files: Iterable, destination: Path) -> None:
        """Move files to a new directory."""
        self.converter.sud_folder = destination
        self.converter.sud_folder.mkdir(parents=True, exist_ok=True)
        for file in files:
            file_path = self.sud_folder / file
            file_path.rename(destination / file)

    def _download_files(self, files_chunk: dict) -> None:
        """Download a single chunk of files."""
        for file_name, file_id in files_chunk.items():
            self.logger.info(f"Processing file: {file_name}")
            try:
                self.source_client.download_file(file_id, file_name, self.sud_folder)
                self.file_logger.log_file_event(file_id, file_name, "success", "download")
            except Exception as e:
                self.file_logger.log_file_event(file_id, file_name, str(e), "download")
                continue

    def _convert_files(self, files_chunk: dict) -> None:
        """Convert a single chunk of files from .sud to .wav."""
        try:
            self.converter.convert_to_wav(self.cfg.pipeline.s3_save)
            for file_name, file_id in files_chunk.items():
                self.file_logger.log_file_event(file_id, file_name, "success", "conversion")
            self.logger.info("Finished converting current file chunk")
            self._clean_directory(self.converter.sud_folder, files_chunk.keys())
        except Exception as e:
            self.logger.error(f"Error converting files: {e}")
            for file_name, file_id in files_chunk.items():
                self.file_logger.log_file_event(file_id, file_name, str(e), "conversion")
        finally:
            self.file_logger.save_and_upload()

    def _resample_wav_files_inplace(self, files_mapping: dict):
        """Resamples a list of .wav files to the specified sample rate and overwrites them."""
        for file_path in glob.glob(f"{self.wav_folder}/*.wav"):
            file_path = Path(file_path)
            file_name = file_path.stem + '.sud'
            try:
                # Load and resample audio
                file_path_tmp = file_path.parent / f"{file_path.stem}_tmp.wav"
                file_path.rename(file_path_tmp)
                check_call(['sox', str(file_path_tmp), str(file_path), 'rate', str(self.resample_rate)])
                file_path_tmp.unlink()
                self.logger.info(f"Resampled {file_path} to {self.resample_rate} Hz")
                self.file_logger.log_file_event(files_mapping[file_name], file_name, "success", "resample")
            except Exception as e:
                self.logger.error(f"Error resampling {file_path}: {e}")
                self.file_logger.log_file_event(files_mapping[file_name], file_name, str(e), "resample")

    def _chunk_files(self, files: Dict[str, str], chunk_size: int) -> Generator[Dict[str, str], None, None]:
        """Split files into chunks."""
        items = list(files.items())
        for i in range(0, len(items), chunk_size):
            yield dict(items[i:min(i + chunk_size, len(items))])

    def _upload_directory_to_s3(self, local_dir: Path, remote_dir: str, bucket: str, additional_syntax: str = "") -> None:
        self.logger.info("Uploading files to s3")
        if len(remote_dir) > 0:
            bucket = f"{bucket}/{remote_dir}"
        subprocess.check_call(
            f'aws s3 sync {local_dir} s3://{bucket}/ {additional_syntax}',
            shell=True
        )
        self.logger.info("Finished uploading files to s3")

    def _process_files_chunk(self, files_chunk: dict, files_mapping: dict):
        """
        Process a chunk of files.
        """
        self._move_chunk_files(files_chunk.keys(), self.sud_folder / "chunk")
        self._convert_files(files_chunk)

        if self.resample_rate is not None:
            self._resample_wav_files_inplace(files_mapping)

        if self.s3_save and self.s3_bucket:
            self._upload_directory_to_s3(self.wav_folder, "", self.s3_bucket,
                                         additional_syntax='--exclude "*" --include "*.wav"')
        self.file_logger.save_and_upload()

    def process_files(self, files_mapping) -> None:
        """Process all files in chunks."""
        chunks = self._chunk_files(files_mapping, self.box_chunk_size)

        for chunk in tqdm(chunks, desc="Processing file chunks"):
            self._download_files(chunk)
            self.logger.info("Finished downloading chunk")

            files_chunks = self._chunk_files(chunk, self.chunk_size)
            for files_chunk in tqdm(files_chunks, desc="Extracting SUD files"):
                self._process_files_chunk(files_chunk, files_mapping)
                files_to_postprocess = [Path(str(f).replace('.sud', '.wav')) for f in files_chunk.keys()]
                if self.clean_wav:
                    self._clean_directory(self.wav_folder, files_to_postprocess)
                else:
                    self._move_directory(self.wav_folder, self.wav_backup_folder, files_to_postprocess)
            self.logger.info("Finished processing chunk")

    def run_predictions(self, files_mapping, outputs_path) -> None:
        """Run predictions on all files in chunks."""
        for wav_file in glob.glob(f"{self.wav_folder}/*.wav"):
            self.cfg.data.test_dataset.file_path = wav_file
            file_name = Path(wav_file).stem + '.sud'
            try:
                inference_main(self.cfg)
                self.file_logger.log_file_event(files_mapping[file_name], file_name, "success", "prediction")
                self._clean_directory(self.wav_folder, [wav_file.split('/')[-1]])
            except Exception as e:
                self.file_logger.log_file_event(files_mapping[file_name], file_name, str(e), "prediction")

            if self.s3_bucket:
                self._upload_directory_to_s3(outputs_path, "predictions", self.s3_bucket)
            self.file_logger.save_and_upload()

    def process_and_run_predictions(self, files_mapping, process_files=True) -> None:
        """Process files and run predictions."""
        chunks = self._chunk_files(files_mapping, self.box_chunk_size)
        outputs_path = Path(hydra.utils.get_original_cwd()).parent.absolute() / "outputs"

        if not process_files:
            self.run_predictions(files_mapping, outputs_path)
        else:
            for chunk in tqdm(chunks, desc="Processing chunks"):
                self._download_files(chunk)
                self.logger.info("Finished downloading chunk")
                files_chunks = self._chunk_files(chunk, self.chunk_size)
                for files_chunk in tqdm(files_chunks, desc="Processing and predicting files"):
                    self._process_files_chunk(files_chunk, files_mapping)
                    self.run_predictions(files_mapping, outputs_path)


@hydra.main(version_base="1.2", config_path="../soundbay/conf", config_name="runs/main_pipeline")
def main(cfg: DictConfig) -> None:
    """CLI entry point."""
    assert cfg.pipeline.box_chunk_size >= cfg.pipeline.chunk_size, "Box chunk size must be greater than or equal to processing chunk size"

    sud_folder = Path(cfg.pipeline.sud_folder if cfg.pipeline.sud_folder else os.path.expanduser("~") + '/SUD2WAV/input_files')
    wav_folder = Path(cfg.pipeline.wav_folder if cfg.pipeline.wav_folder else os.path.expanduser("~") + '/SUD2WAV/output_files')
    wav_backup_folder = Path(cfg.pipeline.wav_backup_folder if cfg.pipeline.wav_backup_folder else os.path.expanduser("~") + '/SUD2WAV/output_files_backup')
    sud_folder.mkdir(parents=True, exist_ok=True)
    wav_folder.mkdir(parents=True, exist_ok=True)
    wav_backup_folder.mkdir(parents=True, exist_ok=True)

    # Initialize the appropriate source client based on configuration
    if cfg.pipeline.source == "box":
        source_client = BoxLocalClient(**cfg.box)
    elif cfg.pipeline.source == "s3":
        source_client = S3Client(cfg.pipeline.s3_bucket, cfg.pipeline.s3_prefix)
    else:
        raise ValueError(f"Unknown source: {cfg.pipeline.source}")

    file_logger = FileLogger(cfg.pipeline.log_folder, cfg.pipeline.log_filename, cfg.pipeline.s3_bucket)
    converter = Sud2WavConverter(sud_folder, wav_folder, wav_backup_folder, cfg.pipeline.sud2wav_path, cfg.pipeline.s3_bucket)
    pipeline = ProcessingPipeline(source_client=source_client, converter=converter, file_logger=file_logger, cfg=cfg)

    if cfg.pipeline.files_path is not None:
        files_list = pd.read_csv(cfg.pipeline.files_path)['files'].unique().tolist()
    else:
        files_list = None
    files_mapping = source_client.get_files_map(files_list)

    if cfg.pipeline.mode == "wav":
        pipeline.process_files(files_mapping)
    elif cfg.pipeline.mode == "predictions":
        pipeline.process_and_run_predictions(files_mapping, process_files=cfg.pipeline.process_files)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    """
    from cli run:
        python scripts/sud_files_pipelines.py pipeline.mode=<predictions or wav> pipeline.source=<box or s3> pipeline.files_path=<path to csv file containing a "files" column>
    """
    main()
