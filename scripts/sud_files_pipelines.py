import os
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Union, Generator, Dict

import pandas as pd
from boxsdk import OAuth2, Client
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import glob

from soundbay.inference import inference_main
from file_logger import FileLogger


def upload_file_to_s3(local_path: Path, remote_path: str, bucket: str) -> None:
    subprocess.check_call(
            f'aws s3 cp {local_path} s3://{bucket}/{remote_path}',
            shell=True
        )

def upload_directory_to_s3(local_dir: Path, remote_dir: str, bucket: str) -> None:
    subprocess.check_call(
            f'aws s3 sync {local_dir} s3://{bucket}/{remote_dir}/',
            shell=True
        )


class BoxClient:
    def __init__(self, client_id: str, client_secret: str, folder_id: str, access_token: str, refresh_token: Union[str, None] = None):
        self.client = self._setup_client(client_id, client_secret, access_token, refresh_token)
        self.folder_id = folder_id
        self.logger = logging.getLogger(__name__)

    def _setup_client(self,
                      client_id: str,
                      client_secret: str,
                      access_token: str,
                      refresh_token) -> Client:
        """Initialize Box API client."""
        auth = OAuth2(
            client_id=client_id,
            client_secret=client_secret,
            access_token=access_token,
            refresh_token=refresh_token
        )
        return Client(auth)

    def get_files_map(self, folder_id: str, files_list: List[str]) -> dict:
        """Get files ids from Box folder."""
        self.logger.info("Getting files ids from Box")
        items = self.client.folder(folder_id).get_items(limit=None)
        return {item.name: item.id for item in items if item.name in files_list}

    def download_file(self, file_id: str, file_name: str, output_dir: Path) -> None:
        """Download a single file from Box."""
        output_path = output_dir / file_name
        with open(output_path, 'wb') as f:
            self.client.file(file_id).download_to(f)


class Sud2WavConverter:
    def __init__(self,
                 sud_folder: Path,
                 wav_folder: Path,
                 sud2wav_path: Optional[str] = None,
                 s3_bucket: Optional[str] = None):
        self.sud_folder = sud_folder
        self.wav_folder = wav_folder
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

        if s3_save and self.s3_bucket:
            self.logger.info("Uploading .wav files to s3")
            upload_directory_to_s3(self.wav_folder, "", self.s3_bucket)
            self.logger.info("Finished uploading .wav files to s3")


class ProcessingPipeline:
    def __init__(self,
                 box_client: BoxClient,
                 converter: Sud2WavConverter,
                 file_logger: FileLogger,
                 cfg: DictConfig):
        self.cfg = cfg
        self.box_client = box_client
        self.converter = converter
        self.file_logger = file_logger
        self.chunk_size = cfg.pipeline.chunk_size
        self.sud_folder = converter.sud_folder
        self.wav_folder = converter.wav_folder
        self.clean_sud = cfg.pipeline.delete_sud_files
        self.clean_wav = cfg.pipeline.delete_wav_files
        self.logger = logging.getLogger(__name__)

    def _clean_directory(self, directory: Path) -> None:
        """Safely clean a directory."""
        self.logger.info(f"Cleaning directory {directory}")
        try:
            for file in directory.glob("*"):
                file.unlink()
        except Exception as e:
            self.logger.error(f"Failed to clean directory {directory}: {str(e)}")

    def _download_files(self, files_chunk: dict) -> None:
        """Download a single chunk of files."""
        for file_name, file_id in files_chunk.items():
            self.logger.info(f"Processing file: {file_name}")
            try:
                self.box_client.download_file(file_id, file_name, self.sud_folder)
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
        except Exception as e:
            self.logger.error(f"Error converting files: {e}")
            for file_name, file_id in files_chunk.items():
                self.file_logger.log_file_event(file_id, file_name, str(e), "conversion")

    def process_chunk(self, files_chunk: dict) -> None:
        """Process a single chunk of files."""
        self._download_files(files_chunk)
        self._convert_files(files_chunk)

    def _chunk_files(self, files: Dict[str, str]) -> Generator[Dict[str, str], None, None]:
        """Split files into chunks."""
        items = list(files.items())
        for i in range(0, len(items), self.chunk_size):
            yield dict(items[i:i + min(i + self.chunk_size, len(items))])

    def process_files(self, files_mapping) -> None:
        """Process all files in chunks."""
        chunks = self._chunk_files(files_mapping)
        self._clean_directory(self.wav_folder)

        for chunk in tqdm(chunks, desc="Processing file chunks"):
            self.process_chunk(chunk)
            self.logger.info("Finished processing chunk")

            if self.clean_sud:
                self._clean_directory(self.sud_folder)
            if self.clean_wav:
                self._clean_directory(self.wav_folder)
            self.file_logger.save_and_upload()

    def run_predictions(self, files_mapping, process_files=True) -> None:
        """Process files and run predictions."""
        chunks = self._chunk_files(files_mapping)
        outputs_path = Path(hydra.utils.get_original_cwd()).parent.absolute() / "outputs"

        for chunk in tqdm(chunks, desc="Processing and predicting"):
            if process_files:
                self.process_chunk(chunk)

            for wav_file in glob.glob(f"{self.wav_folder}/*.wav"):
                self.cfg.data.test_dataset.file_path = wav_file
                file_name = Path(wav_file).stem + '.sud'
                try:
                    inference_main(self.cfg)
                    self.file_logger.log_file_event(files_mapping[file_name], file_name, "success", "prediction")
                except Exception as e:
                    self.file_logger.log_file_event(files_mapping[file_name], file_name, str(e), "prediction")

                if self.cfg.pipeline.s3_bucket:
                    self.logger.info("Uploading predictions to s3")
                    upload_directory_to_s3(outputs_path, "predictions", self.cfg.pipeline.s3_bucket)
                self.file_logger.save_and_upload()

            if self.clean_sud:
                self._clean_directory(self.sud_folder)
            self._clean_directory(self.wav_folder)


@hydra.main(version_base="1.2", config_path="../soundbay/conf", config_name="runs/main_pipeline")
def main(cfg: DictConfig) -> None:
    """CLI entry point."""
    sud_folder = Path(cfg.pipeline.sud_folder if cfg.pipeline.sud_folder else os.path.expanduser("~") + '/SUD2WAV/input_files')
    wav_folder = Path(cfg.pipeline.wav_folder if cfg.pipeline.wav_folder else os.path.expanduser("~") + '/SUD2WAV/output_files')
    sud_folder.mkdir(parents=True, exist_ok=True)
    wav_folder.mkdir(parents=True, exist_ok=True)

    box_client = BoxClient(**cfg.box)
    file_logger = FileLogger(cfg.pipeline.log_folder, cfg.pipeline.log_filename, cfg.pipeline.s3_bucket)
    converter = Sud2WavConverter(sud_folder, wav_folder, cfg.pipeline.sud2wav_path, cfg.pipeline.s3_bucket)
    pipeline = ProcessingPipeline(box_client=box_client, converter=converter, file_logger=file_logger, cfg=cfg)

    files_list = pd.read_csv(cfg.pipeline.files_path)['files'].unique().tolist()
    files_mapping = box_client.get_files_map(cfg.box.folder_id, files_list)

    if cfg.pipeline.mode == "wav":
        pipeline.process_files(files_mapping)
    elif cfg.pipeline.mode == "predictions":
        pipeline.run_predictions(files_mapping, process_files=cfg.pipeline.process_files)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    """
    from cli run:
        python scripts/sud_files_pipelines.py pipeline.mode=<predictions or wav> pipline.files_path=<path to csv file containing a "files" column>
    """
    main()