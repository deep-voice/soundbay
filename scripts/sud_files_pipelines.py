import os
import subprocess
import logging
import soundfile as sf
from pathlib import Path
from typing import List, Optional, Union, Generator, Dict, Iterable

import librosa
import pandas as pd
from boxsdk import OAuth2, Client
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import glob

from soundbay.inference import inference_main
from file_logger import FileLogger


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
        self.box_chunk_size = cfg.pipeline.box_chunk_size
        self.sud_folder = converter.sud_folder
        self.wav_folder = converter.wav_folder
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
            file_name = Path(file_path).stem + '.sud'
            try:
                # Load and resample audio
                audio, sr = librosa.load(file_path, sr=self.resample_rate)

                # Overwrite the original file
                sf.write(file_path, audio, self.resample_rate)
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

        for chunk in tqdm(chunks, desc="Processing BOX file chunks"):
            self._download_files(chunk)
            self.logger.info("Finished downloading BOX chunk")

            files_chunks = self._chunk_files(chunk, self.chunk_size)
            for files_chunk in tqdm(files_chunks, desc="Extracting SUD files"):
                self._process_files_chunk(files_chunk, files_mapping)
                if self.clean_wav:
                    self._clean_directory(self.wav_folder,
                                          [Path(str(f).replace('.sud', '.wav')) for f in files_chunk.keys()])
            self.logger.info("Finished processing box chunk")

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
            for chunk in tqdm(chunks, desc="Processing BOX chunks"):
                self._download_files(chunk)
                self.logger.info("Finished downloading BOX chunk")
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
        pipeline.process_and_run_predictions(files_mapping, process_files=cfg.pipeline.process_files)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    """
    from cli run:
        python scripts/sud_files_pipelines.py pipeline.mode=<predictions or wav> pipline.files_path=<path to csv file containing a "files" column>
    """
    main()