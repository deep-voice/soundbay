from torch.utils.data import Dataset
import torch
import pandas as pd
import random
import soundfile as sf
from itertools import starmap, repeat
from torchvision import transforms
from data_augmentation import ChainedAugmentations
import torchaudio
from hydra.utils import instantiate, DictConfig
from pathlib import Path
from copy import deepcopy
import numpy as np

class ClassifierDataset(Dataset):
    '''
    class for storing and loading data.
    '''
    def __init__(self, data_path, metadata_path, augmentations, augmentations_p, preprocessors,
                 seq_length=1, len_buffer=0.1, data_sample_rate=44100, sample_rate=44100, mode="train",
                 equalize_data=False, slice_flag=False):
        """
        __init__ method initiates ClassifierDataset instance:
        Input:
        data_path - string
        metadata_path - string
        augmentations - list of classes (not instances) from data_augmentation.py
        augmentations_p - array of probabilities (float64)
        preprocessors - list of classes from preprocessors (TBD function)

        Output:
        ClassifierDataset Object - inherits from Dataset object in PyTorch package
        """
        self.audio_dict = self._create_audio_dict(Path(data_path))
        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(self.metadata_path)
        self.mode = mode
        self.seq_length = seq_length
        self.sample_rate = sample_rate
        self.data_sample_rate = data_sample_rate
        self.sampler = torchaudio.transforms.Resample(orig_freq=data_sample_rate, new_freq=sample_rate)
        self._preprocess_metadata(len_buffer, equalize_data, slice_flag)
        self.augmenter = self._set_augmentations(augmentations, augmentations_p)
        self.preprocessor = self.set_preprocessor(preprocessors)

    def _create_audio_dict(self, data_path: Path) -> dict:
        """
            create reference dict to extract audio files from metadata annotation
            Input:
            data_path - Path object
            Output:
            audio_dict contains references to audio paths given name from metadata
        """
        audio_paths = data_path.rglob('*.wav')
        return {x.name.strip('.wav'): x for x in audio_paths}

    def _preprocess_metadata(self, len_buffer, equalize=False, slice_flag=False):
        """
        function _preprocesses_metadata grabs calls with minimal length of self.seq_length + len_buffer
        Input:
        len_buffer - float64
        Output:
        ClassifierDataset object with self.metadata dataframe after applying the condition
        """

        def _equalize_distribution(df_object):
            len_pos = len(df_object[df_object['label'] == 1])
            len_neg = len(df_object[df_object['label'] == 0])
            diff = len_pos - len_neg
            if diff == 0:
                return df_object
            label = 0 if diff > 0 else 1
            multiplier = int(np.ceil(abs(diff) / min(len_neg, len_pos)))
            additive = pd.concat(deepcopy([self.metadata[self.metadata['label'] == label]]) * multiplier)
            additive = additive[:abs(diff)]
            df_object = pd.concat([df_object, additive])
            assert len(df_object[df_object['label'] == 1]) == len(df_object[df_object['label'] == 0])
            return df_object

        self.metadata = self.metadata[self.metadata['call_length'] > (self.seq_length + len_buffer)]
        if equalize:
            self.metadata = _equalize_distribution(self.metadata)
        if slice_flag:
            self._slice_sequence(len_buffer)

        self.metadata.reset_index(drop=True, inplace=True)

    def _grab_fields(self, idx):
        """
        grabs fields from metadata according to idx
        input :idx
        output: begin_time - start time of segment
                end_time - end time of segment
                path_to_file - full path to file
        """
        filename = self.metadata['filename'][idx]
        begin_time = self.metadata['begin_time'][idx]
        end_time = self.metadata['end_time'][idx]
        path_to_file = self.audio_dict[filename]
        orig_sample_rate = sf.info(path_to_file).samplerate
        assert orig_sample_rate == self.data_sample_rate
        begin_time = int(begin_time * orig_sample_rate)
        end_time = int(end_time * orig_sample_rate)
        label = self.metadata['label'][idx]
        return path_to_file, begin_time, end_time, label


    def _slice_sequence(self, len_buffer):
        """
        function _slice_sequence process metadata list call lengths to be sliced according to self.seq_length
        Input:
        len_buffer - float64
        self
        Output:
        self.metadata sliced according to buffers
        """
        self.metadata = self.metadata.reset_index(drop=True)
        sliced_times = list(starmap(np.arange, zip(self.metadata['begin_time'], self.metadata['end_time'], repeat(self.seq_length))))
        new_begin_time = list(x[:-1] for x in sliced_times)
        duplicate_size_vector = [len(list_elem) for list_elem in new_begin_time] # vector to duplicate original dataframe
        new_begin_time = np.concatenate(new_begin_time) # vectorize to array
        new_end_time = np.concatenate(list(x[1:] for x in sliced_times)) # same for end_times
        self.metadata = self.metadata.iloc[self.metadata.index.repeat(duplicate_size_vector)].reset_index(drop=True)
        self.metadata['begin_time'] = new_begin_time
        self.metadata['end_time'] = new_end_time
        self.metadata['call_length'] = np.shape(self.metadata)[0] * [self.seq_length]
        return

    def _get_audio(self, path_to_file, begin_time, end_time):
        """
        _get_audio gets a path_to_file from _grab_fields method and also begin_time and end_time
        and returns the audio segment in a torch.tensor

        input:
        path_to_file - string
        begin_time - int
        end_time - int

        output:
        audio - pytorch tensor (1-D array)
        """
        if self.mode == "train":
            start_time = random.randint(begin_time, end_time - self.seq_length * self.data_sample_rate)
        else:
            start_time = begin_time
        data, _ = sf.read(path_to_file, start=start_time, stop=start_time + self.seq_length * self.data_sample_rate)
        # TODO support in the future batching of long samples into short seq_len samples, might need this during inference
        audio = torch.tensor(data, dtype=torch.float).unsqueeze(0)
        return audio

    def _set_augmentations(self, augmentations_dict, augmentations_p):
        """
        get augmentations list and instantiate - TBD
        """
        if augmentations_dict is not None:
            augmentations_list = [instantiate(args) for args in augmentations_dict.values()]
        else:
            augmentations_list = []
        augmenter = ChainedAugmentations(augmentations_list, augmentations_p) if self.mode == 'train' else torch.nn.Identity()
        return augmenter

    @staticmethod
    def set_preprocessor(preprocessors_args):
        """
        function set_preprocessor takes preprocessors_args as an argument and creates a preprocessor object
        to be applied later on the audio segment

        input:
        preprocessors_args - list of classes from torchvision

        output:
        preprocessor - Composes several transforms together (transforms object)
        """
        if len(preprocessors_args) > 0:
            processors_list = [instantiate(args) for args in preprocessors_args.values()]
            preprocessor = transforms.Compose(processors_list)
        else:
            preprocessor = torch.nn.Identity()
        return preprocessor

    def __getitem__(self, idx):
        '''
        __getitem__ method loads item according to idx from the metadata

        input:
        idx - int

        output:
        audio, label - torch tensor (1-d if no spectrogram is applied/ 2-d if applied a spectrogram
        , int (if mode="train" only)
        '''
        path_to_file, begin_time, end_time, label = self._grab_fields(idx)
        audio = self._get_audio(path_to_file, begin_time, end_time)
        audio = self.sampler(audio)
        audio = self.augmenter(audio)
        audio = self.preprocessor(audio)

        if self.mode == "train" or self.mode == "val":
            label = self.metadata["label"][idx]
            return audio, label

        elif self.mode == "test":
            return audio

    def __len__(self):
        return self.metadata.shape[0]


class PeakNormalize:
    """Convert array to lay between 0 to 1"""

    def __call__(self, sample):

        return (sample - sample.min()) / (sample.max() - sample.min())

class UnitNormalize:
    """Remove mean and divide by std to normalize samples"""

    def __call__(self, sample):

        return (sample - sample.mean()) / (sample.std() + 1e-8)


class InferenceDataset(Dataset):
    '''
    class for storing and loading data.
    '''
    def __init__(self, file_path: (str, Path),
                 preprocessors: DictConfig,
                 seq_length: float = 1,
                 data_sample_rate: int = 44100,
                 sample_rate: int = 44100):
        """
        __init__ method initiates InferenceDataset instance:
        Input:

        Output:
        InferenceDataset Object - inherits from Dataset object in PyTorch package
        """
        self.file_path = file_path
        self.metadata_path = self.file_path  # alias to support inference pipeline
        self.seq_length = seq_length
        self.sample_rate = sample_rate
        self.data_sample_rate = data_sample_rate
        self.sampler = torchaudio.transforms.Resample(orig_freq=data_sample_rate, new_freq=sample_rate)
        self.preprocessor = ClassifierDataset.set_preprocessor(preprocessors)
        self._create_start_times()

    def _create_start_times(self):
        """
            create reference dict to extract audio files from metadata annotation
            Input:
            data_path - Path object
            Output:
            audio_dict contains references to audio paths given name from metadata
        """
        audio_len = sf.info(self.file_path).duration
        self._start_times = np.arange(0, audio_len//self.seq_length * self.seq_length, self.seq_length).astype(int)

    def _get_audio(self, begin_time):
        """
        _get_audio gets a path_to_file from _grab_fields method and also begin_time and end_time
        and returns the audio segment in a torch.tensor

        input:
        path_to_file - string
        begin_time - int
        end_time - int

        output:
        audio - pytorch tensor (1-D array)
        """
        data, _ = sf.read(self.file_path, start=begin_time,
                          stop=begin_time + self.seq_length * self.data_sample_rate)
        audio = torch.tensor(data, dtype=torch.float).unsqueeze(0)
        return audio

    def __getitem__(self, idx):
        '''
        __getitem__ method loads item according to idx from the metadata

        input:
        idx - int

        output:
        audio, label - torch tensor (1-d if no spectrogram is applied/ 2-d if applied a spectrogram
        , int (if mode="train" only)
        '''
        begin_time = self._start_times[idx] * self.data_sample_rate
        audio = self._get_audio(begin_time)
        audio = self.sampler(audio)
        audio = self.preprocessor(audio)

        return audio, idx

    def __len__(self):
        return len(self._start_times)