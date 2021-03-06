import torchvision
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import torchaudio
import pandas as pd
import random
import soundfile as sf
from itertools import starmap, repeat
from hydra.utils import instantiate
from typing import Union
from omegaconf import DictConfig
from pathlib import Path
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from soundbay.data_augmentation import ChainedAugmentations


class BaseDataset(Dataset):
    '''
    class for storing and loading data.
    '''
    def __init__(self, data_path, metadata_path, augmentations, augmentations_p, preprocessors,
                 seq_length=1, data_sample_rate=44100, sample_rate=44100, mode="train",
                 slice_flag=False, margin_ratio=0, split_metadata_by_label=False):
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
        self.dtype_dict = {'filename': 'str'}
        metadata = pd.read_csv(self.metadata_path, dtype=self.dtype_dict)
        self.metadata = self._update_metadata_by_mode(metadata, mode, split_metadata_by_label)
        self.mode = mode
        self.seq_length = seq_length
        self.sample_rate = sample_rate
        self.data_sample_rate = data_sample_rate
        self.sampler = torchaudio.transforms.Resample(orig_freq=data_sample_rate, new_freq=sample_rate)
        self._preprocess_metadata(slice_flag)
        self.augmenter = self._set_augmentations(augmentations, augmentations_p)
        self.preprocessor = self.set_preprocessor(preprocessors)
        assert (0 <= margin_ratio) and (1 >= margin_ratio)
        self.margin_ratio = margin_ratio
        self.items_per_classes = np.unique(self.metadata['label'], return_counts=True)[1]
        weights = 1 / self.items_per_classes
        self.samples_weight = np.array([weights[t] for t in self.metadata['label'] ])


    @staticmethod
    def _update_metadata_by_mode(metadata, mode, split_metadata_by_label):
        if split_metadata_by_label:
            metadata = metadata[metadata['split_type'] == mode]
        return metadata

    def _create_audio_dict(self, data_path: Path) -> dict:
        """
            create reference dict to extract audio files from metadata annotation
            Input:
            data_path - Path object
            Output:
            audio_dict contains references to audio paths given name from metadata
        """
        audio_paths = data_path.rglob('*.wav')
        return {x.name.replace('.wav', ''): x for x in audio_paths}

    def _preprocess_metadata(self, slice_flag=False):
        """
        function _preprocesses_metadata grabs calls with minimal length of self.seq_length + len_buffer
        Input:
            slice_flag: bool, default = False
                If true, the metadata file is sliced into segments of lengths self.seq_length.
        Output:
        ClassifierDataset object with self.metadata dataframe after applying the condition
        """

        # All calls are worthy (because we can later create a bigger slice contain them that is still a call in
        # _get_audio) but only long enough background sections will do.
        self.metadata = self.metadata[
            ((self.metadata['call_length'] >= self.seq_length) & (self.metadata['label'] == 0)) |
            (self.metadata['label'] > 0)
            ]

        if slice_flag:
            self._slice_sequence()

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
        if 'channel' in self.metadata.columns:
            channel = self.metadata['channel'][idx]
        else:
            channel = None
        return path_to_file, begin_time, end_time, label, channel


    def _slice_sequence(self):
        """
        function _slice_sequence process metadata list call lengths to be sliced according to self.seq_length
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

    def _get_audio(self, path_to_file, begin_time, end_time, label, channel=None):
        raise NotImplementedError

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
        For train/ val modes -
        audio_processed, label, audio_raw, idx - torch tensor (1-d if no spectrogram is applied/ 2-d if applied a spectrogram
        , int (if mode="train" only), 2-d tensor, int

        For test - audio_processed - torch tensor (1-d if no spectrogram is applied/ 2-d if applied a spectrogram


        '''
        path_to_file, begin_time, end_time, label, channel = self._grab_fields(idx)
        audio = self._get_audio(path_to_file, begin_time, end_time, label, channel)
        audio_raw = self.sampler(audio)
        audio_augmented = self.augmenter(audio_raw)
        audio_processed = self.preprocessor(audio_augmented)

        if self.mode == "train" or self.mode == "val":
            label = self.metadata["label"][idx]
            return audio_processed, label, audio_raw, idx

        elif self.mode == "test":
            return audio_processed

    def __len__(self):
        return self.metadata.shape[0]




class ClassifierDataset(BaseDataset):
    '''
    This class inherits all the traits from BaseDataset and handles cases that include Background noise
    (margin ratio feature is implemented)
    '''


    def _get_audio(self, path_to_file, begin_time, end_time, label, channel=None):
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
        seg_length = end_time - begin_time
        requested_seq_length = int(self.seq_length * self.data_sample_rate)
        last_start_time = sf.info(path_to_file).frames - requested_seq_length
        # Do all this stuff only to calls in training set, because otherwise _slice_sequence has already been done
        if self.mode == "train":
            if seg_length >= requested_seq_length:
                # Only for calls we can safely add sections out of the call and label it as call
                if (self.margin_ratio != 0) and (label > 0):
                    # self.margin_ratio ranges from 0 to 1 - indicates the relative part from seq_len to exceed call_length
                    margin_len_begin = int(requested_seq_length * self.margin_ratio)
                    margin_len_end = int(requested_seq_length * (1 - self.margin_ratio))
                    start_time = random.randint(max(begin_time - margin_len_begin, 0),
                                                min(end_time - margin_len_end, last_start_time))
                else:
                    start_time = random.randint(begin_time, min(end_time - requested_seq_length, last_start_time))

                if start_time < 0:
                    start_time = 0
            else:  # We know we can only arrive here with label > 0 because we filtered out short bg segments.
                # If the call is too short, the selected interval can be any interval of length requested_seq_length
                # that contains it.
                short_call_margin = requested_seq_length - seg_length
                # start time is between short_call_margin before begin time, and the latest time you can start and still
                # both contain the whole call and not get out of the file
                start_time = random.randint(max(begin_time - short_call_margin, 0),
                                            min(begin_time, last_start_time))
        else:
            start_time = begin_time
        data, _ = sf.read(str(path_to_file), start=start_time,
                          stop=start_time + requested_seq_length)
        if channel is not None:
            data = data[:, channel - 1]
        if data.shape[0] < 1:
            raise ValueError(f"Audio segment is empty. {path_to_file}: "
                             f"{start_time}, {start_time + requested_seq_length}")
        audio = torch.tensor(data, dtype=torch.float).unsqueeze(0)
        return audio


class NoBackGroundDataset(BaseDataset):
    '''
    This  class inherits all the traits from BaseDataset and handles cases with no Background noise (calls only dataset)
    '''



    def _get_audio(self, path_to_file, begin_time, end_time, label, channel=None):
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
        if (self.mode == "train"):
            start_time = random.randint(begin_time, end_time - int(self.seq_length * self.data_sample_rate))
            if start_time < 0:
                start_time = 0
        else:
            start_time = begin_time
        data, _ = sf.read(str(path_to_file), start=start_time,
                          stop=start_time + int(self.seq_length * self.data_sample_rate))
        if channel is not None:
            data = data[:, channel-1]
        if data.shape[0] < 1:
           raise ValueError(f"Audio segment is empty. {path_to_file}: "
                            f"{start_time}, {start_time + int(self.seq_length * self.data_sample_rate)}")
        audio = torch.tensor(data, dtype=torch.float).unsqueeze(0)
        return audio


class PeakNormalize:
    """Convert array to lay between 0 to 1"""

    def __call__(self, sample):

        return (sample - sample.min()) / (sample.max() - sample.min())


class UnitNormalize:
    """Remove mean and divide by std to normalize samples"""

    def __call__(self, sample):

        return (sample - sample.mean()) / (sample.std() + 1e-8)


class SlidingWindowNormalize:
    """ Based on Sliding window augmentations of
    https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/whale_cnn.py
        Translated to torch
        Has 50/50 chance of activating H sliding window or V sliding window

        Must come after spectrogram and before AmplitudeToDB
    """

    def __init__(self, sr: float, n_fft: int, lower_cutoff: float = 50, norm=True,
                 inner_ratio: float = 0.06, outer_ratio: float = 0.5):
        self.sr = sr
        self.n_fft = n_fft
        self.lower_cutoff = lower_cutoff
        self.norm = norm
        self.inner_ratio = inner_ratio
        self.outer_ratio = outer_ratio

    def spectrogram_norm(self, spect):

        min_f_ind = int((self.lower_cutoff / (self.sr / 2)) * self.n_fft)

        mval, sval = np.mean(spect[min_f_ind:, :]), np.std(spect[min_f_ind:, :])
        fact_ = 1.5
        spect[spect > mval + fact_ * sval] = mval + fact_ * sval
        spect[spect < mval - fact_ * sval] = mval - fact_ * sval
        spect[:min_f_ind, :] = mval

        return spect

    # slidingWindowV Function from: https://github.com/nmkridler/moby2/blob/master/metrics.py
    def slidingWindow(self, torch_spectrogram, dim=0):
        ''' slidingWindow Method
                Enhance the contrast vertically (along frequency dimension) for dim=0 and
                horizontally (along temporal dimension) for dim=1

                Args:
                    torch_spectrogram: 2-D numpy array image
                    dim: dimension to do the sliding window across
                Returns:
                    Q: 2-D numpy array image with vertically-enhanced contrast

        '''
        if dim not in {0, 1}:
            raise ValueError('dim must be 0 or 1')

        spect = torch_spectrogram.cpu().clone().numpy()
        spect_shape = spect.shape
        spect = spect.squeeze()
        if self.norm:
            spect = self.spectrogram_norm(spect)

        # Set up the local mean window
        wInner = np.ones(int(self.inner_ratio * spect.shape[dim]))
        # Set up the overall mean window
        wOuter = np.ones(int(self.outer_ratio * spect.shape[dim]))
        # Remove overall mean and local mean using np.convolve
        for i in range(spect.shape[1-dim]):
            if dim == 0:
                spect[:, i] = spect[:, i] - (
                        np.convolve(spect[:, i], wOuter, 'same') - np.convolve(spect[:, i], wInner, 'same')) / (
                            wOuter.shape[0] - wInner.shape[0])
            elif dim == 1:
                spect[i, :] = spect[i, :] - (
                        np.convolve(spect[i, :], wOuter, 'same') - np.convolve(spect[i, :], wInner, 'same')) / (
                                      wOuter.shape[0] - wInner.shape[0])

        spect[spect < 0] = 0.
        return torch.from_numpy(spect).reshape(spect_shape)

    def __call__(self, x):

        if random.random() < 0.5:
            return self.slidingWindow(x, dim=0)
        else:
            return self.slidingWindow(x, dim=1)


class Resize(torchvision.transforms.Resize):

    def __init__(self, size):
        super().__init__(list(size))


class InferenceDataset(Dataset):
    '''
    class for storing and loading data.
    '''
    def __init__(self, file_path: Union[str, Path],
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
        self._start_times = np.arange(0, audio_len//self.seq_length * self.seq_length, self.seq_length)

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
        data, orig_sample_rate = sf.read(self.file_path, start=begin_time,
                          stop=begin_time + int(self.seq_length * self.data_sample_rate))
        assert orig_sample_rate == self.data_sample_rate, \
            f'sample rate is {orig_sample_rate}, should be {self.data_sample_rate}'
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
        begin_time = int(self._start_times[idx] * self.data_sample_rate)
        audio = self._get_audio(begin_time)
        audio = self.sampler(audio)
        audio = self.preprocessor(audio)

        return audio

    def __len__(self):
        return len(self._start_times)
