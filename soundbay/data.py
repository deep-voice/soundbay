import ast
import random
from itertools import starmap, repeat
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torchvision
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torchvision import transforms
from audiomentations import Compose


class BaseDataset(Dataset):
    """
    class for storing and loading data.
    """
    def __init__(self, data_path, metadata_path, augmentations, augmentations_p, preprocessors, label_type,
                 seq_length=1, data_sample_rate=44100, sample_rate=44100, mode="train",
                 slice_flag=False, margin_ratio=0, split_metadata_by_label=False, path_hierarchy: int = 0):
        """
        __init__ method initiates ClassifierDataset instance:
        Input:
        data_path - string
        metadata_path - string
        augmentations - list of classes audiogemtations
        augmentations_p - array of probabilities (float64)
        preprocessors - list of classes from preprocessors (TBD function)
        path_hierarchy - enables working with data that is organized in a hierarchy of folders. The default value is 0,
        which means all the audio files are flattened in the same folder. If the value is 1, the audio files are
        organized in one folder per class, and so on. The annotations in the metadata has to be aligned with the path
        hierarchy, and to include the parent folder names in the filename column.
        Example:
            path_hierarchy = 0:
            - main_folder
                - file1.wav
                - file2.wav
                - file3.wav
            path_hierarchy = 1:
            - main_folder
                - sub_folder1
                    - file1.wav
                    - file5.wav
                - sub_folder2
                    - file2.wav
                    - file4.wav
                - sub_folder3
                    - file3.wav
                    - file8.wav
        Output:
        ClassifierDataset Object - inherits from Dataset object in PyTorch package
        """
        self.audio_dict = self._create_audio_dict(Path(data_path), path_hierarchy=path_hierarchy)
        self.metadata_path = metadata_path
        self.dtype_dict = {'filename': 'str'}
        self.label_type = label_type
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
        self.num_classes = self._get_num_classes()
        self.samples_weight = self._get_samples_weight()

    @staticmethod
    def _update_metadata_by_mode(metadata, mode, split_metadata_by_label):
        if split_metadata_by_label:
            metadata = metadata[metadata['split_type'] == mode]
        return metadata

    def _create_audio_dict(self, data_path: Path, path_hierarchy=0) -> dict:
        """
            create reference dict to extract audio files from metadata annotation
            Input:
            data_path - Path object
            Output:
            audio_dict contains references to audio paths given name from metadata
        """
        def get_parent_path(path, path_hierarchy):
            parent_path_parts = path.parts[:-1]
            assert len(parent_path_parts) > path_hierarchy, \
                (f"Make sure path_hierarchy:{path_hierarchy} is smaller than actual files hierarchy "
                 f"{len(parent_path_parts)}")
            return '/'.join(parent_path_parts[len(parent_path_parts) - path_hierarchy:])

        audio_paths = list(data_path.rglob('*.wav'))
        return {f'{get_parent_path(x, path_hierarchy)}/{x.name[:-4]}'.strip('/'): x for x in audio_paths}

    def _preprocess_metadata(self, slice_flag=False):
        """
        function _preprocesses_metadata grabs calls with minimal length of self.seq_length + len_buffer
        Input:
            slice_flag: bool, default = False
                If true, the metadata file is sliced into segments of lengths self.seq_length.
        Output:
            ClassifierDataset object with self.metadata dataframe after applying the condition
        """
        self.metadata['label'] = self._preprocess_target()
        is_noise = self.metadata['label'].apply(self._is_noise)

        # All calls are worthy (because we can later create a bigger slice contain them that is still a call in
        # _get_audio) but only long enough background sections will do.
        self.metadata = self.metadata[((self.metadata['call_length'] >= self.seq_length) & is_noise) | (~is_noise)]

        # sometimes the bbox's end time exceeds the file's length
        for name, sub_df in self.metadata.groupby('filename'):
            duration = sf.info(str(self.audio_dict[name])).duration
            if not all(sub_df['end_time'] <= duration):
                print(f'seems like some tags in file {name} have bigger end_time than its duration')
                print(f"file {name} --- int(duration): {int(duration)} --- biggest end time: {sub_df['end_time'].max()}")

        if slice_flag:
            self._slice_sequence()

        self.metadata.reset_index(drop=True, inplace=True)

    def _preprocess_target(self) -> pd.Series:
        """
        Preprocesses the label column in the metadata. If the label is a string, it is evaluated and converted to an
        integer or a list of integers.
        """
        if pd.api.types.is_string_dtype(self.metadata['label']):
            assert self.metadata['label'].str.match(r'^(\[|\()?(\d+)(\s*,\s*\d+)*(\]|\))?$').all(), \
                "label should be a string that could be evaluated as a list of integers or integers."
            self.metadata['label'] = self.metadata['label'].apply(ast.literal_eval)
            if self.metadata['label'].apply(lambda x: isinstance(x, (list, tuple))).all():
                self.metadata['label'] = self.metadata['label'].apply(np.array, dtype=int)
        return self.metadata['label']


    @staticmethod
    def _is_noise(value: Union[int, np.ndarray]) -> bool:
        """
        Checks if the value is a noise, i.e., if it is equal to 0.
        """
        assert (isinstance(value, (int, np.integer)) | isinstance(value, np.ndarray)), "value should be either int or np.ndarray"
        return np.sum(value) == 0

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
        count_values_before = self.metadata.astype({'label': str}).value_counts('label', sort=False) # for validating that the following code doesn't lose samples
        sliced_times = list(starmap(np.arange, zip(self.metadata['begin_time'], self.metadata['end_time'], repeat(self.seq_length))))
        # add the last sequence at the end of this list for calls only (only if it does not exceed the file)
        sliced_times = list([np.append(s, self.metadata.loc[i, 'end_time']) if (not self._is_noise(self.metadata.loc[i, 'label']))
                             else s for i, s in enumerate(sliced_times)])
        new_begin_time = list(x[:-1] for x in sliced_times)
        duplicate_size_vector = [len(list_elem) for list_elem in new_begin_time] # vector to duplicate original dataframe
        new_begin_time = np.concatenate(new_begin_time) # vectorize to array
        new_end_time = np.concatenate(list(x[1:] for x in sliced_times)) # same for end_times
        self.metadata = self.metadata.iloc[self.metadata.index.repeat(duplicate_size_vector)].reset_index(drop=True)
        self.metadata['begin_time'] = new_begin_time
        self.metadata['end_time'] = new_end_time
        self.metadata['call_length'] = np.shape(self.metadata)[0] * [self.seq_length]
        count_values_after = self.metadata.astype({'label': str}).value_counts('label', sort=False)
        if not all(count_values_after >= count_values_before):
            print(f'Note: seems like _slice_sequence erases data.\nbefore:{count_values_before}\n'
                  f'after:{count_values_after}')
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
        self._train_augmenter = Compose(augmentations_list, p=augmentations_p, shuffle=True)
        self._val_augmenter = torch.nn.Identity()

    def augment(self, x):
        if self.mode == 'train':
            return torch.tensor(self._train_augmenter(x.numpy(), self.sample_rate), dtype=torch.float32)
        else:
            return self._val_augmenter(x)

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

    def _get_num_classes(self) -> int:
        """
        Returns the number of classes in the metadata.
        """
        if self.label_type == 'multi_label':
            label_lengths = self.metadata['label'].apply(len)
            assert label_lengths.nunique() == 1, "All labels should have the same length"
            return label_lengths.iloc[0]
        else:
            return self.metadata['label'].nunique()

    def _get_samples_weight(self) -> np.ndarray:
        """
        Returns the weight of each sample in the dataset:
            - if the label is integer, the weight is the inverse of the class count.
            - if the label is a list, the weight is the inverse of the minimum class count.
        """
        if self.label_type == 'multi_label':
            noise_counts = self.metadata['label'].apply(self._is_noise).sum()
            class_counts = np.sum(self.metadata['label'])
            per_sample_min_class_count = (self.metadata['label'].apply(
                lambda x: class_counts[x.astype(bool)].min() if not self._is_noise(x) else noise_counts))
            return (1 / per_sample_min_class_count).values
        else:
            weights = 1 / np.unique(self.metadata['label'], return_counts=True)[1]
            return np.array([weights[t] for t in self.metadata['label']])


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
        audio_augmented = self.augment(audio_raw)
        audio_processed = self.preprocessor(audio_augmented)

        if self.mode == "train" or self.mode == "val":
            label = self.metadata["label"][idx]
            return audio_processed, label, audio_raw, {"idx": idx, "begin_time": begin_time, "org_file": Path(path_to_file).stem}

        elif self.mode == "test":
            return audio_processed

    def __len__(self):
        return self.metadata.shape[0]




class ClassifierDataset(BaseDataset):
    """
    This class inherits all the traits from BaseDataset and handles cases that include Background noise
    (margin ratio feature is implemented)
    """

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
                if (self.margin_ratio != 0) and (not self._is_noise(label)):
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
            if begin_time < last_start_time:
                start_time = begin_time
            else:
                print(f'in {path_to_file}, one of the val\'s begin times is too big and exceeding the file so it was set to be smaller\nbegin time:{begin_time}, last_start_time:{last_start_time}')
                start_time = last_start_time
        data, _ = sf.read(str(path_to_file), start=start_time,
                          stop=start_time + requested_seq_length)
        if channel is not None and data.ndim > 1:
            assert channel > 0, f"channel as to be a positive integer, got {channel}"
            data = data[:, channel - 1]
        elif channel is None and data.ndim > 1:
            data = data[:, 0] # when channel is not specified, take the first channel
        if data.shape[0] < 1:
            raise ValueError(f"Audio segment is empty. {path_to_file}: "
                             f"{start_time}, {start_time + requested_seq_length}")
        audio = torch.tensor(data, dtype=torch.float).unsqueeze(0)
        return audio


class NoBackGroundDataset(BaseDataset):
    """
    This  class inherits all the traits from BaseDataset and handles cases with no Background noise (calls only dataset)
    """

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

        return (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)


class MinFreqFiltering:
    """Cut the spectrogram frequency axis to make it start from min_freq
    ***Note: In case a MaxFreqFiltering is implemented, the max_freq should be greater than min_freq***

    input:
        min_freq_filtering - int
        sample_rate - int

    output:
        spectrogram - pytorch tensor (3-D array)
    """

    def __init__(self, min_freq_filtering, sample_rate):
        self.min_freq_filtering = min_freq_filtering
        self.sample_rate = sample_rate

    def edit_spectrogram_axis(self, sample):
        if self.min_freq_filtering > self.sample_rate / 2 or self.min_freq_filtering < 0:
            raise ValueError("min_freq_filtering should be greater than 0, and smaller than sample_rate/2")
        max_freq_in_spectrogram = self.sample_rate / 2
        min_value = sample.size(dim=1) * self.min_freq_filtering / max_freq_in_spectrogram
        min_value = int(np.floor(min_value))
        sample = sample[:, min_value:, :]

        return sample

    def __call__(self, sample):

        return self.edit_spectrogram_axis(sample)


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
                 sample_rate: int = 44100,
                 overlap: float = 0):
        """
        __init__ method initiates InferenceDataset instance:
        Input:

        Output:
        InferenceDataset Object - inherits from Dataset object in PyTorch package
        """
        assert  0 <= overlap < 1, f'overlap should be between 0 and 1, got {overlap}'

        self.file_path = Path(file_path)
        self.metadata_path = self.file_path  # alias to support inference pipeline
        self.seq_length = seq_length
        self.sample_rate = sample_rate
        self.data_sample_rate = data_sample_rate
        self.overlap = overlap
        self.sampler = torchaudio.transforms.Resample(orig_freq=data_sample_rate, new_freq=sample_rate)
        self.preprocessor = ClassifierDataset.set_preprocessor(preprocessors)
        self.metadata = self._create_inference_metadata()

    def _create_inference_metadata(self) -> pd.DataFrame:
        """
        create metadata to be used in the inference dataset
        in case we have a directory, we will iterate over all files in the directory
        and create metadata for each file and merge it together
        For a single file, we will create metadata for that file
        """
        all_data_frames = []
        if self.file_path.is_dir():
            all_files = [self.file_path / x for x in self.file_path.iterdir()]
        else:
            all_files = [self.file_path]
        for file in all_files:
            if file.suffix not in ['.wav', '.WAV']:
                raise ValueError(f'InferenceDataset only supports .wav files, got {file.suffix}')
            file_start_time = self._create_start_times(file)
            for channel_num in range(sf.info(file).channels):
                metadata = pd.DataFrame({'filename': [file] * len(file_start_time),
                                         'channel': [channel_num] * len(file_start_time),
                                         'begin_time': file_start_time,
                                         'end_time': file_start_time + self.seq_length})
                all_data_frames.append(metadata)
        metadata = pd.concat(all_data_frames, ignore_index=True)
        return metadata

    def _create_start_times(self, filepath: Path) -> np.ndarray:
        """
            create reference dict to extract audio files from metadata annotation
            Input:
            data_path - Path object
            Output:
            audio_dict contains references to audio paths given name from metadata
        """
        audio_len = sf.info(filepath).duration
        step = self.seq_length * (1-self.overlap)
        start_times =  np.arange(0, audio_len, step)
        filtered_start_times = start_times[np.where(start_times <= audio_len - self.seq_length)]
        # if (duration - seq_length) is not a multiple of the step size, add the last segment
        if filtered_start_times[-1] < audio_len - self.seq_length:
            filtered_start_times = np.append(filtered_start_times, audio_len - self.seq_length)
        return filtered_start_times

    def _get_audio(self, filepath: Path, channel: int, begin_time: float) -> torch.Tensor:
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
        duration = sf.info(filepath).duration
        begin_time = int(begin_time * self.data_sample_rate)
        stop_time = begin_time + int(self.seq_length * self.data_sample_rate)
        assert duration * self.data_sample_rate >= stop_time, f"trying to load audio from {begin_time} to {stop_time} but audio is only {duration} long"
        data, orig_sample_rate = sf.read(filepath, start=begin_time, stop=stop_time, always_2d=True)
        data = data[:, channel]
        assert orig_sample_rate == self.data_sample_rate, \
            f'sample rate is {orig_sample_rate}, should be {self.data_sample_rate}'
        audio = torch.tensor(data, dtype=torch.float).unsqueeze(0)
        return audio

    def __getitem__(self, idx: int):
        '''
        __getitem__ method loads item according to idx from the metadata.

        input:
        idx - int

        output:
        audio -  torch tensor (1-d if no spectrogram is applied/ 2-d if applied a spectrogram
        '''
        filepath, channel, begin_time = self.metadata.loc[idx, ['filename', 'channel', 'begin_time']]
        audio = self._get_audio(filepath=filepath, channel=channel, begin_time=begin_time)
        audio = self.sampler(audio)
        audio = self.preprocessor(audio)

        return audio

    def __len__(self):
        return len(self.metadata)
