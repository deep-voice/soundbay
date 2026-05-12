import ast
import random
from itertools import starmap, repeat
from pathlib import Path
from typing import Union

# import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
# import torchvision
from hydra.utils import instantiate
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


class MultiCallDetectionDataset(BaseDataset): ## Maybe create the dataset before hand and only load it as it takes a lot of time...
    """
    This class return a audio segment as x and a list of for detection in boundary box format for y. 
    y vector is: Batch X MAX_LABELS X [x, y, w, h, class_1, class_2, ..., class_n, confidence]
    """
    def __init__(self, data_path: Path, metadata_path: Path,
                 augmentations, augmentations_p, preprocessors,
                 n_fft: int, hop_length: int, label_type: str,
                 n_classes: int = 3,
                 seq_length=1.0, orig_sample_rate=44100, wanted_sample_rate=44100,
                 max_overlap_labels=5, margin_ratio=0.0, file_name_col='filename',
                 begin_time_col='begin_time', end_time_col='end_time', 
                 low_freq_col='low_freq', high_freq_col='high_freq', 
                 label_col='label', channel_col='channel', add_random_margin=True,
                 verbose=False):
        
        self.data_path = Path(data_path)
        self.metadata_path = Path(metadata_path)
        self.max_overlap_labels = max_overlap_labels
        self.verbose = verbose

        self.file_name_col = file_name_col
        self.begin_time_col = begin_time_col
        self.end_time_col = end_time_col
        self.low_freq_col = low_freq_col
        self.high_freq_col = high_freq_col
        self.label_col = label_col
        self.channel_col = channel_col
        self.n_classes = n_classes

        self.seq_length = seq_length
        self.orig_sample_rate = orig_sample_rate
        self.wanted_sample_rate = wanted_sample_rate
        assert (0 <= margin_ratio) and (1 >= margin_ratio)
        self.margin_ratio = margin_ratio

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.label_type = label_type

        self.to_spectrogram = T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=2.0)
        self.to_db = T.AmplitudeToDB(stype="power", top_db=80)

        # self.metadata = pd.DataFrame()
        self.metadata = pd.read_csv(self.metadata_path)
        self.preprocessor = self.set_preprocessor(preprocessors)
        
        self.segments_df = self._create_segments(add_random_margin=add_random_margin)
        self._set_augmentations(augmentations, augmentations_p)
        self.resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=wanted_sample_rate)

    @staticmethod
    def find_annotated_segments(file_df: pd.DataFrame, shpil_sec: float = 1.0):
        sorted_df = file_df.sort_values(by='begin_time')
        segments = []
        current_segment_start = sorted_df.iloc[0]['begin_time']
        current_segment_end = sorted_df.iloc[0]['end_time']
        for _, row in sorted_df.iterrows():
            if row['begin_time'] <= current_segment_end + shpil_sec:
                current_segment_end = max(current_segment_end, row['end_time'])
            else:
                segments.append((current_segment_start, current_segment_end)) # TODO: add metadata needed
                current_segment_start = row['begin_time']
                current_segment_end = row['end_time']
        segments.append((current_segment_start, current_segment_end))
        return segments
        
    @staticmethod
    def cut_segment_into_chunks(segment_start: float, segment_end: float, 
                                chunk_size: float = 1.0, hop_size: float = 0.5) -> pd.DataFrame:
        chunks = []
        current_start = segment_start
        while current_start + chunk_size <= segment_end:
            current_end = current_start + chunk_size
            chunks.append((current_start, current_end))
            current_start += hop_size
        
        if current_start < segment_end:
            chunks.append((segment_end - chunk_size, segment_end))

        return pd.DataFrame(chunks, columns=['begin_time', 'end_time'])
    
    def _get_file_duration(self, file_name):
        if str(file_name).lower().endswith('.wav'):
            file = str(file_name)
        else:
            file = str(file_name) + '.wav'
        return sf.info(self.data_path / file).duration

    def _create_segments(self, add_random_margin=True):
        """
        function that calcualte the metadata of the dataset.
        """
        df = pd.read_csv(self.metadata_path)
        # remove duplications
        df = df.drop_duplicates()
        segments_list = []
        for file in df[self.file_name_col].unique():
            file_df = df[df[self.file_name_col] == file]
            segments = self.find_annotated_segments(file_df)
            for segment_start, segment_end in segments:
                if segment_start < 0 and segment_end > 0:
                    if self.verbose:
                        print(f'seems like some tags in file {file} have negative begin_time, setting it to 0\nbegin time:{segment_start}')
                    segment_start = 0
                    segment_end = self.seq_length
                elif segment_start < 0 and segment_end < 0:
                    if self.verbose:
                        print(f'seems like some tags in file {file} have negative end_time continuing...')
                    continue

                if add_random_margin:
                    margin_start = np.random.uniform(-self.margin_ratio, self.margin_ratio) * self.seq_length

                    # start segment should not be negative
                    segment_start = max(0, segment_start + margin_start)
                    # segment_end = segment_start + self.seq_length
                    
                    file_duration = self._get_file_duration(file)
                    
                    margin_end = np.random.uniform(-self.margin_ratio, self.margin_ratio) * self.seq_length
                    segment_end = min(file_duration, segment_end + margin_end)

                    # segment_start -= max(0, np.random.uniform(0, self.margin_ratio) * self.seq_length)
                    # segment_end += max(0, np.random.uniform(0, self.margin_ratio) * self.seq_length)

                segment_df = self.cut_segment_into_chunks(segment_start, segment_end, 
                                                          chunk_size=self.seq_length, 
                                                          hop_size=self.seq_length * (1 - self.margin_ratio))
                segment_df[self.file_name_col] = file
                segments_list.append(segment_df)

        segments_df = pd.concat(segments_list, ignore_index=True)
        return segments_df            

    def _set_augmentations(self, augmentations_dict, augmentations_p):
        """
        get augmentations list and instantiate - TBD
        """
        if augmentations_dict is not None:
            augmentations_list = [instantiate(args) for args in augmentations_dict.values()]
        else:
            augmentations_list = []
        self._augmenter = Compose(augmentations_list, p=augmentations_p, shuffle=True)

    def __len__(self):
        return len(self.segments_df)
    
    @staticmethod
    def find_annotations_in_segment(file_df: pd.DataFrame, segment_start: float, segment_end: float):
        return file_df[(file_df['begin_time'] < segment_end) & (file_df['end_time'] > segment_start)]

    def _get_labels(self, idx):
        file_name = self.segments_df['filename'][idx]
        begin_time = self.segments_df['begin_time'][idx]
        end_time = self.segments_df['end_time'][idx]

        file_df = self.metadata[self.metadata[self.file_name_col] == file_name]
        annotations_in_segment = self.find_annotations_in_segment(file_df, begin_time, end_time)

        # transform to bounding box format: [x, y, w, h, class_1, class_2, ..., class_n, confidence]
        annotations_in_segment = annotations_in_segment.sort_values(by='begin_time').head(self.max_overlap_labels)
        labels = torch.zeros((self.max_overlap_labels, self.n_classes + 5), dtype=torch.float)  # +5 for x, y, w, h, confidence

        for i, (_, row) in enumerate(annotations_in_segment.iterrows()):
            if i >= self.max_overlap_labels:
                break
            x = max(0, row['begin_time'] - begin_time) / self.seq_length
            y = max(0, row['low_freq']) / (self.wanted_sample_rate / 2) # normalize between 0 and 1 to fit the spectrogram frequency bins
            # y = ( (self.wanted_sample_rate / 2) - row['high_freq'] ) / (self.wanted_sample_rate / 2) # flip y axis to match spectrogram representation
            w = min(end_time, row['end_time']) - max(begin_time, row['begin_time'])
            w = w / self.seq_length
            h = (row['high_freq'] - row['low_freq']) / (self.wanted_sample_rate / 2) # normalize between 0 and 1 to fit the spectrogram frequency bins
            # get class in one-hot encoding format:
            cls = row[self.label_col]
            one_hot_class = torch.zeros(self.n_classes)
            
            if cls == 0: # noise
                conf = 0.0
            else:
                one_hot_class[int(cls) - 1] = 1.0 # classes are 1-indexed in the metadata
                conf = 1.0

            labels[i] = torch.tensor([x, y, w, h, *one_hot_class, conf], dtype=torch.float)

        # sort by x coordinate (time) to make sure the order of the labels is consistent
        labels = labels[labels[:, 0].argsort()]

        return labels
    
    def get_all_labels(self):
        all_labels = []
        for idx in range(len(self.segments_df)):
            labels = self._get_labels(idx)
            all_labels.append(labels)
        return all_labels
    
    def _get_audio(self, idx):
        file_name = self.segments_df['filename'][idx]
        begin_time = self.segments_df['begin_time'][idx]
        if begin_time < 0:
            if self.verbose:
                print(f'seems like some tags in file {file_name} have negative begin_time, setting it to 0\nbegin time:{begin_time}')
            begin_time = 0
        
        if '.wav' not in str(file_name):
            file_name = str(file_name) + '.wav'
        audio, _ = torchaudio.load(self.data_path / file_name,
                                   frame_offset=int(begin_time * self.orig_sample_rate),
                                   num_frames=int(self.seq_length * self.orig_sample_rate))
     
        return audio
    
    def _get_spectrogram(self, audio):
        # S = librosa.stft(audio.squeeze().numpy(), n_fft=self.n_fft, hop_length=self.hop_length)
        # S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        spec = self.to_spectrogram(audio)
        spec = self.to_db(spec)
        spec = (spec - spec.mean()) / (spec.std() + 1e-9) # normalize spectrogram
        return spec
    
    def augment(self, x):
        return torch.tensor(self._augmenter(x.numpy(), self.wanted_sample_rate), dtype=torch.float32)
            
    def __getitem__(self, idx):
        # get audio
        audio_raw = self._get_audio(idx)
        audio_raw = self.resampler(audio_raw)
        audio_raw = self.augment(audio_raw)
        audio = self.preprocessor(audio_raw)
        spec = self._get_spectrogram(audio_raw)

        # get labels
        label = self._get_labels(idx)
        return spec, label
        

if __name__ == "__main__":
    # Example usage
    print("Creating dataset...")
    data_dir = Path("/mnt/d/DeepVoice/soundbay/datasets/fannie_project")
    dataset = MultiCallDetectionDataset(
        data_path=data_dir ,
        metadata_path=data_dir  / "train_updated.csv",
        augmentations=None,
        augmentations_p=0.5,
        preprocessors=[],
        n_classes=2,
        seq_length=20.0,
        orig_sample_rate=1_000,
        wanted_sample_rate=128,
        max_overlap_labels=4,
        margin_ratio=0.1,
        add_random_margin=True,
        n_fft=256,
        hop_length=16,
        label_type='multi_label'
    )
    print(f"Dataset length: {len(dataset)}")
    random_idx = random.randint(0, len(dataset) - 1)
    for i in range(min(5, len(dataset))):
        audio, labels = dataset[i]
        print(f"Audio shape: {audio.shape}, Labels shape: {labels.shape}")
        print(f"Labels: {labels}")

    # find max boxes:
    from tqdm import tqdm
    max_boxes = 0
    try:
        for i in tqdm(range(len(dataset))):
                _, labels = dataset[i]
                num_boxes = (labels[:, 4] > 0).sum().item()
                max_boxes = max(max_boxes, num_boxes)
    except Exception as e:
        print(max_boxes)
        print(f"Error occurred while processing dataset: {e}")
    print(f"Max boxes in any segment: {max_boxes}")