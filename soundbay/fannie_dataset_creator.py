import os
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from soundbay.utils.metadata_processing import get_dir_wav_info, load_dir_annotation_with_file_time, load_annotation_correct_file_time, multi_target_from_time_intervals_df

class FannieDatasetCreator:
    def __init__(self, df: pd.DataFrame | None = None, random_state: int = 42) -> None:
        if df is not None:
            self.ann_df = df
        else:
            self.ann_df = pd.DataFrame()
        self.random_state = random_state

    def load_annotations(self, ann_path: Path, wav_ann: Path, label: str = "", filename_prefix: str = "LF_", filename_suffix: str | list[str] = ".Table.1.selections.txt") -> None:
        if ann_path.is_dir():
            ann_df = load_dir_annotation_with_file_time(ann_path, wav_info_dir=wav_ann, filename_prefix=filename_prefix, filename_suffix=filename_suffix)
        elif ann_path.exists():
            assert isinstance(filename_suffix, str), "For single file annotation, filename_suffix must be a string."

            ann_df = load_annotation_correct_file_time(ann_path, wav_ann, filename_prefix, filename_suffix)
        else:
            raise FileNotFoundError(f"Annotation path {ann_path} does not exist.")
        
        # add label column if provided
        if label:
            ann_df['label'] = label
        
        # ann_df = ann_df.drop(columns=['filename'])
        ann_df['wav_file'] = ann_df['wav_file'].apply(lambda x: Path(x).stem)
        ann_df['filename'] = ann_df['wav_file']
        self.ann_df = pd.concat([self.ann_df, ann_df], ignore_index=True)

    def add_monthly_noise(self, wav_path: Path, labels: list[str], wanted_year: int = -1, wanted_month: int = -1) -> None:
        wav_infos = get_dir_wav_info(wav_path)
        wav_infos['wav_file'] = wav_infos['wav_file'].apply(lambda x: Path(x).stem)

        if wanted_month > 0:
            wav_infos['year'] = wav_infos['date_time'].apply(lambda x: x.year)
            wav_infos = wav_infos[wav_infos.year == wanted_year]

        if wanted_month > 0:
            wav_infos['month'] = wav_infos['date_time'].apply(lambda x: x.month)
            wav_infos = wav_infos[wav_infos.month == wanted_month]
        # calculate the needed time to add noise as the same time of all the wanted labels. The noise would be with random length from min duration to max durations of the wanted labels
        min_time = float('inf')
        max_time = 0.0
        calls_durations = 0.0
        for label in labels:
            if label not in self.ann_df['label'].unique():
                raise ValueError(f"Label {label} not found in existing annotations.")
            
            label_df = self.ann_df[self.ann_df['label'] == label]
            min_time = min(min_time, label_df['call_length'].min())
            max_time = max(max_time, label_df['call_length'].max())
            calls_durations += label_df["call_length"].sum()

        # add noise annotations
        total_choosen_time = 0.0
        used_wav_files = set()
        pbar = tqdm(wav_infos.itertuples(), total=calls_durations, desc="Adding monthly noise annotations")

        new_annotations = {
            'begin_time': [],
            'end_time': [],
            'label': [],
            'filename': [],
            'duration': [],
            'call_length': []
        }
        while total_choosen_time < calls_durations:
            row_idx = np.random.randint(0, len(wav_infos))
            wav_info = wav_infos.iloc[row_idx]
            if wav_info.wav_file in used_wav_files:
                continue
            
            used_wav_files.add(wav_info.wav_file)

            # choose random duration between min_time and max_time
            duration = np.random.uniform(min_time, max_time)
            if duration > wav_info.duration:
                duration = wav_info.duration - 0.1  # leave at least 0.1 second

            begin_time = np.random.uniform(0, wav_info.duration - duration)
            end_time = begin_time + duration

            new_annotations['begin_time'].append(begin_time)
            new_annotations['end_time'].append(end_time)
            new_annotations['label'].append('Noise')
            new_annotations['filename'].append(wav_info.wav_file)
            new_annotations['duration'].append(np.nan)
            new_annotations['call_length'].append(duration)

            total_choosen_time += duration
            pbar.update(duration)
            
            if len(used_wav_files) >= len(wav_infos):
                used_wav_files = set()

        pbar.close()
        noise_ann_df = pd.DataFrame(new_annotations)
        self.ann_df = pd.concat([self.ann_df, noise_ann_df], ignore_index=True)


    def shuffle_annotations(self) -> None:
        self.ann_df = self.ann_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

    def add_call_length(self) -> None:
        self.ann_df['call_length'] = self.ann_df['end_time'] - self.ann_df['begin_time']

    def keep_necessary_columns(self, columns: list[str] = ["begin_time", "end_time", "label", "filename", "duration", "call_length"]) -> None:
        self.ann_df = self.ann_df[columns]

    def remove_file_name_prefix(self, prefix: str = "LF_") -> None:
        self.ann_df['filename'] = self.ann_df['filename'].str.replace(f"^{prefix}", "", regex=True)

    def remove_month(self, month_year: str) -> None:
        self.ann_df['month_year'] = self.ann_df.apply(lambda x: f"{x.date_time.year}_{x.date_time.month:02d}", axis=1)
        self.ann_df = self.ann_df[self.ann_df['month_year'] != month_year].reset_index(drop=True)
        self.ann_df = self.ann_df.drop(columns=['month_year'])

    def labels_to_indices(self, label_dict: dict[str, int]) -> None:
        self.ann_df['label'] = self.ann_df['label'].map(label_dict)

    def split_dataset(self, test_size: float = 0.2) -> tuple['FannieDatasetCreator', 'FannieDatasetCreator']:
        train_df, test_df = train_test_split(self.ann_df, test_size=test_size, random_state=self.random_state, stratify=self.ann_df['label'])
        train_creator = FannieDatasetCreator(train_df.reset_index(drop=True), random_state=self.random_state)
        test_creator = FannieDatasetCreator(test_df.reset_index(drop=True), random_state=self.random_state)
        return train_creator, test_creator

    def to_csv(self, save_path: Path) -> None:
        self.ann_df.to_csv(save_path, index=False)

    @property
    def annotations(self) -> pd.DataFrame:
        return self.ann_df
    
    def __len__(self) -> int:
        return len(self.ann_df)


if __name__ == "__main__":
    base_path = Path(os.getcwd())
    fannie_path = base_path / "datasets" / "fannie_project"

    creator = FannieDatasetCreator()

    #### Antarctic Blue
    creator.load_annotations(
        ann_path=fannie_path / "ANT_BLUE",
        wav_ann=fannie_path,
        label="ANT_BLUE"
    )

    creator.load_annotations(
        ann_path = fannie_path / "CallUnitsAndNoise" / "ANT_BLUE",
        wav_ann=fannie_path,
        filename_suffix=".Z.unit.1.selections.txt",
        label="ANT_BLUE"
    )

    #### Madagascar Blue annotations
    creator.load_annotations(
        ann_path=fannie_path / "MAD_BLUE",
        wav_ann=fannie_path,
        label="MAD_BLUE"
    )

    creator.load_annotations(
        ann_path = fannie_path / "CallUnitsAndNoise" / "MAD_BLUE" / "UNIT_1",
        wav_ann=fannie_path,
        filename_suffix=".Mad.unit.1.selections.txt",
        label="MAD_BLUE"
    )

    creator.load_annotations(
        ann_path = fannie_path / "CallUnitsAndNoise" / "MAD_BLUE" / "UNIT_2",
        wav_ann=fannie_path,
        filename_suffix=".Mad.unit.2.selections.txt",
        label="MAD_BLUE"
    )

    #### Noise annotations
    creator.load_annotations(
        ann_path = fannie_path / "CallUnitsAndNoise" / "NOISE",
        wav_ann=fannie_path,
        filename_suffix=".Noise.selections.txt",
        label="Noise"
    )

    creator.shuffle_annotations()
    creator.remove_file_name_prefix(prefix="LF_")
    creator.remove_month(month_year="2022_03")
    creator.keep_necessary_columns()
    creator.add_call_length()
    creator.add_monthly_noise(
        wav_path=fannie_path,
        labels=["ANT_BLUE", "MAD_BLUE"],
        wanted_year=2021,
        wanted_month=11
    )

    creator.labels_to_indices({
        "Noise": 0,
        "ANT_BLUE": 1,
        "MAD_BLUE": 2
    })
    train_creator, test_creator = creator.split_dataset(test_size=0.2)
    train_creator, val_creator = train_creator.split_dataset(test_size=0.1)

    print(f"train annotations: {len(train_creator)}")
    print(f"val annotations: {len(val_creator)}")
    print(f"test annotations: {len(test_creator)}")

    train_creator.to_csv(fannie_path / "train_updated.csv")
    val_creator.to_csv(fannie_path / "val_updated.csv")
    test_creator.to_csv(fannie_path / "test_updated.csv")