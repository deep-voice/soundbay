import pandas as pd
from pathlib import Path
import argparse
import soundfile as sf
from tqdm import tqdm
import numpy as np


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-raven-folder", "-ir",
                        help="Path to the directory containing the raven files to be merged.", type=str)
    parser.add_argument("--input-audio-folder", "-ia",
                        help="Path to the directory containing the audio files that are aligned to the annotations.",
                        type=str)
    parser.add_argument("--output-path", "-o",
                        help="Path the the output path of the merged raven annotation", type=str)
    parser.add_argument("--output-path-index", "-oi",
                        help="Path the the output path of the index of files order", type=str)
    parser.add_argument("--include-begin-file", "-ibf",  dest="include_begin_file", action="store_true")
    parser.add_argument("--no-begin-file", "-nbf", dest="include_begin_file", action="store_false")
    parser.set_defaults(include_begin_file=True)
    return parser


def main() -> None:
    """
    This script is used to merge multiple raven annotation files into one file.
    """
    # configurations:
    args = make_parser().parse_args()
    raven_folder = Path(args.input_raven_folder)
    audio_folder = Path(args.input_audio_folder)
    output_path = Path(args.output_path)
    output_path_index = Path(args.output_path_index)
    include_begin_file = args.include_begin_file
    # get the list of raven files
    raven_files = list(raven_folder.glob('*.txt'))
    # get the list of audio files
    audio_files = list(audio_folder.glob('*.wav'))
    # sort the audio files by name, should be the order by start time as well
    audio_files = sorted(audio_files)
    assert len(raven_files) == len(audio_files), "The number of raven files and audio files should be the same."
    # create a mapping between the raven files and the audio files
    adapted_files_list = []
    for file in tqdm(audio_files, desc="Mapping audio files to raven files"):
        file_stem = file.stem
        adapted_raven_files = [raven_file for raven_file in raven_files if file_stem in raven_file.stem]
        assert len(adapted_raven_files) == 1, f"Expected one raven file for {file_stem}, found {len(adapted_raven_files)}"
        raven_file = adapted_raven_files[0]
        res = {"name": file_stem, "audio_file": file, "raven_file": raven_file}
        adapted_files_list.append(res)
    # create a list to store the dataframes
    df_list = []
    # iterate over the raven files
    seconds_offset = 0
    entries_offset = 0
    for entry in tqdm(adapted_files_list, desc="Merging raven files"):
        # read the raven file
        df = pd.read_csv(entry["raven_file"], sep="\t")
        # add the offset to the begin and end time
        df['Begin Time (s)'] += seconds_offset
        df['End Time (s)'] += seconds_offset
        df['Selection'] += entries_offset
        df['Begin File'] = [entry["audio_file"].name] * df.shape[0]
        # get the audio file duration
        audio_file_duration = sf.info(entry["audio_file"]).duration
        # add the audio file duration to the offset
        seconds_offset += audio_file_duration
        entries_offset += df.shape[0]
        # add the dataframe to the list
        df_list.append(df)

    # concatenate the dataframes
    concatenated_df = pd.concat(df_list)

    unique_files = concatenated_df["Begin File"].unique()
    # save unique files to a file
    np.savetxt(output_path_index, unique_files, fmt='%s')
    # remove the begin file column if not needed
    if not include_begin_file:
        concatenated_df = concatenated_df.drop(columns=["Begin File"])

    # save the concatenated dataframe
    concatenated_df.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
