import pandas as pd
from pathlib import Path
import argparse
import soundfile as sf
from tqdm import tqdm


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-raven-folder", "-ir",
                        help="Path to the directory containing the raven files to be merged.", type=str)
    parser.add_argument("--input-audio-folder", "-ia",
                        help="Path to the directory containing the audio files that are aligned to the annotations.",
                        type=str)
    parser.add_argument("--output-path", "-o",
                        help="Path the the output path of the merged raven annotation", type=str)
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
    # get the list of raven files
    raven_files = list(raven_folder.glob('*.txt'))
    # get the list of audio files
    audio_files = list(audio_folder.glob('*.wav'))
    # sort the audio files by name, should be the order by start time as well
    audio_files = sorted(audio_files)
    assert len(raven_files) == len(audio_files), "The number of raven files and audio files should be the same."
    # create a mapping between the raven files and the audio files
    adapted_files_list = []
    for file in audio_files:
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
    for entry in tqdm(adapted_files_list):
        # read the raven file
        df = pd.read_csv(entry["raven_file"], sep="\t")
        # add the offset to the begin and end time
        df['Begin Time (s)'] += seconds_offset
        df['End Time (s)'] += seconds_offset
        df['Selection'] += entries_offset
        # get the audio file duration
        audio_file_duration = sf.info(entry["audio_file"]).duration
        # add the audio file duration to the offset
        seconds_offset += audio_file_duration
        entries_offset += df.shape[0]
        # add the dataframe to the list
        df_list.append(df)

    # concatenate the dataframes
    concatenated_df = pd.concat(df_list)
    # save the concatenated dataframe
    concatenated_df.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
