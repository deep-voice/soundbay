"""Code to extract noise only audio files from a mixed dataset"""

import os
import argparse
import random

import numpy as np

import pandas as pd
import soundfile as sf


def write_noise_file(noise_file_path, noise, sample_rate, n_keep, write_length):
    """Write a noise only audio file to disk, keeping only one every n_keep sections of length write_length.

    Parameters:
    -----------
    noise_file_path: str
        Path to save the noise file to
    noise: np.ndarray
        Noise only audio
    sample_rate: int
        Sample rate of the audio
    n_keep: int
        Save one in n_keep sections of the original file
    write_length: float
        Length of noise only audio files to save
    """
    for i in range(0, noise.shape[0], int(write_length * sample_rate)):
        if random.uniform(0, 1) < 1.0 / n_keep:
            end = i + int(write_length * sample_rate)
            if end > noise.shape[0]:
                break
            sf.write(noise_file_path + f'_{i}.wav', noise[i:end, :], sample_rate)


def main():
    parser = argparse.ArgumentParser(description='Extract noise only audio files from a mixed dataset')
    parser.add_argument('--path', required=True, help='path to dataset')
    parser.add_argument('--annot', required=True, help='path to annotation file for the training data')
    parser.add_argument('--out', required=True, help='path to output folder')
    parser.add_argument('--len', type=float, required=True, help='length of noise only audio files to save (seconds)')
    parser.add_argument('--keep', type=int, default=10, help='save one in keep sections of the original file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--force', action='store_true', help='force overwrite of existing files')
    args = parser.parse_args()

    # Read annotation file
    metadata_df = pd.read_csv(args.annot)

    # Set random seed
    random.seed(args.seed)

    # Iterate over all files in the dataset
    training_files = sorted(metadata_df['filename'].unique())
    for file in training_files:
        audio, sr = sf.read(os.path.join(args.path, file + '.wav'), always_2d=True)
        noise_file_path = os.path.join(args.out, file)
        if os.path.exists(noise_file_path) and not args.force:
            continue

        # Get file metadata
        file_annotations = metadata_df[metadata_df['filename'] == file]

        # Extract begin and end times for times with signal at some channel
        no_channel = file_annotations[['begin_time', 'end_time', 'label']].groupby(['begin_time', 'end_time']).max()
        signal = no_channel[no_channel.label != 0]

        if len(signal) == 0:
            # If no signal is present, write the whole file as noise
            write_noise_file(noise_file_path, audio, sr, args.keep, args.len)
            continue

        # Find the indices of the time slices containing signal
        to_remove = (slice(int(row[0][0] * sr), int(row[0][1] * sr)) for row in signal.itertuples(index=True))

        # Convert that into a list of indices to remove
        audio_indices = np.arange(audio.shape[0])
        to_remove = np.hstack([audio_indices[i] for i in to_remove])

        # Remove the indices from the audio
        just_noise = np.take(audio, sorted(set(audio_indices) - set(to_remove)), axis=0)

        # Save the noise only audio. Save in pieces of args.len seconds, keeping only one every args.keep
        write_noise_file(noise_file_path, just_noise, sr, args.keep, args.len)


if __name__ == '__main__':
    main()
