import pandas as pd
from datetime import datetime
import pytz
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def load_csvs(annotations_dir: pathlib.Path, split_type: str, max_duration: float) -> List[pd.DataFrame]:
    """Process all annotation files and create a DataFrame with relative timestamps."""
    annotation_files = list(annotations_dir.glob("*.csv"))

    # Process all annotations
    all_dfs = []
    for csv_file in annotation_files:
        df = pd.read_csv(csv_file)

        # Extract file start time from filename and convert to datetime with UTC timezone
        df['file_start'] = df['filename'].str.split('_').str[0].apply(
            lambda x: datetime.strptime(x, '%Y-%m-%dT%H-%M-%S').replace(tzinfo=pytz.UTC)
        )

        # Convert timestamps to UTC if they're not already
        df['start_datetime'] = pd.to_datetime(df['start_datetime']).dt.tz_convert('UTC')
        df['end_datetime'] = pd.to_datetime(df['end_datetime']).dt.tz_convert('UTC')

        # Calculate relative times
        df['begin_time'] = (df['start_datetime'] - df['file_start']).dt.total_seconds()
        df['end_time'] = (df['end_datetime'] - df['file_start']).dt.total_seconds()

        # Ensure times are within the legit range: [0, max_duration]
        df['begin_time'] = df['begin_time'].clip(0, max_duration)
        df['end_time'] = df['end_time'].clip(0, max_duration)

        df['call_length'] = df['end_time'] - df['begin_time']
        df['source'] = split_type

        all_dfs.append(df)
    return all_dfs


def boxplot_freq_per_label(df_calls: pd.DataFrame) -> None:
    label_counts = df_calls.label.value_counts()

    # Set figure and style
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")

    # Set shared y-axis limits
    ymax = max(df_calls['low_frequency'].max(), df_calls['high_frequency'].max())
    ymin = 0

    # First subplot: low frequency
    plt.subplot(1, 2, 1)
    ax1 = sns.boxplot(data=df_calls, x='label', y='low_frequency')
    plt.title('Low Frequency per Label')
    plt.xticks(rotation=45)
    ax1.set_ylim(ymin, ymax)

    # Annotate counts on top of each box
    for i, label in enumerate(ax1.get_xticklabels()):
        count = label_counts.get(int(label.get_text()), 0)
        ax1.text(i, ax1.get_ylim()[1], f'n={count}',
                 ha='center', va='top', fontsize=9, color='black')

    # Second subplot: high frequency
    plt.subplot(1, 2, 2)
    ax2 = sns.boxplot(data=df_calls, x='label', y='high_frequency')
    plt.title('High Frequency per Label')
    plt.xticks(rotation=45)
    ax2.set_ylim(ymin, ymax)

    # Annotate counts on top of each box
    for i, label in enumerate(ax2.get_xticklabels()):
        count = label_counts.get(int(label.get_text()), 0)
        ax2.text(i, ax2.get_ylim()[1], f'n={count}',
                 ha='center', va='top', fontsize=9, color='black')

    plt.show()


def segment_frequency_overlap(df, N=1) -> pd.DataFrame:
    # Step 1: Compute frequency ranges per label
    freq_stats = df.groupby('label').agg(
        low_median=('low_frequency', 'median'),
        low_std=('low_frequency', 'std'),
        high_median=('high_frequency', 'median'),
        high_std=('high_frequency', 'std')
    ).reset_index()

    freq_stats['freq_min'] = freq_stats['low_median'] - N * freq_stats['low_std']
    freq_stats['freq_max'] = freq_stats['high_median'] + N * freq_stats['high_std']

    # Step 2: Collect all boundary points
    boundaries = sorted(set(freq_stats['freq_min'].tolist() + freq_stats['freq_max'].tolist()))

    # Step 3: Define segments from boundaries
    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        active_labels = freq_stats[
            (freq_stats['freq_min'] < end) & (freq_stats['freq_max'] > start)
        ]['label'].tolist()
        if active_labels:
            segments.append({
                'segment_start': start,
                'segment_end': end,
                'labels': active_labels
            })

    return pd.DataFrame(segments)


def distribution_per_source(dff: pd.DataFrame) -> None:
    # Create a count plot showing the distribution of labels per source
    plt.figure(figsize=(12, 6))
    sns.countplot(x='source', hue='label', data=dff)
    plt.title('Comparison of Value Counts per Label for Each Source')
    plt.xlabel('Source')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def compare_dataset_sources(train_df, val_df):
    train_counts = train_df['dataset'].value_counts()
    val_counts = val_df['dataset'].value_counts()
    combined = pd.DataFrame({'Train': train_counts, 'Validation': val_counts}).fillna(0)

    combined.plot(kind='bar', figsize=(12, 6))
    plt.title('Dataset Source Distribution')
    plt.xlabel('Dataset')
    plt.show()


def barplot_labels_per_dataset(train_df, val_df):
    def plot_contribution(df, title):
        # Compute counts
        counts = df.groupby(['label', 'dataset']).size().unstack(fill_value=0)

        # Plot
        counts.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='tab20')

        plt.title(title)
        plt.xlabel('Label')
        plt.ylabel('Dataset Contribution per Label')
        plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    plot_contribution(train_df, 'Train Set: Dataset Contribution per Label')
    plot_contribution(val_df, 'Validation Set: Dataset Contribution per Label')


def plot_frequency_distributions(train_df, val_df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    sns.histplot(train_df['low_frequency'], kde=True, ax=axes[0, 0], color='blue')
    axes[0, 0].set_title('Train Low Frequency')

    sns.histplot(val_df['low_frequency'], kde=True, ax=axes[0, 1], color='green')
    axes[0, 1].set_title('Validation Low Frequency')

    sns.histplot(train_df['high_frequency'], kde=True, ax=axes[1, 0], color='blue')
    axes[1, 0].set_title('Train High Frequency')

    sns.histplot(val_df['high_frequency'], kde=True, ax=axes[1, 1], color='green')
    axes[1, 1].set_title('Validation High Frequency')

    plt.tight_layout()
    plt.show()


def plot_call_length(train_df, val_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(train_df['call_length'], kde=True, ax=axes[0], color='blue')
    axes[0].set_title('Train Call Length')

    sns.histplot(val_df['call_length'], kde=True, ax=axes[1], color='green')
    axes[1].set_title('Validation Call Length')

    plt.tight_layout()
    plt.show()
