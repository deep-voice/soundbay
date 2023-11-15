import click
from pathlib import Path
from typing import Optional
import pandas as pd


@click.command()
@click.option('--dataset-path', required=True, type=Path, help='Dataset path')
@click.option('--split-type', type=click.Choice(['train', 'test', 'val']), default=None, help='split type')
def main(dataset_path: Path, split_type: Optional[str]):
    print(f"dataset: {dataset_path}, model: {split_type}")
    df = pd.read_csv(dataset_path)
    if split_type is not None:
        assert 'split_type' in df.columns, f"split_type column not found in {dataset_path}"
        df = df[df['split_type'] == split_type]
    out_dict = {}
    for label, group in df.groupby(['label']):
        group_len = len(group)
        total_audio = group['call_length'].sum()
        out_dict[label] = {'Number of calls': group_len, 'Total Audio length (sec)': f'{total_audio:.3f}',
                           'Total Audio length (hr)': f'{total_audio / 3600:.3f}'}
    print(out_dict)


if __name__ == '__main__':
    main()

