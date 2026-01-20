from pathlib import Path
import pandas as pd
import os
from soundbay.utils.metadata_processing import multi_target_from_time_intervals_df

if __name__ == "__main__":
    base_path = Path(os.getcwd())
    fannie_path = base_path / "datasets" / "fannie_project"
    
    train_annotation_file = fannie_path / "train_updated.csv"
    df = pd.read_csv(train_annotation_file)

    df.label = df.groupby('filename', as_index=False).apply(multi_target_from_time_intervals_df, n_classes=3, overlap_threshold_pct=0.0, noise_class_value=0).reset_index(level=0, drop=True)

    df.to_csv(fannie_path / "train_multi_labels.csv", index=False)
    
    # validation
    val_annotation_file = fannie_path / "val_updated.csv"
    df = pd.read_csv(val_annotation_file)
    df.label = df.groupby('filename', as_index=False).apply(multi_target_from_time_intervals_df, n_classes=3, overlap_threshold_pct=0.0, noise_class_value=0).reset_index(level=0, drop=True)
    df.to_csv(fannie_path / "val_multi_labels.csv", index=False)

    # test
    test_annotation_file = fannie_path / "test_updated.csv"
    df = pd.read_csv(test_annotation_file)
    
    df.label = df.groupby('filename', as_index=False).apply(multi_target_from_time_intervals_df, n_classes=3, overlap_threshold_pct=0.0, noise_class_value=0).reset_index(level=0, drop=True)
    
    df.to_csv(fannie_path / "test_multi_labels.csv", index=False)
