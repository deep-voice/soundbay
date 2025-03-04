from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataframe_by_label(df, label_column, train_size=0.1, random_state=13):
    # Get unique labels
    labels = df[label_column].unique()
    
    # Initialize empty DataFrames for the split
    df1 = pd.DataFrame()
    
    # Split each label group
    for label in labels:
        label_data = df[df[label_column] == label]
        split1, _ = train_test_split(label_data, train_size=train_size, random_state=random_state)
        df1 = pd.concat([df1, split1])
    
    return df1


dtype_dict = {'filename': 'str'}
metadata_path = 'datasets/mozambique_2018/combined_annotations_filtered_train.csv'

full_meta_df = pd.read_csv(metadata_path, dtype=dtype_dict)

df10 = split_dataframe_by_label(full_meta_df, 'label', train_size=0.1)
df10.to_csv(metadata_path.replace('.csv', '_10.csv'))

df20 = split_dataframe_by_label(full_meta_df, 'label', train_size=0.2)
df20.to_csv(metadata_path.replace('.csv', '_20.csv'))

df30 = split_dataframe_by_label(full_meta_df, 'label', train_size=0.3)
df30.to_csv(metadata_path.replace('.csv', '_30.csv'))

df40 = split_dataframe_by_label(full_meta_df, 'label', train_size=0.4)
df40.to_csv(metadata_path.replace('.csv', '_40.csv'))

df50 = split_dataframe_by_label(full_meta_df, 'label', train_size=0.5)
df50.to_csv(metadata_path.replace('.csv', '_50.csv'))

df60 = split_dataframe_by_label(full_meta_df, 'label', train_size=0.6)
df60.to_csv(metadata_path.replace('.csv', '_60.csv'))

df70 = split_dataframe_by_label(full_meta_df, 'label', train_size=0.7)
df70.to_csv(metadata_path.replace('.csv', '_70.csv'))

df80 = split_dataframe_by_label(full_meta_df, 'label', train_size=0.8)
df80.to_csv(metadata_path.replace('.csv', '_80.csv'))

df90 = split_dataframe_by_label(full_meta_df, 'label', train_size=0.9)
df90.to_csv(metadata_path.replace('.csv', '_90.csv'))

print('fun fun in the sun, data split is done:)')