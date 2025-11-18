import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
import yaml

from dclde_2026 import config, dataset, utils


def save_yolo_data(df_chip_split, df_ann, audio_loader, split_name):
    """
    Iterates over a df split, generates data, and saves to disk.
    """
    img_dir = os.path.join(config.YOLO_PREPROC_DIR, split_name, 'images')
    lbl_dir = os.path.join(config.YOLO_PREPROC_DIR, split_name, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    # Create a dataset instance (NOT a dataloader)
    ds = dataset.AudioObjectDataset(
        df_chip_split, 
        df_ann, 
        audio_loader, 
        is_train=(split_name == 'train'),
        use_beats=False,  # We want spectrograms for YOLO
        model_type='yolo'  # Enable YOLO-specific processing
    )

    print(f"Generating and saving {split_name} data...")
    for i in tqdm(range(len(ds))):
        try:
            # 1. Generate spectrogram and labels
            specs_norm, labels_tensor = ds[i]
            if specs_norm is None:
                continue

            # 2. Get 3-channel [0, 255] image
            # specs_norm is [3, H, W] and [0, 1]
            img = (specs_norm.numpy() * 255).astype(np.uint8)
            # Transpose to [H, W, 3] for saving
            img = np.transpose(img, (1, 2, 0))

            # 3. Save image at natural dimensions (no resizing)
            img_filename = f"chip_{i:06d}.png"
            cv2.imwrite(os.path.join(img_dir, img_filename), img)


            # 5. Save labels
            # labels_tensor is (N, 5) with (class_id, x, y, w, h)
            # This is already the correct format!
            lbl_filename = f"chip_{i:06d}.txt"
            np.savetxt(
                os.path.join(lbl_dir, lbl_filename),
                labels_tensor.numpy(),
                fmt="%.6f"
            )
        except Exception as e:
            print(f"Warning: Failed to process chip {i}. {e}")


def main():
    # 1. Load annotation and chip data
    df_ann = dataset.create_full_annotation_df()
    df_chips = dataset.create_chip_list(df_ann)
    
    # 2. Create splits
    gkf = GroupKFold(n_splits=config.N_SPLITS)
    splits = list(gkf.split(df_chips, groups=df_chips['group']))
    
    val_fold = 0
    test_fold = 1
    
    val_idx = splits[val_fold][1]
    test_idx = splits[test_fold][1]
    all_idx = set(range(len(df_chips)))
    train_idx = list(all_idx - set(val_idx) - set(test_idx))

    df_train = df_chips.iloc[train_idx].reset_index(drop=True)
    df_val = df_chips.iloc[val_idx].reset_index(drop=True)
    df_test = df_chips.iloc[test_idx].reset_index(drop=True)
    
    audio_loader = dataset.GCSAudioLoader(
        bucket_name=config.GCS_AUDIO_BUCKET_NAME,
        cache_dir=config.LOCAL_AUDIO_CACHE_DIR if config.ENABLE_AUDIO_CACHE else None,
        enable_cache=config.ENABLE_AUDIO_CACHE
    )

    # 3. Process and save each split
    save_yolo_data(df_train, df_ann, audio_loader, 'train')
    save_yolo_data(df_val, df_ann, audio_loader, 'val')
    save_yolo_data(df_test, df_ann, audio_loader, 'test')


    # 4. Create data.yaml
    yaml_data = {
        'train': os.path.join(config.YOLO_PREPROC_DIR, 'train', 'images'),
        'val': os.path.join(config.YOLO_PREPROC_DIR, 'val', 'images'),
        'test': os.path.join(config.YOLO_PREPROC_DIR, 'test', 'images'),
        'nc': config.NUM_CLASSES,
        'names': config.CLASSES  # ['AB', 'HW', 'KW', 'UndBio']
    }
    
    with open('dclde_2026/data.yaml', 'w') as f:
        yaml.dump(yaml_data, f)
        
    print("---" * 10)
    print("YOLO data preprocessing complete!")
    print(f"Dataset saved to: {config.YOLO_PREPROC_DIR}")
    print(f"Config file saved to: dclde_2026/data.yaml")
    print("\nTo train YOLOv8, run:")
    print(f"yolo train data=dclde_2026/data.yaml model=yolov8n.pt imgsz={config.YOLO_IMG_SIZE} ...")


if __name__ == "__main__":
    main()

