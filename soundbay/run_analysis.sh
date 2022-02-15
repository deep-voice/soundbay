#!/bin/bash

DATASETS3=('../datasets/orcasound/orcasound_test_podcast3.csv'
'../datasets/orcasound/orcasound_test_podcast3_bg_from_calls.csv'
)

DATASETS6=('../datasets/orcasound/orcasound_test_podcast6.csv'
'../datasets/orcasound/orcasound_test_podcast6_bg_from_calls.csv'
)

CHECKPOINTS=$(ls -1 ../models/orca_splits/*/*/*/best.pth)

for checkpoint in $CHECKPOINTS
  do
    for dataset in "${DATASETS3[@]}"
      do
        python inference.py --config-name runs/main_inference checkpoint.path=$checkpoint \
         data=orcasound_testset_inference data.test_dataset.metadata_path=$dataset;
        sleep 2;
    done
    for dataset in "${DATASETS6[@]}"
      do
        python inference.py --config-name runs/main_inference checkpoint.path=$checkpoint \
         data=orcasound_testset_inference data.test_dataset.metadata_path=$dataset \
         data.test_dataset.data_path='../datasets/orcasound/train_data/wav';
        sleep 2;
    done
done