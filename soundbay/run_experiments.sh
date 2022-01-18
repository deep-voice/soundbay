#!/bin/bash
METADATAS=(../datasets/orcasound/training_splits_to_test/all_together_bg_from_calls_train_val_splits.csv
../datasets/orcasound/training_splits_to_test/all_together_train_val_splits.csv
../datasets/orcasound/training_splits_to_test/without_3_bg_from_calls_train_val_splits.csv
../datasets/orcasound/training_splits_to_test/without_3_train_val_splits.csv
../datasets/orcasound/training_splits_to_test/without_6_bg_from_calls_train_val_splits.csv
../datasets/orcasound/training_splits_to_test/without_6_train_val_splits.csv
../datasets/orcasound/training_splits_to_test/without_7_11_12_train_val_splits.csv
)

NAMES=(all_together_bg
all_together
without_3_bg
without_3
without_6_bg
without_6
without_7_11_12
)

for index in ${!METADATAS[*]}; do
  python train.py --config-name runs/orcasound data.train_dataset.metadata_path=${METADATAS[$index]} \
  data.val_dataset.metadata_path=${METADATAS[$index]} ${array2[$index]} experiment.name=${NAMES[$index]} \
  experiment.run_id=${NAMES[$index]} experiment.group_name="Orcasound_Splits" optim.epochs=20;
  sleep 30;
done
