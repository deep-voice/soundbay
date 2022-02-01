The dataset is a processed version of the data available in [OrcaData](https://github.com/orcasound/orcadata) by [OrcaSound](https://github.com/orcasound).

Calls of exactly 2.45 seconds were removed since many of them are machine labeled.

Original files were resamples do the same sample rate of 20000 Hz.

* The dataset can be downloaded using the following command:

```
aws --no-sign-request s3 cp s3://deepvoice-external/OrcaSound/orcasound_processed_train_data.tar.gz .
```

* Extract the dataset to the correct folder:

```
mv orcasound_processed_train_data.tar.gz <path-to-soundbay>/datasets/orcasound
tar -xzf orcasound_processed_train_data.tar.gz .
```

* To train with this dataset simply use the data conf file related to it:

```
cd <path-to-soundbay>/src
python train.py data=orcasound
```

