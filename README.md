<img src="figs/logo_deepvoice.png" width="400"/>

## Deep Learning Framework for Bioacoustics
Soundbay is an intuitive and comprehensive Python-based framework for training deep learning models and apply them in bioacoustic research. We focus our efforts on marine mammals communication, however, we believe the framework is applicable to a broader scope.

## Installation
Before installing, please verify you have a working Python environment, with pytorch, torchvision and torchaudio installed according to your local hardware.<br>
More info on installing pytorch, torchaudio and torchvision can be found in the [guide](https://pytorch.org/get-started/locally/).
Make sure to use python 3.8-3.12 (as of mid-2025 python 3.13 can cause dependency issues)

```sh
git clone https://github.com/deep-voice/soundbay
cd soundbay
```
Installation of the packages:
```sh
pip install -e .
```

Or
```sh
pip install -r requirements.txt
```

[sox](http://sox.sourceforge.net/) is also required and can be installed using:
```
sudo apt update
sudo apt install sox
``` 

## Usage

### Experiment management philosophy 
The framework uses standard YAML files for configuration management, handled by OmegaConf and Pydantic for validation.
We provide modular recipes inside [conf](soundbay/conf/) to run experiments.
Configuration is strictly typed and validated before execution.

### Data structure
Path to the datafolder should be passed as an argument for training. The data folder or subfolders should contain `.wav` files. 
A `.csv` file should accompany the data, serve as metadata from which the training pipeline takes samples with their corresponding labels.
A toy example is available for [data](tests/assets/data) and the respective [annotations](tests/assets/annotations/sample_annotations.csv)

### Training Example

Running train and inference commands are done from src.
Run the following command for a toy problem training:
```sh
python soundbay/train.py --config soundbay/conf/runs/main.yaml
```
The toy run uses the default configuration from the provided yaml file.
Parameters can be overridden from the command line using dot notation:
```sh
python soundbay/train.py --config soundbay/conf/runs/main_unit_norm.yaml data.batch_size=8 experiment.manual_seed=4321
```
This runs training with the specified config file, overriding the batch_size in the data section and manual_seed in the experiment section. 

### inference Example
To run the predictions of the model on a single audio file use the inference script:
```sh
python soundbay/inference.py --config soundbay/conf/runs/inference_single_audio.yaml --checkpoint <PATH/TO/MODEL/CHECKPOINT> --file <PATH/TO/FILE>
```
To run the predictions of the model on a labeled test partition use:
```sh
python soundbay/inference.py --config soundbay/conf/runs/main_inference.yaml --checkpoint <PATH/TO/MODEL/CHECKPOINT> data.test_dataset.data_path=<PATH/TO/DATA> data.test_dataset.metadata_path=<PATH/TO/METADATA> 
```

## License

This library is licensed under the GNU Affero General Public License v3.0 License.
