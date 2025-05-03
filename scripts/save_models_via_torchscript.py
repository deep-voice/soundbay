import torch
import click
from soundbay.encapsulated_models import EfficientNet2D

# ad-hoc script to save an example model via torchscript, create something more general later
@click.command()
@click.option("--model-path", "-p", type=str, help="The path to the model to save.")
@click.option("--output-path", "-o", type=str, help="The path to save the model.")
def save_models_via_torchscript(model_path, output_path):
    ckpt_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    ckpt_args = ckpt_dict['args']
    ckpt = ckpt_dict['model']
    melspec_kwargs = ckpt_args._preprocessors.mel_spectrogram
    del melspec_kwargs._target_
    model = EfficientNet2D(num_classes=3, pretrained=True, melspec_kwargs=melspec_kwargs)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.to("cpu")
    traced_model = torch.jit.trace(model, torch.randn(1, 1, 2000))
    traced_model.save(output_path)

if __name__ == "__main__":
    save_models_via_torchscript()