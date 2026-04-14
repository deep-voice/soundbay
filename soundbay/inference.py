from typing import Union, Literal

import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from scipy.special import expit
import hydra
from pathlib import Path
import os
import pandas
import datetime
from omegaconf import OmegaConf

import librosa
import soundfile as sf

from soundbay.results_analysis import inference_csv_to_raven
from soundbay.utils.logging import Logger
from soundbay.utils.checkpoint_utils import merge_with_checkpoint
from soundbay.conf_dict import models_dict, datasets_dict

from soundbay.utils.metrics import MetricsCalculator


def predict_proba(model: torch.nn.Module, data_loader: DataLoader,
                  device: torch.device = torch.device('cpu'),
                  selected_class_idx: Union[None, int] = None,
                  proba_norm_func: Literal["softmax", "sigmoid"] = 'softmax'
                  ) -> np.ndarray:
    """
    calculates the predicted probability to belong to a class for all the samples in the dataset given a specific model
    Input:
        model: the wanted trained model for the inference
        data_loader: dataloader class, containing the dataset location, metadata, batch size etc.
        device: cpu or gpu - torch.device()
        selected_class_idx: the wanted class for prediction. must be bound by the number of classes in the model

    Output:
        proba: the vector of the predictions normalized to probabilities.

    """
    assert proba_norm_func in ['softmax', 'sigmoid'], 'proba_norm_func must be softmax or sigmoid'
    all_predictions = []
    with torch.no_grad():
        model.eval()
        for audio in tqdm(data_loader):
            audio = audio.to(device)

            predicted_probability = model(audio).cpu().numpy()
            if selected_class_idx is None:
                all_predictions.extend(predicted_probability)
            else:
                if selected_class_idx in list(range(predicted_probability.shape[1])):
                    return predicted_probability[:, selected_class_idx]
                else:
                    raise ValueError(f'selected class index {selected_class_idx} not in output dimensions')
        if proba_norm_func == 'softmax':
            proba = softmax(all_predictions, 1)
        elif proba_norm_func == 'sigmoid':
            proba = expit(all_predictions)
        else:
            raise ValueError('proba_norm_func must be softmax or sigmoid')
        return proba


def load_model(model_params, checkpoint_state_dict):
    """
    load_model receives model params and state dict, instantiating a model and loading trained parameters.
    Input:
        model_params: config arguments of model object
        checkpoint_state_dict: dict including the train parameters to be loaded to the model
    Output:
        model: nn.Module object of the model
    """

    model_params = OmegaConf.to_container(model_params) 
    model = models_dict[model_params.pop('_target_')](**model_params)
    model.load_state_dict(checkpoint_state_dict)
    return model


def infer_with_metadata(
        device,
        batch_size,
        dataset_args,
        model_args,
        checkpoint_state_dict,
        output_path,
        model_name,
        proba_norm_func,
        label_type
):
    """
        This functions takes the ClassifierDataset dataset and produces the model prediction to a file
        Input:
            device: cpu/gpu
            batch_size: the number of samples the model will infer at once
            dataset_args: the required arguments for the dataset class
            model_path: directory for the wanted trained model
            output_path: directory to save the prediction file
        """
    # set paths and create dataset
    test_dataset = datasets_dict[dataset_args['_target_']](data_path = dataset_args['data_path'],
    metadata_path=dataset_args['metadata_path'], augmentations=dataset_args['augmentations'],
    augmentations_p=dataset_args['augmentations_p'],
    preprocessors=dataset_args['preprocessors'],
    seq_length=dataset_args['seq_length'], data_sample_rate=dataset_args['data_sample_rate'],
    sample_rate=dataset_args['sample_rate'], 
    mode=dataset_args['mode'], slice_flag=dataset_args['slice_flag'], path_hierarchy=dataset_args['path_hierarchy'],
    label_type=label_type
    )

    # load model
    model = load_model(model_args, checkpoint_state_dict).to(device)

    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=0,
                                 pin_memory=False)

    # predict
    predict_prob = predict_proba(model, test_dataloader, device, None, proba_norm_func)

    results_df = pandas.DataFrame(predict_prob)  # add class names
    if hasattr(test_dataset, 'metadata'):
        concat_dataset = pandas.concat([test_dataset.metadata, results_df],
                                       axis=1)  # TODO: make sure metadata column order matches the prediction df order
        metrics_dict = MetricsCalculator(
                label_list=concat_dataset["label"].values.tolist(),
                pred_list=np.argmax(predict_prob, axis=1).tolist(),
                pred_proba_list=predict_prob,
                label_type=label_type).calc_all_metrics()
        print(metrics_dict)
    else:
        concat_dataset = results_df
        print("Notice: The dataset has no ground truth labels")

    # save file
    dataset_name = Path(test_dataset.metadata_path).stem
    filename = f"Inference_results-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{model_name}-{dataset_name}.csv"
    output_file = output_path / filename
    concat_dataset.to_csv(index=False, path_or_buf=output_file)

    # save raven file

    return


def infer_without_metadata(
        device,
        batch_size,
        dataset_args,
        model_args,
        checkpoint_state_dict,
        output_path,
        model_name,
        save_raven,
        threshold,
        label_names,
        raven_max_freq,
        proba_norm_func,
        kwargs
):
    """
        This functions takes the InferenceDataset dataset and produces the model prediction to a file, by iterating
        on all channels in the file and concatenating the results.
        Input:
            device: cpu/gpu
            batch_size: the number of samples the model will infer at once
            dataset_args: the required arguments for the dataset class
            model_path: directory for the wanted trained model
            output_path: directory to save the prediction file
    """
    # load model
    model = load_model(model_args, checkpoint_state_dict).to(device)
    all_raven_list = []
    dataset_args = dict(dataset_args)
    dataset_type = dataset_args.pop('_target_')
    test_dataset = datasets_dict[dataset_type](**dataset_args)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=0,
                                 pin_memory=False)

    # predict
    predict_prob = predict_proba(model, test_dataloader, device, None, proba_norm_func)
    label_names = ['Noise'] + [f'Call_{i}' for i in
                               range(1, predict_prob.shape[1] + 1)] if label_names is None else label_names

    results_df = pandas.DataFrame(predict_prob, columns=label_names)

    concat_dataset = pandas.concat([test_dataset.metadata, results_df], axis=1)
    # create raven file
    raven_max_freq = dataset_args['sample_rate'] // 2 if raven_max_freq is None else raven_max_freq
    if save_raven:
        for file, df in concat_dataset.groupby('filename'):
            file_raven_lists = []
            for i in range(1, predict_prob.shape[1]):
                file_raven_lists.append(
                                       inference_csv_to_raven(results_df=df,
                                                              num_classes=predict_prob.shape[1],
                                                              seq_len=dataset_args['seq_length'],
                                                              selected_class=label_names[i],
                                                              threshold=threshold,
                                                              class_name=label_names[i],
                                                              max_freq=raven_max_freq)
                                   )
            whole_file_df = pd.concat(file_raven_lists, axis=0).sort_values('Begin Time (s)')
            whole_file_df['Selection'] = np.arange(1, len(whole_file_df) + 1)
            all_raven_list.append((file, whole_file_df))

    #save file
    dataset_name = Path(test_dataset.metadata_path).stem
    filename = f"Inference_results-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{model_name}-{dataset_name}.csv"
    output_file = output_path / filename
    concat_dataset = concat_dataset.sort_values(by=['filename', 'begin_time'])
    concat_dataset.to_csv(index=False, path_or_buf=output_file)

    # Save raven file
    if save_raven:
        if Path(test_dataset.metadata_path).is_dir():
            output_path = output_path / dataset_name
            output_path.mkdir(exist_ok=True)
        for filename, raven_out_df in all_raven_list:
            save_raven_file(filename, raven_out_df, output_path, model_name)

    return

def infer_proba(
        device,
        batch_size,
        dataset_args,
        model_args,
        checkpoint_state_dict,
        label_names,
        proba_norm_func
):
    """
        This functions takes the InferenceDataset dataset and produces the model prediction to a file, by iterating
        on all channels in the file and concatenating the results.
        Input:
            device: cpu/gpu
            batch_size: the number of samples the model will infer at once
            dataset_args: the required arguments for the dataset class
            model_path: directory for the wanted trained model
            output_path: directory to save the prediction file
    """
    # load model
    model = load_model(model_args, checkpoint_state_dict).to(device)
    all_raven_list = []
    dataset_args = dict(dataset_args)
    dataset_type = dataset_args.pop('_target_')
    test_dataset = datasets_dict[dataset_type](**dataset_args)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=0,
                                 pin_memory=False)

    # predict
    predict_prob = predict_proba(model, test_dataloader, device, None, proba_norm_func)
    label_names = ['Noise'] + [f'Call_{i}' for i in
                               range(1, predict_prob.shape[1] + 1)] if label_names is None else label_names

    results_df = pandas.DataFrame(predict_prob, columns=label_names)
    return results_df


def save_raven_file(filename, raven_out_df, output_path, model_name):
    raven_filename = f"{filename.stem}-Raven-inference_results-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{model_name}.txt"
    raven_output_file = output_path / raven_filename
    raven_out_df.to_csv(index=False, path_or_buf=raven_output_file, sep='\t')


def inference_to_file(
    device,
    batch_size,
    dataset_args,
    model_args,
    checkpoint_state_dict,
    output_path,
    model_name,
    save_raven,
    threshold,
    label_names,
    raven_max_freq,
    proba_norm_func,
    label_type
):
    """
    This functions takes the dataset and produces the model prediction to a file
    Input:
        device: cpu/gpu
        batch_size: the number of samples the model will infer at once
        dataset_args: the required arguments for the dataset class
        model_path: directory for the wanted trained model
        output_path: directory to save the prediction file
    """
    if dataset_args._target_.endswith('ClassifierDataset') or dataset_args._target_.endswith('NoBackGroundDataset'):
        infer_with_metadata(device,
                         batch_size,
                         dataset_args,
                         model_args,
                         checkpoint_state_dict,
                         output_path,
                         model_name,
                         proba_norm_func,
                         label_type
                         )
    elif dataset_args._target_.endswith('InferenceDataset'):
        infer_without_metadata(device,
                          batch_size,
                          dataset_args,
                          model_args,
                          checkpoint_state_dict,
                          output_path,
                          model_name,
                          save_raven,
                          threshold,
                          label_names,
                          raven_max_freq,
                          proba_norm_func)
    else:
        raise ValueError('Only ClassifierDataset or InferenceDataset allowed in inference')

def get_start_end_times(preds, seq_length, overlap, threshold=0.98):
    times = []
    start_time = 0.0
    end_time = 0.0
    prev_val = False
    for i in range(len(preds)):
        curr = preds[i] >= threshold
        if curr and not prev_val:
            start_time = i * seq_length * (1 - overlap)
        if not curr and prev_val:
            end_time = (i-1) * seq_length * (1 - overlap)
            times.append({'start_time': start_time, 'end_time': end_time, 'duration': end_time - start_time, 'prob': preds[i-1]})
        prev_val = curr
    if prev_val:
        end_time = (len(preds) - 1) * seq_length * (1 - overlap)
        times.append({'start_time': start_time, 'end_time': end_time, 'duration': end_time - start_time})
    return times

def get_predictions(preds, seq_length, overlap, threshold=0.98, min_time=5.0):
    preds = get_start_end_times(preds, seq_length, overlap, threshold)
    filtered_preds = []
    for v in preds:
        if v['duration'] >= min_time:
            filtered_preds.append(v)
    return filtered_preds

def get_frequency_boundaries(segment, sr, percentile=95, min_freq_threshold=4, max_freq_threshold=None):
    spec = librosa.stft(segment, n_fft=512, hop_length=256)
    # spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
    spec_db = np.abs(spec)

    # remove time frames with abnormally high broadband power
    time_power = np.sum(spec_db ** 2, axis=0)
    q1, q3 = np.percentile(time_power, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        upper_thresh = time_power.mean() + 3 * time_power.std()
    else:
        upper_thresh = q3 + 3 * iqr
    keep_mask = time_power <= upper_thresh
    print(f'Removed {np.sum(~keep_mask)} out of {len(keep_mask)} time frames due to high broadband power.')
    if not np.any(keep_mask):  # fallback: keep all if everything is flagged
        keep_mask = np.ones_like(time_power, dtype=bool)
    spec_db = spec_db[:, keep_mask]

    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    freqs_diff = freqs[1] - freqs[0]
    spec_db = spec_db[freqs >= min_freq_threshold, :]
    freqs = freqs[freqs >= min_freq_threshold]
    if max_freq_threshold is not None:
        spec_db = spec_db[freqs <= max_freq_threshold, :]
        freqs = freqs[freqs <= max_freq_threshold]

    mean_spec_db = np.mean(spec_db, axis=1)
    
    threshold_db = np.percentile(mean_spec_db, percentile)
    min_freq = None
    max_freq = None
    for f, db in zip(freqs, mean_spec_db):
        if db >= threshold_db:
            if min_freq is None:
                min_freq = f - freqs_diff / 2
            max_freq = f + freqs_diff / 2
    return min_freq, max_freq

def convert_preds_to_raven_format(predictions, class_name, waveform, sr, channel=1, min_time_sec = 2):
    raven_bboxes = {
        'Selection': [],
        'Begin Time (s)': [],
        'End Time (s)': [],
        'Low Freq (Hz)': [],
        'High Freq (Hz)': [],
        'Channel': [],
        'Class Name': [],
        'Probability': []
    }
    cnt = 1
    for v in predictions:
        if (v['end_time'] - v['start_time']) < min_time_sec:
            continue
        segment = waveform[int(v['start_time'] * sr): int(v['end_time'] * sr)]
        low_freq, high_freq = get_frequency_boundaries(segment, sr, percentile=90, min_freq_threshold=15, max_freq_threshold=35)
        if high_freq > 50:
            continue  # skip this prediction if frequency band is too wide

        raven_bboxes['Selection'].append(int(cnt))
        cnt += 1
        raven_bboxes['Begin Time (s)'].append(v['start_time'])
        raven_bboxes['End Time (s)'].append(v['end_time'])

        # calculate frequency bounds
        
        if low_freq is not None and high_freq is not None:
            raven_bboxes['Low Freq (Hz)'].append(round(low_freq, 2))
            raven_bboxes['High Freq (Hz)'].append(round(high_freq, 2))
        else:
            raven_bboxes['Low Freq (Hz)'].append(0)
            raven_bboxes['High Freq (Hz)'].append(100)
        raven_bboxes['Channel'].append(channel)
        raven_bboxes['Class Name'].append(class_name)
        raven_bboxes['Probability'].append(v.get('prob', 1.0))
    return raven_bboxes

def continous_inference_to_raven_file(
    device,
    batch_size,
    dataset_args,
    model_args,
    checkpoint_state_dict,
    output_path,
    model_name,
    threshold,
    label_names,
    raven_max_freq,
    proba_norm_func,
    selected_class_idx=None,
    minimum_time_sec=5.0
):
    model = load_model(model_args, checkpoint_state_dict).to(device)
    # all_raven_list = []
    dataset_args = dict(dataset_args)
    dataset_type = dataset_args.pop('_target_')
    test_dataset = datasets_dict[dataset_type](**dataset_args)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=0,
                                 pin_memory=False)

    # predict
    # predict_prob = predict_proba(model, test_dataloader, device, selected_class_idx, proba_norm_func)
    all_predictions = []
    with torch.no_grad():
        model.eval()
        for audio in tqdm(test_dataloader):
            audio = audio.to(device)
            outputs = model(audio).cpu().numpy()
            probs = softmax(outputs, axis=1)
            all_predictions.append(probs)
    all_predictions = np.vstack(all_predictions)

    # label_names = [f'Call_{i}' for i in
    #                            range(1, predict_prob.shape[1] + 1)] if label_names is None else label_names
    
    pred_df = get_predictions(all_predictions[:, selected_class_idx], dataset_args['seq_length'], dataset_args['overlap'], threshold, min_time=minimum_time_sec)

    # create raven file
    waveform, sr = sf.read(dataset_args['file_path'])
    raven_class = convert_preds_to_raven_format(pred_df, 
                                                class_name=label_names[selected_class_idx], 
                                                channel=1, min_time_sec=int(minimum_time_sec),
                                                waveform=waveform, sr=sr)
    raven_df = pd.DataFrame(raven_class)
    raven_df = raven_df.sort_values(by='Begin Time (s)').reset_index(drop=True)
    file_name = Path(test_dataset.metadata_path).stem
    raven_df.to_csv(index=False, path_or_buf=f'./outputs/TwoClasses_{file_name}_predictions_raven.txt', sep='\t')

    # #save file
    # dataset_name = Path(test_dataset.metadata_path).stem
    # filename = f"Inference_results-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{model_name}-{dataset_name}.csv"
    # output_file = output_path / filename
    # concat_dataset = concat_dataset.sort_values(by=['filename', 'begin_time'])
    # concat_dataset.to_csv(index=False, path_or_buf=output_file)

    # # Save raven file
    # if save_raven:
    #     if Path(test_dataset.metadata_path).is_dir():
    #         output_path = output_path / dataset_name
    #         output_path.mkdir(exist_ok=True)
    #     for filename, raven_out_df in all_raven_list:
    #         save_raven_file(filename, raven_out_df, output_path, model_name)

    return

@hydra.main(config_name="/runs/main_inference.yaml", config_path="conf", version_base='1.2')
def inference_main(args) -> None:
    """
    The main function for running predictions using a trained model on a wanted dataset. The arguments are inserted
    using hydra from a configuration file.
    Input:
        args: from conf file (for instance: main_inference.yaml)
    """
    # set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    working_dirpath = Path(hydra.utils.get_original_cwd())
    os.chdir(working_dirpath)
    output_dirpath = working_dirpath.parent.absolute() / "outputs"
    output_dirpath.mkdir(exist_ok=True)

    ckpt_dict = torch.load(args.experiment.checkpoint.path, map_location=torch.device('cpu'), weights_only=False)
    ckpt_args = ckpt_dict['args']
    args = merge_with_checkpoint(args, ckpt_args)
    ckpt = ckpt_dict['model']

    default_norm_func = 'softmax' if args.data.label_type == 'single_label' else 'sigmoid'

    if args.experiment.continous:
        if args.experiment.save_raven:
            continous_inference_to_raven_file(
                device=device,
                batch_size=args.data.batch_size,
                dataset_args=args.data.test_dataset,
                model_args=args.model.model,
                checkpoint_state_dict=ckpt,
                output_path=output_dirpath,
                model_name=Path(args.experiment.checkpoint.path).parent.stem,
                threshold=args.experiment.threshold,
                label_names=args.data.label_names,
                raven_max_freq=args.experiment.raven_max_freq,
                proba_norm_func=args.data.get('proba_norm_func', default_norm_func), # using "get" for backward compatibility,
                selected_class_idx=args.data.get('selected_class_idx', 0),
                minimum_time_sec=args.experiment.get('minimum_time_sec', 5.0)
            )
        else:
            raise NotImplementedError("Continuous inference is implemented only to raven file")
    else:
        if args.experiment.save_raven:
            inference_to_file(
                device=device,
                batch_size=args.data.batch_size,
                dataset_args=args.data.test_dataset,
                model_args=args.model.model,
                checkpoint_state_dict=ckpt,
                output_path=output_dirpath,
                model_name=Path(args.experiment.checkpoint.path).parent.stem,
                save_raven=args.experiment.save_raven,
                threshold=args.experiment.threshold,
                label_names=args.data.label_names,
                raven_max_freq=args.experiment.raven_max_freq,
                proba_norm_func=args.data.get('proba_norm_func', default_norm_func), # using "get" for backward compatibility,
                label_type=args.data.label_type
            )
        else:
            results_df = infer_proba(
                device=device,
                batch_size=args.data.batch_size,
                dataset_args=args.data.test_dataset,
                model_args=args.model.model,
                checkpoint_state_dict=ckpt,
                label_names=args.data.label_names,
                proba_norm_func=args.data.get('proba_norm_func', default_norm_func) # using "get" for backward compatibility,
            )
            print(results_df)
            results_df.to_csv(index=False, path_or_buf=output_dirpath / f"inference_results-{Path(args.experiment.checkpoint.path).parent.stem}.csv")
    
    print("Finished inference")

if __name__ == "__main__":
    inference_main()
