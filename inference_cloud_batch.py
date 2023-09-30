from typing import Generator, Union
import boto3
import tempfile
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
import hydra
from hydra import compose, initialize
from pathlib import Path
import os
import pandas
import datetime
from hydra.utils import instantiate
import soundfile as sf
from omegaconf import OmegaConf
from typing import List

from soundbay.results_analysis import inference_csv_to_raven
from soundbay.utils.logging import Logger
from soundbay.utils.checkpoint_utils import merge_with_checkpoint
from soundbay.conf_dict import models_dict, datasets_dict
import logging
import json




def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global model_name
    global AWS_ACCESS_KEY_ID
    global AWS_SECRET_ACCESS_KEY
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    # global AWS_ACCESS_KEY_ID
    # global AWS_SECRET_ACCESS_KEY

    # os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    # os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
    # os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    # print(f'aki: {AWS_ACCESS_KEY_ID}')
    # print(f'sak: {AWS_SECRET_ACCESS_KEY}')
    # for name, value in os.environ.items():
    #     print(f'{name}')
    model_dir = os.getenv("AZUREML_MODEL_DIR")

    if AWS_ACCESS_KEY_ID is not None and AWS_SECRET_ACCESS_KEY is not None:
        print('setting aws credentials!')
        model_path = os.path.join(model_dir ,'s3_model.pth')
        # model_path = os.path.join(
        #     model_dir, "model/s3_model.pth"
        # )


        s3_model_path = os.getenv("S3_INPUT_MODEL_PATH")
        s3_model_path_split = s3_model_path.split('/')
        bucket_name = s3_model_path_split[2]
        obj_name = '/'.join(s3_model_path_split[3:])
        print(f'downloading s3 model to infer from bucket: {bucket_name} and obj_name: {obj_name} to model_path: {model_path}')
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, obj_name, model_path)



    elif model_dir is None:
        model_path = './model/best.pth'
    else:
        model_path = os.path.join(
            model_dir, "best.pth"
        )

    # s3_model_path = os.getenv("S3_INPUT_MODEL_PATH")
    # s3_model_path_split = s3_model_path.split('/')
    # bucket_name = s3_model_path_split[2]
    # obj_name = '/'.join(s3_model_path_split[3:])
    # print(f'downloading s3 model from bucket: {bucket_name} and obj_name: {obj_name} to model_path: {model_path}')
    # s3 = boto3.client('s3')
    # s3.download_file(bucket_name, obj_name, model_path)
    if not torch.cuda.is_available():
        ckpt_dict = torch.load(model_path, map_location=torch.device('cpu'))
        print('cuda is not available for ckpt!')
    else:
        ckpt_dict = torch.load(model_path, map_location=torch.device('cuda')) 
        print('cuda is available for ckpt!')

    model_name = Path(model_path).stem
    
    global args
    with initialize(config_path="soundbay/conf", version_base='1.2'):
        args = compose(config_name="/runs/inference_single_audio_manatees")




    ckpt_args = ckpt_dict['args']
    args = merge_with_checkpoint(args, ckpt_args)
    ckpt = ckpt_dict['model']

    checkpoint_state_dict=ckpt
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    model_params = OmegaConf.to_container(args.model.model) 
    model = models_dict[model_params.pop('_target_')](**model_params)

    model.load_state_dict(checkpoint_state_dict)
    model.to(device)
    logging.info("Init complete")





def predict_proba(model: torch.nn.Module, data_loader: DataLoader,
                  device: torch.device = torch.device('cpu'),
                  selected_class_idx: Union[None, int] = None,
                  ) -> np.ndarray:
    """
    calculates the predicted probability to belong to a class for all the samples in the dataset given a specific model
    Input:
        model: the wanted trained model for the inference
        data_loader: dataloader class, containing the dataset location, metadata, batch size etc.
        device: cpu or gpu - torch.device()
        selected_class_idx: the wanted class for prediction. must be bound by the number of classes in the model

    Output:
        softmax_activation: the vector of the predictions of all the samples after a softmax function

    """
    all_predictions = []
    print(f'model device on cuda: {next(model.parameters()).is_cuda}')
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
        softmax_activation = softmax(all_predictions, 1)
        return softmax_activation


def predict(model: torch.nn.Module, data_loader: Generator[torch.tensor, None, None],
            device: torch.device = torch.device('cpu'),
            threshold: Union[float, None] = None, selected_class_idx: int = 1
            ) -> np.ndarray:
    """
   calculates the predicted probability to belong to a class for all the samples in the dataset given a specific model
    Input:
        model: the wanted trained model for the inference
        data_loader: dataloader class, containing the dataset location, metadata, batch size etc.
        device: cpu or gpu - torch.device()
        threshold: a number between 0 and 1 to decide if the classification is positive
        selected_class_idx: the wanted class for prediction. must be bound by the number of classes in the model

    Output:
        prediction: binary prediction given the threshold, the model and the wanted class
    """
    if threshold is None:
        predicted_probability = predict_proba(model, data_loader, device)
        return predicted_probability
    else:
        predicted_probability = predict_proba(model, data_loader, device,
                                              selected_class_idx=selected_class_idx).reshape((-1, 1))
        return (predicted_probability > threshold).reshape((-1, 1))




def infer_multi_file(
        device,
        batch_size,
        dataset_args,
        model_args,
        checkpoint_state_dict,
        output_path,
        # model_name,
        save_raven,
        threshold
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
    mode=dataset_args['mode']
    )

    # load model
    # model = load_model(model_args, checkpoint_state_dict).to(device)

    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=0,
                                 pin_memory=False)

    # predict
    predict_prob = predict_proba(model, test_dataloader, device, None)

    results_df = pandas.DataFrame(predict_prob)
    if hasattr(test_dataset, 'metadata'):
        concat_dataset = pandas.concat([test_dataset.metadata, results_df],
                                       axis=1)  # TODO: make sure metadata column order matches the prediction df order
        metrics_dict = Logger.get_metrics_dict(concat_dataset["label"].values.tolist(),
                                               np.argmax(predict_prob, axis=1).tolist(),
                                               results_df.values)
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

    return filename


def infer_single_file(
        device,
        batch_size,
        dataset_args,
        model_args,
        checkpoint_state_dict,
        output_path,
        # model_name,
        save_raven,
        threshold
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
    # model = load_model(model_args, checkpoint_state_dict).to(device)
    # Check how many channels
    # Temporary load the s3 wav file from dataset_args.file_path
    s3 = boto3.client('s3')

    # s3 = boto3.client('s3')
    # print('downloading s3 file!')
    s3_filename = 's3://deepvoice-datasets/misc/stam_text.txt'
    # get file to local temp file



    # bucket = dataset_args.file_path.split('/')[2]
    # key = '/'.join(dataset_args.file_path.split('/')[3:])
    # with tempfile.TemporaryFile(mode='w+b') as f:
    #     s3.download_fileobj(bucket, key, f)

    num_channels = sf.info(dataset_args.file_path).channels
    all_channel_list = []
    all_channel_raven_list = []
    dataset_args = dict(dataset_args)
    dataset_type = dataset_args.pop('_target_')
    for channel in range(num_channels):
        # set paths and create dataset
        # TODO it's pretty weird to iterate and create dataset for a single file,
        #  the better solution imo should be an inference dataset that can handle multiple channels
        #  and create num samples equal to num channels
        test_dataset = datasets_dict[dataset_type](channel=channel, **dataset_args)
        test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, 
                                     pin_memory=False)

        # predict
        print(f'device for data inference is: {device}')
        predict_prob = predict_proba(model, test_dataloader, device, None)

        results_df = pandas.DataFrame(predict_prob)
        if hasattr(test_dataset, 'metadata'):
            concat_dataset = pandas.concat([test_dataset.metadata, results_df], axis=1) #TODO: make sure metadata column order matches the prediction df order
            metrics_dict = Logger.get_metrics_dict(concat_dataset["label"].values.tolist(),
                                                   np.argmax(predict_prob, axis=1).tolist(),
                                                   results_df.values)
            print(metrics_dict)
        else:
            concat_dataset = results_df
            print("Notice: The dataset has no ground truth labels")

        # create raven file
        if save_raven:
            all_channel_raven_list.append(
                inference_csv_to_raven(results_df, predict_prob.shape[1], dataset_args['seq_length'], 1, threshold, 'call',
                                       channel, dataset_args['data_sample_rate'] // 2)
            )

        # add to general inference result
        concat_dataset.insert(0, 'channel', [channel + 1] * len(concat_dataset))
        all_channel_list.append(concat_dataset)

    #save file
    dataset_name = Path(test_dataset.metadata_path).stem
    filename = f"Inference_results-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{model_name}-{dataset_name}.csv"
    # output_file = os.path.join(output_path , filename)
    output_df = pd.concat(all_channel_list, axis=0)

    # Save raven file
    if save_raven:
        raven_filename = f"Raven inference_results-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{model_name}-{dataset_name}.raven"
        raven_output_file = output_path / raven_filename
        raven_out_df = pd.concat(all_channel_raven_list, axis=0
                  ).sort_values('Begin Time (s)')
        raven_out_df['Selection'] = np.arange(1, len(raven_out_df)+1)
        raven_out_df.to_csv(index=False, path_or_buf=raven_output_file, sep='\t')

    return output_df


def inference_to_file(
    device,
    batch_size,
    dataset_args,
    output_path,
    save_raven,
    threshold,   
    model_args=None,  
    checkpoint_state_dict=None,
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
    if dataset_args._target_.endswith('ClassifierDataset'):
        infer_multi_file(device,
                         batch_size,
                         dataset_args,
                         model_args,
                         checkpoint_state_dict,
                         output_path,
                         save_raven,
                         threshold)
    elif dataset_args._target_.endswith('InferenceDataset'):
        out_df = infer_single_file(
                            device,
                          batch_size,
                          dataset_args,
                          model_args,
                          checkpoint_state_dict,
                          output_path,
                          save_raven,
                          threshold)
    else:
        raise ValueError('Only ClassifierDataset or InferenceDataset allowed in inference')
    return out_df

def run(mini_batch: List[str]) -> pd.DataFrame:
    """
    The main function for running predictions using a trained model on a wanted dataset. The arguments are inserted
    using hydra from a configuration file.
    Input:
        args: from conf file (for instance: main_inference.yaml)
    """
 


     

    for name, value in os.environ.items():
        print(f'{name}')
    os.getenv('AWS_ACCESS_KEY_ID')
    os.getenv('AWS_SECRET_ACCESS_KEY')
    data_sample_rate = os.getenv("DATA_SAMPLE_RATE")
    print(f'data_sample_rate: {data_sample_rate}')
    args.data.data_sample_rate = int(data_sample_rate)
    # inference_filename = os.getenv("INFERENCE_FILENAME")
    # print("downloading from s3: ", inference_filename)
    # infer_split = inference_filename.split('/')
    # bucket = infer_split[2]
    # # key path
    # key = '/'.join(infer_split[3:])
    # s3.download_file(bucket, key, 'infer_file.wav')
    # # args.data.test_dataset.file_path = 'infer_file.wav'

    # inference_filename = 'infer_file.wav'
    inference_filename = '/mnt/batch_file.wav'


    s3_data_path = os.getenv("S3_INPUT_DATA_PATH")
    s3_data_path_split = s3_data_path.split('/')
    bucket_name = s3_data_path_split[2]
    obj_name = '/'.join(s3_data_path_split[3:])
    print(f'downloading s3 file to infer from bucket: {bucket_name} and obj_name: {obj_name} to file_path: {inference_filename}')
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, obj_name, inference_filename)

    print("inference_filename to infer: ", inference_filename)
    args.data.test_dataset.file_path = str(inference_filename)




    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    

    out_df = inference_to_file(
        device=device,
        batch_size=args.data.batch_size,
        dataset_args=args.data.test_dataset,
        output_path=args.experiment.output_path,
        save_raven=args.experiment.save_raven,
        threshold=args.experiment.threshold
    )
    # upload out_df to s3
    s3_output_data_path = os.getenv("S3_OUTPUT_DATA_PATH")
    s3_output_data_path_split = s3_output_data_path.split('/')
    bucket_name = s3_output_data_path_split[2]
    obj_name = '/'.join(s3_output_data_path_split[3:])
    # save csv to temp file
    preds_filename_local = '/mnt/preds.csv'
    # with tempfile.TemporaryFile(mode='w+b') as f:
    print('saving csv to temp file and uploading to s3')
    out_df.to_csv(preds_filename_local)
    s3.upload_file(preds_filename_local, bucket_name, obj_name)    
    print("Finished inference upload")
    logging.info("Request processed")
    return out_df
    


