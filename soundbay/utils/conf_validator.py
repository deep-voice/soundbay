from typing import Optional
from pydantic import BaseModel, validator
from conf_dict import datasets_dict, criterion_dict, models_dict 

class Dataset(BaseModel):
    batch_size: int
    num_workers: int
    sample_rate: int
    data_sample_rate: int
    max_freq: int
    label_type: str
    min_freq: int
    n_fft: int
    hop_length: int
    train_dataset: dict
    val_dataset: dict
    class Config:
        title = "Dataset"
        allow_mutation = False
        validate_assignment = True
        anystr_lower = False
        validate_all = True
        use_enum_values = True

    @validator("num_workers")
    def validate_num_workers(cls, num_workers:int):
        if num_workers < 0:
            raise ValueError(f"Number of works is smaller than 0 {num_workers}")
        return num_workers


    @validator("train_dataset")
    def validate_train_dataset(cls, train_dataset:dict) :
        if train_dataset['_target_'] not in datasets_dict.keys(): 
            raise ValueError(f"Train_dataset is not allowed from type {train_dataset['_target_']}")
        return train_dataset


    @validator("data_sample_rate")
    def validate_data_sample_rate(cls, data_sample_rate:int) :
        if data_sample_rate < 0: 
            raise ValueError(f"data_sample_rate must be a positive integer")
        return data_sample_rate 

    @validator("sample_rate")
    def validate_sample_rate(cls, sample_rate:int) :
        if sample_rate < 0: 
            raise ValueError(f"sample_rate must be a positive integer")
        return sample_rate

    @validator("label_type")
    def validate_label_type(cls, label_type:str) :
        if label_type not in ['single_label', 'multi_label']:
            raise ValueError(f"Label type is not allowed: {label_type}")
        return label_type


class Model(BaseModel):
    criterion: dict
    model: dict

    class Config:
        title = "Model"
        allow_mutation = False
        validate_assignment = True
        anystr_lower = True
        validate_all = True
        use_enum_values = True

    @validator("criterion")
    def validate_criterion(cls, criterion:dict):
        if criterion['_target_'] not in criterion_dict.keys():
            raise ValueError(f"'This criterion is not allowed: {criterion['_target_']}")
        return criterion


    @validator("model")
    def validate_model(cls, model:dict):
        if model['_target_'] not in models_dict.keys():
            raise ValueError(f"'This model is not allowed: {model['_target_']}")
        if model['_target_'] == 'models.ChristophCNN':
            print('Attention: Make sure to use the preprocessors=_preproccesors_sliding_window argument when running Chritoph CNN!')
        return model


class Augmentations(BaseModel):
    frequency_masking: dict
    time_masking: dict
    random_noise: Optional[dict]

    class Config:
        title = "Augmentations"
        validate_assignment = True
        anystr_lower = True
        validate_all = True
        use_enum_values = True


    @validator("frequency_masking")
    def validate_frequency_masking(cls, frequency_masking:dict):
        return frequency_masking        
    
    @validator("time_masking")
    def validate_time_masking(cls, time_masking:dict):
        return time_masking     
        
    @validator("random_noise")
    def validate_random_noise(cls, random_noise:dict):
        return random_noise     


class Preprocessors(BaseModel):
    sliding_window_norm: Optional[dict]
    spectrogram: Optional[dict]
    resize: Optional[dict]

    class Config:
        title = "Preprocessors"
        validate_assignment = True
        anystr_lower = True
        validate_all = True
        use_enum_values = True


    @validator("sliding_window_norm")
    def validate_sliding_window_norm(cls, sliding_window_norm:dict):
        return sliding_window_norm        
    
    @validator("spectrogram")
    def validate_spectrogram(cls, spectrogram:dict):
        return spectrogram     
        
    @validator("resize")
    def validate_resize(cls, resize:dict):
        return resize   
    
class Config(BaseModel):
    data: Dataset
    model: Model
    augmentations: Augmentations
    preprocessors: Preprocessors
    class Config:
        title = "Config"
        fields = {'augmentations': '_augmentations',
                    'preprocessors': '_preprocessors'} # pydantic ignores private varibles - need to add alias



