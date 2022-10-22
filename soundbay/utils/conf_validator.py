from pydantic import BaseModel, validator



class Dataset(BaseModel):
    # label_names: ListConfig
    batch_size: int
    num_workers: int
    sample_rate: int
    data_sample_rate: int
    max_freq: int
    min_freq_filtering: int 
    n_fft: int
    hop_length: int
    train_dataset: dict
    val_dataset: dict
    class Config:
        title = "Dataset"
        # max_anystr_length = 10
        allow_mutation = False
        validate_assignment = True
        anystr_lower = False
        validate_all = True
        use_enum_values = True

    @validator("num_workers")
    def validate_num_workers(cls, num_workers:int):
        if num_workers > 4:
            raise ValueError(f"Number of works is larger than 4 {num_workers}")
        return num_workers


    @validator("train_dataset")
    def validate_train_dataset(cls, train_dataset:dict) :
        possible_train_datasets = ['soundbay.data.ClassifierDataset', 'soundbay.data.BaseDataset']
        if train_dataset['_target_'] not in possible_train_datasets: 
            raise ValueError(f"Train_dataset is not allowed from type {train_dataset['_target_']}")
        return train_dataset

class Model(BaseModel):
    criterion: dict
    model: dict

    class Config:
        title = "Model"
        # max_anystr_length = 10
        allow_mutation = False
        validate_assignment = True
        anystr_lower = True
        validate_all = True
        use_enum_values = True

    @validator("criterion")
    def validate_criterion(cls, criterion:int):
        # p = Path(path)
        possible_values = ['torch.nn.MSELoss', 'torch.nn.CrossEntropyLoss']
        if criterion['_target_'] not in possible_values:
            raise ValueError(f"'This criterion is not allowed: {criterion['_target_']}")
        return criterion


    @validator("model")
    def validate_model(cls, model:dict):
        possible_values = ['models.ResNet1Channel', 'models.GoogleResNet50withPCEN']
        if model['_target_'] not in possible_values:
            raise ValueError(f"'This model is not allowed: {model['_target_']}")
        return model

class Config(BaseModel):
    data: Dataset
    model: Model