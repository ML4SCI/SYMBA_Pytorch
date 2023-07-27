from datasets import Data

registered_datasets = [Data.Dataset]

def get_dataset_from_config(config):
    dataset_name = config.dataset_name
    for registered_dataset in registered_datasets:
        if registered_dataset.is_valid_dataset_name(dataset_name):
            return registered_dataset.get_dataset_from_config(config)
    raise ValueError(f"Dataset {dataset_name} is not supported.")
