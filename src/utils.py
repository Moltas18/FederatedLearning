"""
Utility functions for the Federated Learning project.

This file contains helper functions for tasks such as timing, parameter handling,
file I/O, data normalization, and more.
"""

import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from collections import OrderedDict
from flwr.common import Metrics
import yaml
from typing import List, Tuple, Union, Sequence
from pathlib import Path
from time import time
import json
from tqdm import tqdm
import random
from src.metrics.metrics import SSIM, LPIPS, PSNR


def timer(func):
    """
    A decorator function that measures the execution time of a function.

    Args:
        func (function): The function to be timed.

    Returns:
        function: A wrapped version of the input function that prints its execution time.

    Example:
        @timer
        def my_function():
            pass
    """
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        execution_time = t2 - t1
        print(f'Function {func.__name__!r} executed in {(execution_time):.4f}s')
        return result

    return wrap_func


def set_global_seed(seed: int) -> None:
    """
    Set the global random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch (all GPUs)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning


def eval_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Compute the weighted average of metrics based on the number of examples.

    Args:
        metrics (List[Tuple[int, Metrics]]): A list of tuples containing the number of examples
                                             and the corresponding metrics.

    Returns:
        Metrics: A dictionary containing the weighted average of the metrics.
    """
    accuracies = [num_examples * m["validation_accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"validation_accuracy": sum(accuracies) / sum(examples)}


def fit_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Compute the weighted average of metrics for model fitting.

    Args:
        metrics (List[Tuple[int, Metrics]]): A list of tuples containing the number of examples
                                             and the corresponding metrics.

    Returns:
        Metrics: A dictionary containing the weighted average of the metrics.
    """
    total_examples = 0
    weighted_metrics = {}

    for num_examples, client_metrics in metrics:
        total_examples += num_examples
        for metric_name, metric_value in client_metrics.items():
            if metric_name not in weighted_metrics:
                weighted_metrics[metric_name] = 0.0
            weighted_metrics[metric_name] += num_examples * metric_value

    for metric_name in weighted_metrics:
        weighted_metrics[metric_name] /= total_examples

    return weighted_metrics


def set_parameters(net: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Set the parameters of a PyTorch model.

    Args:
        net (torch.nn.Module): The model to update.
        parameters (List[np.ndarray]): A list of NumPy arrays representing the parameters.
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=False)


def get_parameters(net: torch.nn.Module) -> List[np.ndarray]:
    """
    Get the parameters of a PyTorch model as a list of NumPy arrays.

    Args:
        net (torch.nn.Module): The model to extract parameters from.

    Returns:
        List[np.ndarray]: A list of NumPy arrays representing the parameters.
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def serialize_parameters(net: torch.nn.Module, parameters: List[np.ndarray]) -> dict:
    """
    Serialize model parameters into a JSON-compatible dictionary.

    Args:
        net (torch.nn.Module): The model.
        parameters (List[np.ndarray]): A list of NumPy arrays representing the parameters.

    Returns:
        dict: A dictionary with serialized parameters.
    """
    return {k: v.tolist() for k, v in zip(net.state_dict().keys(), parameters)}


def deserialize_parameters(parameters_record: dict) -> List[torch.Tensor]:
    """
    Deserialize parameters from a JSON-compatible dictionary into PyTorch tensors.

    Args:
        parameters_record (dict): A dictionary containing serialized parameters.

    Returns:
        List[torch.Tensor]: A list of PyTorch tensors representing the parameters.
    """
    return [torch.tensor(v).detach() for v in parameters_record.values()]


def dict_to_list_of_tensors(parameters_dict: dict, device: str) -> List[torch.Tensor]:
    """
    Convert a dictionary of parameters into a list of PyTorch tensors.

    Args:
        parameters_dict (dict): A dictionary of parameters.
        device (str): The device to move the tensors to.

    Returns:
        List[torch.Tensor]: A list of PyTorch tensors.
    """
    return [param.clone().to(device) for param in parameters_dict.values()]


def dict_list_to_dict_tensor(parameters_record: dict) -> dict:
    """
    Convert a dictionary of lists into a dictionary of PyTorch tensors.

    Args:
        parameters_record (dict): A dictionary containing lists.

    Returns:
        dict: A dictionary containing PyTorch tensors.
    """
    for key, value in parameters_record.items():
        if isinstance(value, list):
            parameters_record[key] = torch.tensor(value)
    return parameters_record


def write_to_file(data: dict, path: Union[str, Path], filename: str) -> None:
    """
    Write data to a JSONL file.

    Args:
        data (dict): The data to write.
        path (Union[str, Path]): The directory path.
        filename (str): The name of the file.
    """
    with open(f"{path}/{filename}.jsonl", "a", encoding="utf-8") as fp:
        json.dump(data, fp)
        fp.write("\n")


def read_from_file(path: Union[str, Path]) -> list:
    """
    Read data from a JSONL file.

    Args:
        path (Union[str, Path]): The path to the JSONL file.

    Returns:
        list: A list of dictionaries containing the data.
    """
    with open(path, 'r') as file:
        return [json.loads(line) for line in file]


def get_filenames(directory: Union[str, Path]) -> List[str]:
    """
    Get all filenames in a directory.

    Args:
        directory (Union[str, Path]): The directory path.

    Returns:
        List[str]: A list of filenames.
    """
    directory_path = Path(directory)
    return [file.name for file in directory_path.iterdir() if file.is_file()]


def parse_run(run_path: Union[str, Path]) -> pd.DataFrame:
    """
    Parse a run directory and return a DataFrame with the run data.

    Args:
        run_path (Union[str, Path]): The path to the run directory.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed run data.
    """
    # Read run_config
    run_config_path = run_path + 'run_config.jsonl'
    run_config = read_from_file(run_config_path)[0]

    # Read data config
    data_config_path = run_path + 'data_config.jsonl'
    data_config = read_from_file(data_config_path)[0]

    # Get the file names of the parameters saving (from each client)
    parameters_path = run_path + 'parameters/'
    parameters_files = get_filenames(parameters_path)

    # Predefine dict for dataframe
    data = {
        'Server Round': [],
        'Client ID': [],
        'Actual Batch Size': [],
        'Num Batches': [],
        'Partition ID': [],
        'Initial Parameters': [],
        'Updated Parameters': [],
    }

    # Loop through all of the parameter files
    for parameters_file in tqdm(parameters_files, desc="Processing parameter files"):
        run_parameters = read_from_file(parameters_path + parameters_file)
        for training in run_parameters:
            data['Server Round'].append(training['run_info']['server_round'])
            data['Client ID'].append(training['run_info']['client_id'])
            data['Actual Batch Size'].append(training['run_info']['batch_size'])
            data['Num Batches'].append(training['run_info']['num_batches'])
            data['Partition ID'].append(training['run_info']['node_config']['partition-id'])
            data['Initial Parameters'].append(training['parameters']['parameters_before_training'])
            data['Updated Parameters'].append(training['parameters']['parameters_after_training'])

    # Create dataframe
    df = pd.DataFrame(data)

    # Add info from the run_config
    df['Epochs'] = run_config['epochs']
    df['Net'] = run_config['net']
    df['Num Clients'] = run_config['num_clients']
    df['Total Rounds'] = run_config['num_rounds']
    df['Optimizer'] = run_config['optim_method']
    df['Learning Rate'] = run_config['learning_rate']

    # Add info from the data_config
    df['Dataset'] = data_config['dataset']
    df['Data Batch Size'] = data_config['batch_size']
    df['Val/Test Batch Size'] = data_config['val_test_batch_size']
    df['Partitioner'] = data_config['partitioner']
    df['Partition Size'] = data_config['partition_size']
    df['Seed'] = data_config['seed']
    df['Validation Size'] = data_config['val_size']
    df['Include Test Set'] = data_config['include_test_set']
    df['Normalization Means'] = [data_config['normalization_means']] * len(df)
    df['Normalization Stds'] = [data_config['normalization_stds']] * len(df)
    return df


def parse_attack(reconstructions_path: Union[str, Path], perform_mertrics_computation: bool = True) -> pd.DataFrame:
    """
    Parse an attack directory and return a DataFrame with the attack data.

    Args:
        reconstructions_path (Union[str, Path]): The path to the attack directory.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed attack data.
    """
    
    reconstructions_files = get_filenames(reconstructions_path)

    df_dict = {
        'server_round' : [],
        'client_id' : [],
        # 'epochs' :  [],
        'batch_size' : [],
        'num_batches' : [],
        'predicted_images' : [],
        'true_images' : []
    }

    # Loop through all of the parameter files
    for reconstructions_file in tqdm(reconstructions_files, desc="Processing Reconstruction files"):
        client_reconstruction = read_from_file(reconstructions_path + reconstructions_file) # Not sure this works!
        for round in client_reconstruction:            
            df_dict['batch_size'].append(round['run_info']['batch_size'])
            df_dict['server_round'].append(round['run_info']['server_round'])
            # df_dict['epochs'].append(1.0)
            df_dict['client_id'].append(round['run_info']['client_id'])
            df_dict['num_batches'].append(round['run_info']['num_batches'])
            df_dict['predicted_images'].append(torch.Tensor(round['reconstruction']['predicted_images']))
            df_dict['true_images'].append(torch.Tensor(round['reconstruction']['true_images']))

    # Create the dataframe        
    df = pd.DataFrame(df_dict)

    if perform_mertrics_computation:
        # Compute the metrics
        df['psnr'] = df.apply(lambda x: PSNR(x['predicted_images'], x['true_images']), axis=1)
        df['ssim'] = df.apply(lambda x: SSIM(x['predicted_images'], x['true_images']), axis=1)
        df['lpips'] = df.apply(lambda x: LPIPS(x['predicted_images'], x['true_images']), axis=1)

    return df 

def denormalize(img: torch.Tensor, means: Sequence[float], stds: Sequence[float]) -> torch.Tensor:
    """
    Denormalize an image tensor with respect to the given means and standard deviations.

    Args:
        img (torch.Tensor): The image tensor to denormalize.
        means (Sequence[float]): The mean values for each channel.
        stds (Sequence[float]): The standard deviation values for each channel.

    Returns:
        torch.Tensor: The denormalized image tensor.
    """
    device = img.device
    mean = torch.tensor(means, device=device).view(3, 1, 1)
    std = torch.tensor(stds, device=device).view(3, 1, 1)
    return img * std + mean  # Reverse normalization
