import matplotlib.pyplot as plt
import torch
import pandas as pd 
import numpy as np
from collections import OrderedDict
from flwr.common import Metrics
import yaml
from typing import List, Tuple, Union
from pathlib import Path
from time import time 
import json
from tqdm import tqdm
from typing import Union, List, Tuple, Sequence
import random



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

def set_global_seed(seed):
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch (all GPUs)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning
    # torch.use_deterministic_algorithms(True)

def eval_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    '''
    This function is called with the return from the evaluate or fit within FlowerClient 
    '''
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["validation_accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"validation_accuracy": sum(accuracies) / sum(examples)}

def fit_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    '''
    This function is called with the return from the evaluate or fit within FlowerClient 
    '''
    # Initialize accumulators for each metric
    total_examples = 0
    weighted_metrics = {}

    for num_examples, client_metrics in metrics:
        total_examples += num_examples
        for metric_name, metric_value in client_metrics.items():
            if metric_name not in weighted_metrics:
                weighted_metrics[metric_name] = 0.0
            weighted_metrics[metric_name] += num_examples * metric_value

    # Compute the weighted average for each metric
    for metric_name in weighted_metrics:
        weighted_metrics[metric_name] /= total_examples

    return weighted_metrics

def set_parameters(net, parameters: List[np.ndarray]):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=False)

def get_parameters(net) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    
def serialize_parameters(net: torch.nn.Module, parameters: List[np.ndarray]) -> dict:
    # Convert NumPy arrays to JSON-compatible lists
    return {
        k: v.tolist()  
        for k, v in zip(net.state_dict().keys(), parameters)
    }

def deserialize_parameters(parameters_record: dict) -> List[torch.Tensor]:
    # Convert JSON-compatible lists directly to PyTorch tensors
    return [torch.tensor(v).detach() for v in parameters_record.values()]

def dict_to_list_of_tensors(parameters_dict: dict, device: str) -> List[torch.Tensor]:
    return  [param.clone().to(device) for param in parameters_dict.values()]

def dict_list_to_dict_tensor(parameters_record: dict) -> dict:
    # return {k: torch.tensor(v) for k, v in parameters_record.items()}
    for key, value in parameters_record.items():
        if isinstance(value, list):  # Convert lists to tensors
            parameters_record[key] = torch.tensor(value)
    return parameters_record

def write_to_file(data, path, filename):
    with open(f"{path}/{filename}.jsonl", "a", encoding="utf-8") as fp:
        json.dump(data, fp)
        fp.write("\n")  # Ensures each JSON object is on a new line

def read_from_file(path: Union[str, Path]) -> list:
    '''
    This function reads a JSONL file and returns a list of dictionaries
    
    args:
    path (str): The path to the JSONL file

    returns:
    data (dict): A list of dictionaries

    example:
    data = read_from_file('results/2025-02-27/08-51-40/parameters.jsonl')
            '''
    with open(path, 'r') as file:
        return [json.loads(line) for line in file]

def get_filenames(directory: Union[str, Path]) -> List[str]:
    directory_path = Path(directory)
    filenames = [file.name for file in directory_path.iterdir() if file.is_file()]
    return filenames

def parse_run(run_path: Union[str, Path]) -> pd.DataFrame:

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
            'Server Round' : [],
            'Client ID': [],
            'Actual Batch Size': [],
            'Num Batches': [],
            'Partition ID': [],
            'Initial Parameters': [],
            'Updated Parameters': [],
        }
        
        # Loop through all of the parameter files
        for parameters_file in tqdm(parameters_files, desc="Processing parameter files"):
            run_parameters = read_from_file(parameters_path + parameters_file) # Not sure this works!
            for training in run_parameters:
                data['Server Round'].append(training['run_info']['server_round'])
                data['Client ID'].append(training['run_info']['client_id'])
                data['Actual Batch Size'].append(training['run_info']['batch_size']) # This is the actual batch size for this specific round!
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

        # Add info from the data_config. This info will be used to create a new instance of the Data class.
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

def plot_image_samples(images: torch.Tensor) -> None: 

    # Reshape and convert images to a NumPy array
    # matplotlib requires images with the shape (height, width, 3)
    images = images.permute(0, 2, 3, 1).numpy()

    # Denormalize
    images = images / 2 + 0.5

    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(4, 8, figsize=(12, 6))

    # Loop over the images and plot them
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        ax.axis("off")

    # Show the plot
    fig.tight_layout()
    plt.show() 

def plot_multiple_validation_curves(metrics_files, config_files):
    """
    Plots validation accuracy curves from multiple test runs with different batch sizes.
    
    Args:
        metrics_files (list of str): Paths to JSONL metric files containing per-round validation results.
        config_files (list of str): Paths to JSONL config files containing run settings.
    """
    plt.figure(figsize=(10, 6))

    for metrics_file, config_file in zip(metrics_files, config_files):
        # Read config file to extract batch size
        with open(config_file, "r") as f:
            config = json.load(f)
        batch_size = config.get("batch_size", "Unknown")

        # Read validation accuracy from metrics file
        rounds = []
        val_accuracies = []
        with open(metrics_file, "r") as f:
            for line in f:
                data = json.loads(line)
                if "validation_accuracy" in data:
                    rounds.append(len(rounds) + 1)  # Assuming one entry per round
                    val_accuracies.append(data["validation_accuracy"])

        # Plot validation accuracy for this batch size
        plt.plot(rounds, val_accuracies, marker="o", linestyle="-", label=f"Batch {batch_size}")

    # Labels and title
    plt.xlabel("Round")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy per Round for Different Batch Sizes")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_run_results(metrics_path: str, config_path: str) -> None:
    '''
    This function plots the training progress of a simulation
    '''
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    rounds = []

    # Read the metrics file
    with open(metrics_path, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if "train_loss" in data:
                train_loss.append(data["train_loss"])
                train_acc.append(data["train_accuracy"])
                rounds.append(i + 1)  # Round index starts at 1
            elif "validation_loss" in data:
                val_loss.append(data["validation_loss"])
                val_acc.append(data["validation_accuracy"])

    # Read the metadata file
    with open(config_path, "r") as f:
        config = json.load(f)
    legend_info = (
    f"Model: {config['net']}\n"
    f"Clients: {config['num_clients']}\n"
    f"Rounds: {config['num_rounds']}\n"
    f"Epochs: {config['epochs']}\n"
    f"Batch: {config['batch_size']}"
   )
    
    # Plot accuracy and loss
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Accuracy Plot
    ax[0].plot(rounds, train_acc, label="Train Accuracy", marker="o", linestyle="-")
    ax[0].plot(rounds, val_acc, label="Validation Accuracy", marker="s", linestyle="--")
    ax[0].set_title("Training & Validation Accuracy per Round")
    ax[0].set_xlabel("Rounds")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    # Loss Plot
    ax[1].plot(rounds, train_loss, label="Train Loss", marker="o", linestyle="-")
    ax[1].plot(rounds, val_loss, label="Validation Loss", marker="s", linestyle="--")
    ax[1].set_title("Training & Validation Loss per Round")
    ax[1].set_xlabel("Rounds")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    # Add metadata as a text box
    plt.gcf().text(0.75, 0.65, legend_info, fontsize=10, bbox=dict(facecolor='lightgrey', alpha=0.5))

    plt.tight_layout()
    plt.show()

def denormalize(img: torch.Tensor,
                means: Sequence[float],
                stds: Sequence[float]):
    '''
    Denormalizes the image (tensor) with respect to the means and stds.
    '''
    device = img.device
    mean = torch.tensor(means, device=device).view(3, 1, 1)
    std = torch.tensor(stds, device=device).view(3, 1, 1)
    return img * std + mean  # Reverse normalization
