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


torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # If using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    return [torch.tensor(v).cpu().detach() for v in parameters_record.values()]

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

        # Read config
        config_path = run_path + 'run_config.jsonl'
        config = read_from_file(config_path)[0]

        # Get the file names of the parameters saving (from each client)
        parameters_path = run_path + 'parameters/'
        parameters_files = get_filenames(parameters_path)

        # Predefine dict for dataframe
        data = {
            'Server Round' : [],
            'Client ID': [],
            'Batch Size': [],
            'Num Batches': [],
            'Partition ID': [],
            'Initial Parameters': [],
            'Updated Parameters': [],
        }
        
        # Loop through all of the parameter files
        for parameters_file in parameters_files:
            run_parameters = read_from_file(parameters_path + parameters_file) # Not sure this works!
            for training in run_parameters:
                data['Server Round'].append(training['run_info']['server_round'])
                data['Client ID'].append(training['run_info']['client_id'])
                data['Batch Size'].append(training['run_info']['batch_size'])
                data['Num Batches'].append(training['run_info']['num_batches'])
                data['Partition ID'].append(training['run_info']['node_config']['partition-id'])
                data['Initial Parameters'].append(training['parameters']['parameters_before_training'])
                data['Updated Parameters'].append(training['parameters']['parameters_after_training'])

        # Create dataframe
        df = pd.DataFrame(data)
        df['Epochs'] = config['epochs']
        df['Net'] = config['net']
        df['Num Clients'] = config['num_clients']
        df['Total Rounds'] = config['num_rounds']
        df['Optimizer'] = config['optim_method']
        df['Learning Rate'] = config['learning_rate']
        
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

def denormalize(img):
        device = img.device
        mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1)
        return img * std + mean  # Reverse normalization
    
def plot_reconstruction(ground_truth_images: torch.Tensor, reconstructed_images: torch.Tensor) -> None:
    '''Function to plot ground truth and reconstructed images.
    Tensors should be of shape (batch_size, C, H, W).
    '''

    assert ground_truth_images.shape == reconstructed_images.shape, "The input tensors must have the same shape"

    batch_size = ground_truth_images.shape[0]

    # Detach, clone, and denormalize images
    ground_truth_images = denormalize(ground_truth_images.clone().detach())
    reconstructed_images = denormalize(reconstructed_images.clone().detach())

    # Create subplots
    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 3, 6))

    # Ensure `axes` is always a 2D array
    if batch_size == 1:
        axes = axes[:, None]  # Convert 1D array to 2D (shape: (2, 1))

    # Add titles for each row
    axes[0, 0].set_title("Reconstructed Images", fontsize=14, fontweight='bold')
    axes[1, 0].set_title("Ground Truth Images", fontsize=14, fontweight='bold')

    for i in range(batch_size):
        
        axes[0, i].imshow(reconstructed_images[i].permute(1, 2, 0).cpu())
        axes[0, i].axis('off')

        axes[1, i].imshow(ground_truth_images[i].permute(1, 2, 0).cpu())
        axes[1, i].axis('off')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def plot_loss(history_loss:list):
    '''
    This function plots the loss of a model during training
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(history_loss, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss')
    plt.legend()
    plt.show()