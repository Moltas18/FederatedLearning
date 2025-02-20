import matplotlib.pyplot as plt
import torch
from flwr.common import Metrics
import yaml
from typing import List, Tuple
from time import time 
import json

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # If using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        
def load_config(config_path="config/template.yaml"):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

def timer(func): 
    # This function shows the execution time of  
    # the function object passed 
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

def write_to_file(data, path, filename):
    with open(f"{path}/{filename}.jsonl", "a", encoding="utf-8") as fp:
        json.dump(data, fp)
        fp.write("\n")  # Ensures each JSON object is on a new line

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