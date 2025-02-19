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
        metadata = json.load(f)

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
    meta_text = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
    plt.gcf().text(0.75, 0.5, meta_text, fontsize=10, bbox=dict(facecolor='lightgrey', alpha=0.5))

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
