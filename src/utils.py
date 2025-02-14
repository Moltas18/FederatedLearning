import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr.common import Metrics
import yaml
from typing import List, Tuple
from time import time 


def apply_transforms(batch):
    # Instead of passing transforms to CIFAR10(..., transform=transform)
    # we will use this function to dataset.with_transform(apply_transforms)
    # The transforms object is exactly the same

    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def load_datasets(partition_id: int, batch_size: int=1, partitioner=None):

    fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, valloader, testloader

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
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
        return result 
    return wrap_func

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}