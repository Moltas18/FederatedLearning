import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import json
width = 3.3  # Adjust to match \columnwidth
height = width * 0.75  # Aspect ratio (4:3)
FIGSIZE = (width, height)

mpl.rcParams.update({
    "text.usetex": False,  # Use LaTeX rendering for text
    "font.family": "serif",
    "font.size": 15,  # Adjust based on LaTeX document class
    "axes.labelsize": 17,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 15,
})

def plot_reconstruction(ground_truth_images: torch.Tensor, reconstructed_images: torch.Tensor, figsize=FIGSIZE) -> None:
    '''Function to plot ground truth and reconstructed images.
    Tensors should be of shape (batch_size, C, H, W).
    '''

    assert ground_truth_images.shape == reconstructed_images.shape, "The input tensors must have the same shape"

    batch_size = ground_truth_images.shape[0]

    # Create subplots
    _, axes = plt.subplots(2, batch_size, figsize=(batch_size * figsize[0], figsize[1]))

    # Ensure `axes` is always a 2D array
    if batch_size == 1:
        axes = axes[:, None]  # Convert 1D array to 2D (shape: (2, 1))

    # Add titles for each row
    axes[0, 0].set_title("Reconstructed Images", fontweight='bold')
    axes[1, 0].set_title("Ground Truth Images", fontweight='bold')

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
        plt.plot(rounds, val_accuracies, linestyle="-", label=f"Batch {batch_size}")

    plt.xlim(left=0)  # Start x-axis from 1
    # Labels and title
    plt.xlabel("Round")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy per Round for Different Epoch Sizes")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('plot.pdf', format='pdf')
    plt.tight_layout()
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
