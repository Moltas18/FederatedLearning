import matplotlib.pyplot as plt
import matplotlib as mpl
import torch

width = 3.3  # Adjust to match \columnwidth
height = width * 0.75  # Aspect ratio (4:3)
FIGSIZE = (width, height)

mpl.rcParams.update({
    "text.usetex": False,  # Use LaTeX rendering for text
    "font.family": "serif",
    "font.size": 10,  # Adjust based on LaTeX document class
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
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