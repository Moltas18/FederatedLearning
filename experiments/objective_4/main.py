'''
This file can be used to generate parameters. Analysis is done in a seperate file.
'''

import sys
import os
import torch

if __name__ == '__main__':
    
    # Add the root directory to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    # Import local modules
    from src.utils import parse_run, set_global_seed
    from src.plots import plot_reconstruction
    from src.metrics.metrics import SSIM, LPIPS, PSNR, MSE

    # Initialize seed
    seed = 42
    set_global_seed(seed=seed)

    #################################################################################################################
    #                           PARSE WIEGHTS, HYPER PARAMETERS AND DATA CONFIGURATIONS                             #
    #################################################################################################################
    run_path = r'C:\Users\Admin\Documents\github\FederatedLearning\results\2025-03-24\11-32-03\\'
    df = parse_run(run_path = run_path)

    # Pick a run of a client, not actually needed
    run_idx = 0
    run_series = df.iloc[run_idx]
    print(run_series)    
    
    # Load the saved tensors
    ground_truth_images = torch.load(run_path + 'ground_truth_images.pt')
    reconstructed_images = torch.load(run_path + 'reconstructed_images.pt')
    
    # Calculate SSIM score between ground truth and predicted images
    ssim_value = SSIM(reconstructed_images, ground_truth_images)
    print(f"SSIM between reconstructed and ground truth images: {ssim_value}")
    
    # Calculate LPIPS score between ground truth and predicted images
    lpips_value = LPIPS(reconstructed_images, ground_truth_images)
    print(f"LPIPS score between predicted and ground truth images: {lpips_value}")

    # Calculate PSNR score between ground truth and predicted images
    psnr_value = PSNR(reconstructed_images, ground_truth_images)
    print(f"PSNR score between predicted and ground truth images: {psnr_value}")

    # Calculate MSE score between ground truth and predicted images
    mse_value = MSE(reconstructed_images, ground_truth_images)
    print(f"MSE score between predicted and ground truth images: {mse_value}")
     
    plot_reconstruction(ground_truth_images=ground_truth_images, reconstructed_images=reconstructed_images)