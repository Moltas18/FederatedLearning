'''
This file can be used to generate parameters. Analysis is done in a seperate file.
'''

import sys
import os
import torch
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    
    # Add the root directory to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    # Import local modules
    from src.utils import parse_run, set_global_seed, get_filenames, read_from_file, dict_list_to_dict_tensor
    from src.plots import plot_reconstruction
    from src.metrics.metrics import SSIM, LPIPS, PSNR, MSE

    # Initialize seed
    seed = 42
    set_global_seed(seed=seed)

    #################################################################################################################
    #                           PARSE WIEGHTS, HYPER PARAMETERS AND DATA CONFIGURATIONS                             #
    #################################################################################################################
    run_path = r'C:\Users\Admin\Documents\GitHub\FederatedLearning\results\2025-03-27\08-35-42\\'
    # df = parse_run(run_path = run_path)

    # # Pick a run of a client, not actually needed
    # run_idx = 0
    # run_series = df.iloc[run_idx]  
    
    reconstructions_path = run_path + 'reconstruction/'
    reconstructions_files = get_filenames(reconstructions_path)


    print(reconstructions_files)

    # Loop through all of the parameter files
    for reconstructions_file in tqdm(reconstructions_files, desc="Processing parameter files"):
        reconstruction = read_from_file(reconstructions_path + reconstructions_file) # Not sure this works!
        temp_dict = reconstruction[0]
        reconstructed_images = temp_dict['reconstruction']['predicted_images']
        ground_truth_images = temp_dict['reconstruction']['true_images']
        
        true_images_tensor= torch.tensor(reconstructed_images)
        predicted_images_tensor = torch.tensor(ground_truth_images)

        # Plot the reconstruction
        plot_reconstruction(
        ground_truth_images=true_images_tensor,
        reconstructed_images=predicted_images_tensor
    )
        

        # Calculate SSIM score between ground truth and predicted images

        # plot_reconstruction(ground_truth_images=ground_truth_images, reconstructed_images=reconstructed_images)