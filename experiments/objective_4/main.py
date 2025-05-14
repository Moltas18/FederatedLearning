'''
This file can be used to generate parameters. Analysis is done in a seperate file.
'''

import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    
    # Add the root directory to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    # Import local modules
    from src.utils import parse_run, set_global_seed, get_filenames, read_from_file, dict_list_to_dict_tensor, parse_attack
    from src.plots import plot_reconstruction

    # Initialize seed
    seed = 42
    set_global_seed(seed=seed)

    #################################################################################################################
    #                           PARSE WIEGHTS, HYPER PARAMETERS AND DATA CONFIGURATIONS                             #
    #################################################################################################################
    
    run_path = r'C:\Users\Admin\Documents\github\FederatedLearning\results\2025-03-28\15-01-49\\'
    
    reconstructions_path = run_path + 'reconstruction/'
    df = parse_attack(reconstructions_path = reconstructions_path)
    print(df.describe)

    # # Plot the first row in the datafram. 
    # predicted_image_tensor = df.iloc[0]['predicted_images']
    # true_images_tensor = df.iloc[0]['true_images']
    # plot_reconstruction(
    # ground_truth_images=true_images_tensor,
    # reconstructed_images=predicted_image_tensor)
