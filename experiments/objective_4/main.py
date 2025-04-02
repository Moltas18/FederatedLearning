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
    from src.utils import parse_run, set_global_seed, get_filenames, read_from_file, dict_list_to_dict_tensor
    from src.plots import plot_reconstruction
    from src.metrics.metrics import SSIM, LPIPS, PSNR, MSE

    # Initialize seed
    seed = 42
    set_global_seed(seed=seed)

    #################################################################################################################
    #                           PARSE WIEGHTS, HYPER PARAMETERS AND DATA CONFIGURATIONS                             #
    #################################################################################################################
    
    run_path = r'C:\Users\Admin\Documents\GitHub\FederatedLearning\results\2025-04-01\15-10-47\\'
    
    reconstructions_path = run_path + 'reconstruction/'
    reconstructions_files = get_filenames(reconstructions_path)

    df_dict = {
        'server_round' : [],
        'client_id' : [],
        # 'epochs' :  [],
        'batch_size' : [],
        'num_batches' : [],
        'predicted_images' : [],
        'true_images' : []
    }

    # Loop through all of the parameter files
    for reconstructions_file in tqdm(reconstructions_files, desc="Processing Reconstruction files"):
        client_reconstruction = read_from_file(reconstructions_path + reconstructions_file) # Not sure this works!
        for round in client_reconstruction:            
            df_dict['batch_size'].append(round['run_info']['batch_size'])
            df_dict['server_round'].append(round['run_info']['server_round'])
            # df_dict['epochs'].append(1.0)
            df_dict['client_id'].append(round['run_info']['client_id'])
            df_dict['num_batches'].append(round['run_info']['num_batches'])
            df_dict['predicted_images'].append(torch.Tensor(round['reconstruction']['predicted_images']))
            df_dict['true_images'].append(torch.Tensor(round['reconstruction']['true_images']))

    # Create the dataframe        
    df = pd.DataFrame(df_dict)

    # Compute the metrics
    df['psnr'] = df.apply(lambda x: PSNR(x['predicted_images'], x['true_images']), axis=1)
    df['ssim'] = df.apply(lambda x: SSIM(x['predicted_images'], x['true_images']), axis=1)
    df['lpips'] = df.apply(lambda x: LPIPS(x['predicted_images'], x['true_images']), axis=1)

    print(df)

    # Plot the first row in the datafram. 
    predicted_image_tensor = df.iloc[0]['predicted_images']
    true_images_tensor = df.iloc[0]['true_images']
    plot_reconstruction(
    ground_truth_images=true_images_tensor,
    reconstructed_images=predicted_image_tensor)

        
    #     #     reconstructed_images = temp_dict['reconstruction']['predicted_images']
    #     #     ground_truth_images = temp_dict['reconstruction']['true_images']
            
    #     #     true_images_tensor= torch.tensor(reconstructed_images)
    #     #     predicted_images_tensor = torch.tensor(ground_truth_images)

    # # # Plot the reconstruction
    
            

    # #     #     # Calculate SSIM score between ground truth and predicted images

    # #     #     # plot_reconstruction(ground_truth_images=ground_truth_images, reconstructed_images=reconstructed_images)