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
    from src.utils import parse_run, dict_list_to_dict_tensor, set_parameters, denormalize, set_global_seed
    from src.attack.utils import ParameterDifference, GradientApproximation
    from src.plots import plot_reconstruction
    from data.data import Data
    from src.models.CNNcifar import CNNcifar
    from src.attack.SME import SME

    # Initialize seed
    seed = 42
    set_global_seed(seed=seed)

    #################################################################################################################
    #                           PARSE WIEGHTS, HYPER PARAMETERS AND DATA CONFIGURATIONS                             #
    #################################################################################################################
    run_path = r'C:\Users\Admin\Documents\github\FederatedLearning\results\2025-03-18\15-29-39\\'
    df = parse_run(run_path = run_path)

    # Pick a run of a client
    run_idx = 0

    ### Basic example usage ###
    run_series = df.iloc[run_idx]

    print(run_series)
    # Load the parameters from the first client, the round.
    initial_params, updated_params = run_series['Initial Parameters'], run_series['Updated Parameters']

    # Load hyperparameters from the first round
    epochs = run_series['Epochs']
    lr = run_series['Learning Rate']
    partition_id = run_series['Partition ID']

    data = Data(
                batch_size=run_series['Data Batch Size'],
                partitioner=run_series['Partitioner'],
                partition_size=run_series['Partition Size'],
                dataset=run_series['Dataset'],
                seed=run_series['Seed'],
                include_test_set=run_series['Include Test Set'],
                val_size=run_series['Validation Size'],
                val_test_batch_size=run_series['Val/Test Batch Size'],
                normalization_means=run_series['Normalization Means'],
                normalization_stds=run_series['Normalization Stds']
                )
    
    # Load the image of the selected partition
    trainloader, _, _ = data.load_datasets(partition_id=partition_id)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    initial_params = dict_list_to_dict_tensor(initial_params)
    updated_params = dict_list_to_dict_tensor(updated_params)
    
    #################################################################################################################
    #                                           RUN THE DL ATTACK                                                   #
    #################################################################################################################

    # SME Hyperparameters
    alpha = 0.5
    lamb = 0.01
    eta = 1
    beta = 0.001
    iters = 100
    lr_decay = False

    # Victim Model
    net = CNNcifar

    # Initialize an instance of the attack class
    sme = SME(
        trainloader=trainloader,
        net=net,
        w0=initial_params, 
        wT=updated_params,
        device=device,
        alpha=alpha,
        mean_std=run_series['Normalization Means'],
        lamb=lamb
    )

    # Perform the reconstruction
    predicted_images, true_images, true_labels =  sme.reconstruction(eta,
                                                                    beta,
                                                                    iters,
                                                                    lr_decay)
    
    # Detach, clone, and denormalize images, this should probably be done outside!
    ground_truth_images = denormalize(true_images.clone().detach(), run_series['Normalization Means'], run_series['Normalization Stds'])
    reconstructed_images = denormalize(predicted_images.clone().detach(), run_series['Normalization Means'], run_series['Normalization Stds'])
    
    plot_reconstruction(ground_truth_images=ground_truth_images, reconstructed_images=reconstructed_images)