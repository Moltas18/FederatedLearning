'''
This file can be used to generate parameters. Analysis is done in a seperate file.
'''

import sys
import os
import torch


# Function to run the SME attack
def run_sme_attack(config, run_series, run_path, data):
    """
    Run the SME attack on all clients and save the results.

    Args:
        config (dict): Dictionary containing hyperparameters and configurations.
        run_series (pd.Series): Series containing run configurations.
        run_path (str): Path to save the results.
        data (Data): Data object for loading datasets.
    """
    # Extract configurations
    alpha = config["alpha"]
    lamb = config["lamb"]
    eta = config["eta"]
    beta = config["beta"]
    iters = config["iters"]
    lr_decay = config["lr_decay"]
    net = config["victim_model"]
    device = config["device"]
    filtered_df = config["filtered_df"]

    for _, client in filtered_df.iterrows():
        # Extract parameters for the client
        initial_params, updated_params = client['Initial Parameters'], client['Updated Parameters']
        initial_params = dict_list_to_dict_tensor(initial_params)
        updated_params = dict_list_to_dict_tensor(updated_params)
        partition_id = client['Partition ID']

        # Load the data for the selected partition
        trainloader, _, _ = data.load_datasets(partition_id=partition_id)

        # Initialize the SME attack
        sme = SME(
            trainloader=trainloader,
            net=net,
            w0=initial_params,
            wT=updated_params,
            device=device,
            alpha=alpha,
            mean_std=run_series['Normalization Means'],
            lamb=lamb,
        )

        # Perform the reconstruction
        predicted_images, true_images, true_labels = sme.reconstruction(
            eta=eta,
            beta=beta,
            iters=iters,
            lr_decay=lr_decay
        )

        # Denormalize the images
        predicted_images = denormalize(predicted_images, run_series['Normalization Means'], run_series['Normalization Stds'])  
        true_images = denormalize(true_images, run_series['Normalization Means'], run_series['Normalization Stds'])
        
        # Align images with linnear sum assigment
        true_images, predicted_images = allign_images(ground_truth_images=true_images,
                                                      reconstructed_images=predicted_images)

        # Serialize the reconstructed and true images
        serialized_data = {
            'predicted_images': predicted_images.clone().detach().cpu().tolist(),
            'true_images': true_images.clone().detach().cpu().tolist(),
            'true_labels': true_labels.clone().detach().cpu().tolist()
        }

        # Run info from the client
        run_info = {
            'server_round': client['Server Round'],
            'client_id': client['Client ID'],  # Assuming 'Client ID' exists in the DataFrame
            'partition_id': partition_id,
            'batch_size': trainloader.batch_size,
            'num_batches': len(trainloader),
        }

        # Combine run info and serialized data
        combine_data = {
            "run_info": run_info,
            "reconstruction": serialized_data,
        }
        
        # A new directory needs to be created where parameters JSONL-files are stored for each client
        parameters_path = os.path.join(run_path, "reconstruction")
        os.makedirs(parameters_path, exist_ok=True)
        # We create one parameters-file per client inside the parameters directory
        write_to_file(data=combine_data, path=parameters_path, filename=str(client['Client ID']))

if __name__ == '__main__':
    
    # Add the root directory to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    # Import local modules
    from src.utils import parse_run, dict_list_to_dict_tensor, set_parameters, denormalize, set_global_seed, write_to_file
    from src.attack.utils import ParameterDifference, GradientApproximation, allign_images
    from src.plots import plot_reconstruction
    from data.data import Data
    from src.models.CNNcifar import CNNcifar
    from src.attack.SME import SME
    from src.metrics.metrics import SSIM, LPIPS, PSNR, MSE

    # Initialize seed
    seed = 42
    set_global_seed(seed=seed)

    #################################################################################################################
    #                           PARSE WIEGHTS, HYPER PARAMETERS AND DATA CONFIGURATIONS                             #
    #################################################################################################################
    run_path = r'C:\Users\Admin\Documents\GitHub\FederatedLearning\results\2025-03-27\08-35-42\\'
    df = parse_run(run_path = run_path)
    run_series = df.iloc[0]
    # Load hyperparameters from the first round for all test as they are the same for all clients
    epochs = run_series['Epochs']
    lr = run_series['Learning Rate']
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

    #################################################################################################################
    #                                           RUN THE DL ATTACK ON ALL CLIENTS                                    #
    #################################################################################################################
    # Define hyperparameters and configurations
    config = {
        "alpha": 0.5,
        "lamb": 0.01,
        "eta": 1,
        "beta": 0.001,
        "iters": 1000,
        "lr_decay": True,
        "victim_model": CNNcifar,  # Victim Model
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "filtered_df": df,  # Filtered DataFrame (can be modified if needed)
    }

    # Call the function
    run_sme_attack(config=config, run_series=run_series, run_path=run_path, data=data)

