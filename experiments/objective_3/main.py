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
    from src.utils import parse_run, deserialize_parameters, read_from_file, get_filenames, plot_reconstruction, dict_list_to_dict_tensor
    from src.attack.utils import ParameterDifference, GradientApproximation
    from data.data import Data
    from src.models.models import LeNet5


    run_path = 'results/2025-03-05/07-52-19/'
    df = parse_run(run_path = run_path)
    run_idx = 0

    ### Basic example usage ###
    run_series = df.iloc[run_idx]

    # Load the parameters from the first client, the round.
    initial_params, updated_params = run_series['Initial Parameters'], run_series['Updated Parameters']

    # Load hyperparameters from the first round
    epochs = run_series['Epochs']
    lr = run_series['Learning Rate']
    partition_id = run_series['Partition ID']

    # Deserialize the parameters into tensors
    W0, W1 = deserialize_parameters(parameters_record=initial_params), deserialize_parameters(parameters_record=updated_params)

    ### Data configurations
    batch_size = 'full'
    val_test_batch_size = 256
    val_size = 0.5 # 50% of the data is used for validation (we use one image for training and one for validation)
    partitioner = 25000 # 2 images per partition; one for validation and one for training
    seed = 42

    data = Data(batch_size=batch_size,
                partitioner=partitioner,
                seed=seed,
                val_size=val_size,
                val_test_batch_size=val_test_batch_size)
    
    # Load the image of the selected partition
    trainloader, _, _ = data.load_datasets(partition_id=partition_id)
    gt_image = next(iter(trainloader))['img']

    # Load another partition
    trainloader, _, _ = data.load_datasets(partition_id=partition_id+1)
    x = next(iter(trainloader))['img']
    y = next(iter(trainloader))['label']

    # plot_reconstruction(gt_image, recon_image)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    initial_params = dict_list_to_dict_tensor(initial_params)
    updated_params = dict_list_to_dict_tensor(updated_params)
    
    print(x.shape)
