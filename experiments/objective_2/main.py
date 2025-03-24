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
    from src.simulation import Simulation
    from src.models.CNNcifar import CNNcifar
    from src.utils import fit_weighted_average, eval_weighted_average, plot_run_results, read_from_file, parse_run, set_global_seed
    from data.data import Data
    from src.strategy import CustomFedAvg

    # Initialize seed
    seed = 42
    set_global_seed(seed=seed)
    
    ### Configurations

    # Federated learning configurations
    num_clients = 1
    num_rounds = 1
    # Model configurations
    epochs = 1
    net = CNNcifar()
    optimizer = torch.optim.SGD
    lr = 0.004
    criterion = torch.nn.CrossEntropyLoss()

    # Data configurations
    partitioner = num_clients # Number of partitions to make, should almost always be kept this way!
    batch_size = 'full'
    val_size = 0.5 # 50% of the data is used for validation
    training_partition_size = 1 # This is the size of the full datasets which the clients will TRAIN on.
    partition_size = training_partition_size*2 # Trains on half of the images and validates on the other half.
    
    # General configurations
    num_gpus = 1/num_clients
    num_cpus = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Save parameters
    save_parameters = True

    # Fit config function. Mainly used to save parameters
    def fit_config(server_round: int) -> dict:
        # '''this function is called before the fit function in FlowerClient to generate the configuration for training'''
        return {"save_parameters": save_parameters, "server_round": server_round}
    
    # Create FedAvg strategy
    strategy = CustomFedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=num_clients,  # Never sample less than 10 clients for training
        min_evaluate_clients=num_clients,  # Never sample less than 10 clients for evaluation
        min_available_clients=num_clients,  # Wait until all 10 clients are available before training
        evaluate_metrics_aggregation_fn=eval_weighted_average, # function for aggregating evaluation metrics (e.g., accuracy)
        fit_metrics_aggregation_fn=fit_weighted_average, # function for aggregating fit metrics (e.g., train loss, train accuracy)
        on_fit_config_fn=fit_config # function for generating training configuration which saves parameters
    )

    data = Data(batch_size=batch_size,
                partitioner=partitioner,
                partition_size=partition_size,
                seed=seed,
                val_size=val_size)

    sim = Simulation(net=net,
                     data=data,
                     num_clients=num_clients,
                     num_rounds=num_rounds,
                     epochs=epochs,
                     device=device,
                     num_cpus=num_cpus,
                     num_gpus=num_gpus,
                     strategy=strategy,
                     criterion=criterion,
                     optim_method=optimizer,
                     lr=lr,
                     )

    run_path = sim.run_simulation()
    print(run_path)