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
    from src.models.models import LeNet5, CNN 
    from src.utils import fit_weighted_average, eval_weighted_average, plot_run_results, read_from_file
    from data.data import Data
    from src.strategy import CustomFedAvg

    ### Configurations

    # Federated learning configurations
    num_clients = 1
    num_rounds = 5

    # Model configurations
    epochs = 1
    net = LeNet5()
    criterion = torch.nn.CrossEntropyLoss()

    # Data configurations
    batch_size = 'full'
    val_test_batch_size = 256
    val_size = 0.2
    partitioner = num_clients

    # General configurations
    seed = 42
    num_gpus = 1/num_clients
    num_cpus = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Save parameters
    save_parameters = True

    # Fit config function. Mainly used to save parameters
    def fit_config(server_round: int) -> dict:
        '''this function is called before the fit function in FlowerClient to generate the configuration for training'''
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
                seed=seed,
                val_size=val_size,
                val_test_batch_size=val_test_batch_size)

    sim = Simulation(net=net,
                     data=data,
                     num_clients=num_clients,
                     num_rounds=num_rounds,
                     epochs=epochs,
                     device=device,
                     num_cpus=num_cpus,
                     num_gpus=num_gpus,
                     strategy=strategy,
                     criterion=criterion
                     )

    run_path = sim.run_simulation()

    config_path = run_path /  'run_config.jsonl'
    metrics_path = run_path / 'metrics.jsonl'
    parameters_path = run_path / 'parameters.jsonl'

    plot_run_results(metrics_path=metrics_path, config_path=config_path)