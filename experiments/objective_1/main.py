import sys
import os
import torch

if __name__ == '__main__':
    
    # Add the root directory to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    # Import local modules
    from src.simulation import Simulation
    from src.models.CNNcifar import CNNcifar
    from src.models.LeNet5 import LeNet5
    from src.utils import fit_weighted_average, eval_weighted_average, plot_run_results, set_global_seed
    from data.data import Data
    from src.strategy import CustomFedAvg

    # Initialize seed, the set_global_seed function handles seeds in python, torch and numpy but not flower directly.
    seed = 42
    set_global_seed(seed=seed)

    ### Configurations

    # Federated learning configurations
    num_clients = 10
    num_rounds = 60

    # Model configurations
    epochs = 5
    net = CNNcifar()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD
    lr = 0.004

    # Data configurations
    batch_size = 16
    val_test_batch_size = 256
    val_size = 0.2
    partitioner = num_clients
    partition_size = 5000

    # General configurations
    seed = 42
    num_gpus = 1/num_clients
    num_cpus = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Save parameters
    save_parameters = False

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

    config_path = run_path /  'run_config.jsonl'
    metrics_path = run_path / 'metrics.jsonl'

    plot_run_results(metrics_path=metrics_path, config_path=config_path)
