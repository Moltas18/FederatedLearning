'''
This file is the template upon which simulations can be build
'''
import sys
import os
from flwr.server.strategy import FedAvg
from flwr_datasets.partitioner import DirichletPartitioner
import torch

if __name__ == '__main__':
    
    # Add the root directory to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    # Import local modules
    from src.simulation import Simulation
    from src.models.models import LeNet5
    from src.utils import weighted_average
    from data.data import Data

    # Configurations
    num_clients = 2
    num_rounds = 2
    batch_size = "full"
    test_size = 0.2
    seed = 42
    partitioner = num_clients

    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=num_clients,  # Never sample less than 10 clients for training
        min_evaluate_clients=num_clients,  # Never sample less than 5 clients for evaluation
        min_available_clients=num_clients,  # Wait until all 10 clients are available
        evaluate_metrics_aggregation_fn=weighted_average
    )

    data = Data(batch_size=batch_size, partitioner=partitioner, seed=seed, test_size=test_size)

    sim = Simulation(net=LeNet5(),
                     num_clients=num_clients,
                     strategy=strategy,
                     num_rounds=num_rounds,
                     data=data,
                     )

    sim.run_simulation()

    