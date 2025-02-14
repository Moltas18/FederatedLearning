'''
This file is the template upon which simulations can be build
'''
import sys
import os
from flwr.server.strategy import FedAvg
from flwr_datasets.partitioner import DirichletPartitioner


if __name__ == '__main__':
    
    # Add the root directory to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    # Import local modules
    from src.simulation import Simulation
    from src.models.models import Net
    from src.utils import weighted_average

    # Configurations
    num_clients = 2
    num_rounds = 5

    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=num_clients,  # Never sample less than 10 clients for training
        min_evaluate_clients=num_clients,  # Never sample less than 5 clients for evaluation
        min_available_clients=num_clients,  # Wait until all 10 clients are available
        evaluate_metrics_aggregation_fn=weighted_average
    )

    partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            partition_by="label",
            alpha=1,
            seed=42,
            min_partition_size=0,
        )

    sim = Simulation(net=Net(),
                     num_clients=num_clients,
                     strategy=strategy,
                     num_rounds=num_rounds,
                     partitioner=partitioner
                     )

    sim.run_simulation()


    