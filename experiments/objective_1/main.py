'''
This file is the template upon which simulations can be build
'''
import sys
import os

if __name__ == '__main__':
    
    # Add the root directory to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    # Import local modules
    from src.simulation import Simulation
    from src.models.models import Net
    from src.utils import fit_weighted_average, eval_weighted_average
    from data.data import Data
    from src.strategy import CustomFedAvg

    # Configurations
    num_clients = 10
    num_rounds = 50
    batch_size = "full"
    test_size = 0.2
    seed = 42
    partitioner = num_clients
    num_gpus = 1/num_clients
    epochs = 1

    # Create FedAvg strategy
    strategy = CustomFedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=num_clients,  # Never sample less than 10 clients for training
        min_evaluate_clients=num_clients,  # Never sample less than 5 clients for evaluation
        min_available_clients=num_clients,  # Wait until all 10 clients are available
        evaluate_metrics_aggregation_fn=eval_weighted_average,
        fit_metrics_aggregation_fn=fit_weighted_average
    )

    data = Data(batch_size=batch_size, partitioner=partitioner, seed=seed, test_size=test_size)

    sim = Simulation(net=Net(),
                     num_clients=num_clients,
                     strategy=strategy,
                     num_rounds=num_rounds,
                     data=data,
                     num_gpus=num_gpus,
                     epochs=1
                     )

    sim.run_simulation()