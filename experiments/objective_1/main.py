'''This file will be used to run the first objective!'''

import sys
import os
from flwr.server.strategy import FedAvg


if __name__ == '__main__':
    
    # Add the root directory to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    from src.simulation import Simulation
    from src.models.models import Net
    from src.utils import weighted_average

    num_clients = 2
    num_rounds = 1

    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=num_clients,  # Sample all clients for training
        min_evaluate_clients=num_clients,  # Sample all clients for evaluation
        min_available_clients=num_clients,  # Wait until all clients are available
        evaluate_metrics_aggregation_fn=weighted_average
    )

    sim = Simulation(net=Net(),
                     num_clients=num_clients,
                     strategy=strategy,
                     num_rounds=num_rounds,
                     )

    
    sim.run_simulation()