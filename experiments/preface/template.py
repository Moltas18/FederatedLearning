'''
This file is the template upon which simulations can be build
'''
import sys
import os

# Add the root directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch

from datasets.utils.logging import disable_progress_bar

import flwr
from flwr.client import Client, ClientApp
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation


from src.utils import load_datasets
from src.client_app import FlowerClient
from src.models.models import Net



# If available, run with cuda
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()

# Configure backend_config
if DEVICE == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
else:
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# Simulation configuration
num_clients = 5
batch_size = 32
num_rounds = 1

# Create FedAvg strategy
strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=num_clients,  # Never sample less than 10 clients for training
    min_evaluate_clients=num_clients,  # Never sample less than 5 clients for evaluation
    min_available_clients=num_clients,  # Wait until all 10 clients are available
)

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)
    epochs = 1
    device = DEVICE
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, _ = load_datasets(partition_id=partition_id, batch_size=batch_size, num_clients=num_clients)

    return FlowerClient(net, trainloader, valloader, epochs, device).to_client()

def server_fn(context: Context) -> ServerAppComponents:

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create the ClientApp
client = ClientApp(client_fn=client_fn)
# Create the ServerApp
server = ServerApp(server_fn=server_fn)

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=num_clients,
    backend_config=backend_config,
)