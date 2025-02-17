'''
This file is the template upon which simulations can be build
'''
from typing import Type, Union

import torch
import numpy as np
import flwr
from flwr.client import Client, ClientApp
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import Strategy, FedAvg 
from flwr.simulation import run_simulation
from flwr_datasets.partitioner import Partitioner

from src.utils import timer
from src.client_app import FlowerClient, get_parameters, set_parameters
from data.data import Data

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # If using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Simulation:
    
    def __init__(self,
                 net: torch.nn.Module,
                 data: Data, 
                 num_clients: int = 10,
                 num_rounds: int = 5,
                 epochs: int = 1,
                 device: str = 'cuda',
                 num_cpus: int = 1,
                 num_gpus: int = 1,
                 strategy: Strategy = FedAvg(),
                 criterion = None,
                 optim_method = None,
                 partitioner: Union[int, Partitioner]=10,
                 ) -> None:
        
        # Model
        self._net = net

        # We save the original parameters if needed later
        self._orgininal_parameters = [np.copy(param) for param in get_parameters(self._net)]

        # Ensure criterion is properly initialized
        self._criterion = criterion if criterion else torch.nn.CrossEntropyLoss()

        # Ensure optimizer is properly initialized
        self._optim_method = optim_method if optim_method else torch.optim.Adam
        self._optimizer = self._optim_method(self._net.parameters())

        # Training parameters
        self._epochs = epochs

        # Simulation parameters
        self._num_clients = num_clients
        self._num_rounds = num_rounds
        self._strategy = strategy
        self.partitioner = partitioner
        
        # Device parameters
        self._device = device
        self._num_cpus = num_cpus
        self._num_gpus = num_gpus
        
        # Initialization functions
        self._set_backend_config()
        
        # Controlling functions
        self.check_hardware(self._device)
        self.check_strategy(self._strategy, self._num_clients)
        
        # Save data to class
        self._data = data

    def reset_net(self) -> None:
        # Reset the network parameters
        set_parameters(self._net, self._orgininal_parameters)  # Ensure this updates in-place

        # Reinitialize the optimizer to reset its state
        self._optimizer = self._optim_method(self._net.parameters())
        
    def _set_backend_config(self) -> None:
        if self._device == "cuda":
            self._backend_config = {"client_resources": {"num_cpus": self._num_cpus, "num_gpus": self._num_gpus}}
        else:
            self._backend_config = {"client_resources": {"num_cpus": self._num_cpus, "num_gpus": 0.0}}

    def client_fn(self, context: Context) -> Client:
            """Create a Flower client representing a single organization."""

            # Load model
            net = self._net.to(self._device)
            partition_id = context.node_config["partition-id"]
            trainloader, valloader, _ = self._data.load_datasets(partition_id=partition_id)
            
            return FlowerClient(net,
                                trainloader,
                                valloader,
                                self._epochs,
                                self._device,
                                self._criterion,
                                self._optimizer
                                ).to_client()

    def server_fn(self, context: Context) -> ServerAppComponents:

        # Configure the server for 5 rounds of training
        config = ServerConfig(num_rounds=self._num_rounds)

        return ServerAppComponents(strategy=self._strategy, config=config)
    
    @timer
    def run_simulation(self):
        
        # Create the ClientApp
        client = ClientApp(client_fn=self.client_fn)
        # Create the ServerApp
        server = ServerApp(server_fn=self.server_fn)

        # Run simulation

        run_simulation(
            server_app=server,
            client_app=client,
            num_supernodes=self._num_clients,
            backend_config=self._backend_config,
        )

    @staticmethod
    def check_hardware(device: str) -> None:
        prompt = f'Using device: {device}\nDevice name: {torch.cuda.get_device_name(device)}\nFlower {flwr.__version__} / PyTorch {torch.__version__}'
        print(prompt)

    @staticmethod
    def check_strategy(strategy: Strategy, num_clients: int) -> None:
        if strategy.min_available_clients > num_clients:    
            raise ValueError(
                "The number of clients must be greater than the minimum number of clients required to start a round!"
            )

    @property
    def net(self) -> torch.nn.Module:
        return self._net

    @net.setter
    def net(self, value: torch.nn.Module) -> None:
        self._net = value

    @property
    def criterion(self) -> torch.nn.modules.loss._Loss:
        return self._criterion

    @criterion.setter
    def criterion(self, value: torch.nn.modules.loss._Loss) -> None:
        self._criterion = value

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: Type[torch.optim.Optimizer]) -> None:
        self._optimizer = value(self._net.parameters())

    @property
    def epochs(self) -> int:
        return self._epochs

    @epochs.setter
    def epochs(self, value: int) -> None:
        self._epochs = value

    @property
    def num_clients(self) -> int:
        return self._num_clients

    @num_clients.setter
    def num_clients(self, value: int) -> None:
        self._num_clients = value

    @property
    def num_rounds(self) -> int:
        return self._num_rounds

    @num_rounds.setter
    def num_rounds(self, value: int) -> None:
        self._num_rounds = value

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @strategy.setter
    def strategy(self, value: Strategy) -> None:
        self._strategy = value

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        self._device = value

    @property
    def num_cpus(self) -> int:
        return self._num_cpus

    @num_cpus.setter
    def num_cpus(self, value: int) -> None:
        self._num_cpus = value

    @property
    def num_gpus(self) -> int:
        return self._num_gpus

    @num_gpus.setter
    def num_gpus(self, value: int) -> None:
        self._num_gpus = value