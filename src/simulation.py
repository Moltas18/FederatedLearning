'''
This file is the template upon which simulations can be built.
'''

from typing import Type, Union
from datetime import datetime
from pathlib import Path
import json

import torch
import numpy as np
import flwr
from flwr.client import Client, ClientApp
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import Strategy, FedAvg 
from flwr.simulation import run_simulation

from src.utils import timer, write_to_file
from src.client_app import FlowerClient, get_parameters, set_parameters
from data.data import Data

class Simulation:
    """
    A class to manage and run federated learning simulations using Flower.
    """

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
                 criterion=None,
                 optim_method=None,
                 lr: float = 0.001,
                 ) -> None:
        """
        Initialize the simulation with the given parameters.

        Args:
            net (torch.nn.Module): The neural network model to be used.
            data (Data): The dataset object for partitioning and loading data.
            num_clients (int): Number of clients in the simulation.
            num_rounds (int): Number of federated learning rounds.
            epochs (int): Number of local training epochs per client.
            device (str): Device to use ('cuda' or 'cpu').
            num_cpus (int): Number of CPUs to allocate per client.
            num_gpus (int): Number of GPUs to allocate per client.
            strategy (Strategy): Federated learning strategy (default: FedAvg).
            criterion: Loss function (default: CrossEntropyLoss).
            optim_method: Optimizer method (default: Adam).
            lr (float): Learning rate for the optimizer.
        """
        # Model
        self._net = net

        # Save the original parameters for resetting the model
        self._orgininal_parameters = [np.copy(param) for param in get_parameters(self._net)]

        # Initialize loss function
        self._criterion = criterion if criterion else torch.nn.CrossEntropyLoss()

        # Initialize optimizer
        self._optim_method = optim_method if optim_method else torch.optim.Adam
        self._optimizer = self._optim_method(self._net.parameters(), lr=lr)

        # Training parameters
        self._epochs = epochs

        # Simulation parameters
        self._num_clients = num_clients
        self._num_rounds = num_rounds
        self._strategy = strategy
        
        # Device parameters
        self._device = device
        self._num_cpus = num_cpus
        self._num_gpus = num_gpus

        # Save data to class
        self._data = data

        # Validate hardware and strategy
        self.check_hardware(self._device)
        self.check_strategy(self._strategy, self._num_clients)

        # Backend configuration and run directory setup
        self._set_backend_config()
        self.save_path, self.run_dir = self.create_run_dir()

    def reset_net(self) -> None:
        """
        Reset the network parameters to their original state and reinitialize the optimizer.
        """
        set_parameters(self._net, self._orgininal_parameters)  # Ensure this updates in-place
        self._optimizer = self._optim_method(self._net.parameters())

    def _set_backend_config(self) -> None:
        """
        Configure the backend resources for the simulation.
        """
        if self._device == "cuda":
            self._backend_config = {"client_resources": {"num_cpus": self._num_cpus, "num_gpus": self._num_gpus}}
        else:
            self._backend_config = {"client_resources": {"num_cpus": self._num_cpus, "num_gpus": 0.0}}

    def client_fn(self, context: Context) -> Client:
        """
        Create a Flower client representing a single organization.

        Args:
            context (Context): Context object containing client-specific configurations.

        Returns:
            Client: A Flower client instance.
        """
        # Load model and partition data
        net = self._net.to(self._device)
        partition_id = context.node_config["partition-id"]
        trainloader, valloader, _ = self._data.load_datasets(partition_id=partition_id)

        return FlowerClient(
            net=net,
            trainloader=trainloader,
            valloader=valloader,
            epochs=self._epochs,
            device=self._device,
            criterion=self._criterion,
            optimizer=self._optimizer,
            save_path=self.save_path,
            context=context,
        ).to_client()

    def server_fn(self, context: Context) -> ServerAppComponents:
        """
        Create a Flower server with the specified strategy and configuration.

        Args:
            context (Context): Context object containing server-specific configurations.

        Returns:
            ServerAppComponents: The server components for the simulation.
        """
        config = ServerConfig(num_rounds=self._num_rounds)
        return ServerAppComponents(strategy=self._strategy, config=config)

    @timer
    def run_simulation(self) -> Path:
        """
        Run the federated learning simulation.

        Returns:
            Path: The path where the simulation results are saved.
        """
        # Save configuration files
        config_dict = self.get_config_dict()
        write_to_file(data=config_dict, path=self.save_path, filename='run_config')
        
        data_dict = self._data.get_config_dict()
        write_to_file(data=data_dict, path=self.save_path, filename='data_config')
        
        # Add the save path to the strategy
        self._strategy.save_path = self.save_path

        # Create the ClientApp and ServerApp
        client = ClientApp(client_fn=self.client_fn)
        server = ServerApp(server_fn=self.server_fn)

        # Run the simulation
        run_simulation(
            server_app=server,
            client_app=client,
            num_supernodes=self._num_clients,
            backend_config=self._backend_config,
            verbose_logging=True
        )

        return self.save_path

    def get_config_dict(self) -> dict:
        """
        Generate a dictionary containing the simulation configuration.

        Returns:
            dict: The simulation configuration.
        """
        return {
            'net': self._net.__class__.__name__,
            'data': self._data.dataset, 
            'num_clients': self._num_clients,
            'num_rounds': self._num_rounds,
            'epochs': self._epochs,
            'batch_size': self._data._batch_size,
            'device': self._device,
            'num_cpus': self._num_cpus,
            'num_gpus': self._num_gpus,
            'strategy': self.__class__.__name__,
            'criterion': self._criterion.__class__.__name__,
            'optim_method': self._optimizer.__class__.__name__,
            'learning_rate': self._optimizer.param_groups[0]['lr']
        }

    @staticmethod
    def create_run_dir() -> tuple:
        """
        Create a directory to save results from this run.

        Returns:
            tuple: The save path and run directory.
        """
        current_time = datetime.now()
        run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
        save_path = Path.cwd() / f"results/{run_dir}"
        save_path.mkdir(parents=True, exist_ok=False)
        return save_path, run_dir

    @staticmethod
    def check_hardware(device: str) -> None:
        """
        Check the hardware configuration and print device information.

        Args:
            device (str): The device to use ('cuda' or 'cpu').
        """
        prompt = f'Using device: {device}\nDevice name: {torch.cuda.get_device_name(device)}\nFlower {flwr.__version__} / PyTorch {torch.__version__}'
        print(prompt)

    @staticmethod
    def check_strategy(strategy: Strategy, num_clients: int) -> None:
        """
        Validate the strategy configuration.

        Args:
            strategy (Strategy): The federated learning strategy.
            num_clients (int): The number of clients in the simulation.

        Raises:
            ValueError: If the number of clients is less than the minimum required by the strategy.
        """
        if strategy.min_available_clients > num_clients:    
            raise ValueError(
                "The number of clients must be greater than the minimum number of clients required to start a round!"
            )

    # Property and setter functions
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