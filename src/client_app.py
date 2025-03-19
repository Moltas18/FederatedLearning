from flwr.client import  NumPyClient
from flwr.common import Context, ParametersRecord, array_from_numpy
import torch
from collections import OrderedDict
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.classification import Accuracy
from src.utils import write_to_file, serialize_parameters, deserialize_parameters, get_parameters, set_parameters

def get_gradients(net) -> List[np.ndarray]:
    return [val.grad.cpu().numpy() for _, val in net.named_parameters() if val.grad is not None]

def train(net, dataloader, epochs, device, optimizer, criterion) -> float:
    """Train the network using torchmetrics."""
    net.train()
    
    accuracy_metric = Accuracy(task="multiclass", num_classes=net.num_classes).to(device)

    for _ in range(epochs):
        total_loss, total_samples = 0.0, 0

        for batch in dataloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item() * labels.size(0)  # Weighted loss
            total_samples += labels.size(0)
            accuracy_metric.update(outputs, labels)  # Update torchmetrics

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_accuracy = accuracy_metric.compute().item()  # Compute final accuracy

    return avg_loss, avg_accuracy

def test(net, dataloader, device, criterion):
    """Evaluate the network using torchmetrics."""
    net.eval()
    accuracy_metric = Accuracy(task="multiclass", num_classes=net.num_classes).to(device)

    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = net(images)
            batch_loss = criterion(outputs, labels).item() * labels.size(0)  # Weighted loss
            
            total_loss += batch_loss
            total_samples += labels.size(0)
            accuracy_metric.update(outputs, labels)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_accuracy = accuracy_metric.compute().item()  # Compute accuracy

    return avg_loss, avg_accuracy

class FlowerClient(NumPyClient):
    
    def __init__(
        self,
        net : torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        epochs: int,
        device: str,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        save_path: str,
        context: Context, 
        ) -> None:
        
        # Parameters needed from flwr
        self._net = net
        self._trainloader = trainloader
        self._valloader = valloader

        # Parameters used in project
        self._epochs = epochs
        self._device = device
        self._criterion = criterion 
        self._optimizer = optimizer

        # Move model to correct device
        self._net.to(self._device)

        # Save the context to the instance
        self._context = context

        # Save the path to save the parameters
        # include /parameters after the path
        self._save_path = save_path
            
    def get_parameters(self, config):
        return get_parameters(self._net)

    def fit(self, parameters, config):
        ''' Function utilized by the server'''
        
        # Set the parameters to the model
        set_parameters(self._net, parameters)

        # Save serialized initial parameters (These parameters comes from the global model)
        if config['save_parameters']:
            serialized_initial_parameters = serialize_parameters(self._net, parameters)
        
        train_loss, train_acc = train(net=self._net,
                                    dataloader=self._trainloader,
                                    epochs=self._epochs,
                                    device=self._device,
                                    optimizer=self._optimizer,
                                    criterion=self._criterion)
        
        # Get the updated parameters from the model
        updated_parameters = get_parameters(self._net)
        
        # Save serialized updated parameters
        if config['save_parameters']:
            serialized_updated_parameters = serialize_parameters(self._net, updated_parameters)

            # Create a dictionary to save the parameters
            serialized_parameters = {'parameters_before_training': serialized_initial_parameters,
                                     'parameters_after_training': serialized_updated_parameters}
            

            # Run info from config and context
            run_info = {'server_round': config['server_round'],
                        'client_id': self._context.node_id,
                        'node_config': self._context.node_config,
                        'batch_size': self._trainloader.batch_size,
                        'num_batches': len(self._trainloader),
                        }

            # Save the parameters to a file
            self._write_to_file(parameters=serialized_parameters,
                                run_info=run_info)

        metrics = {"train_loss": train_loss, "train_accuracy": train_acc}
        return updated_parameters, len(self._trainloader), metrics

    def evaluate(self, parameters, config):
        ''' Function utilized by the server'''
        set_parameters(self._net, parameters)
        loss, accuracy = test(net=self._net,
                              dataloader=self._valloader,
                              device=self._device,
                              criterion=self._criterion)
        
        metrics = {"validation_accuracy": float(accuracy)} 
        return float(loss), len(self._valloader), metrics
    
    def _write_to_file(self, parameters:dict={}, run_info:dict={})-> None:
        ''' Function to write parameters to a file specifically for parameters'''
        data = {
            "run_info": run_info,
            "parameters": parameters,
        }
        # A new direcotry needs to be created were parameters JSONL-files are stored for each client
        parameters_path = self._save_path / "parameters"
        parameters_path.mkdir(parents=True, exist_ok=True)

        # We create one parameters-file per client inside the parameters directory
        write_to_file(data=data, path=parameters_path, filename=str(self._context.node_id))
        