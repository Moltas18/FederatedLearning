from flwr.client import  NumPyClient
from flwr.common import Context, ConfigsRecord, MetricsRecord
import torch
from collections import OrderedDict
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.classification import Accuracy

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # If using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_parameters(net, parameters: List[np.ndarray]):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

def train(net, dataloader, epochs, device, optimizer, criterion):
    """Train the network using torchmetrics."""
    net.train()
    
    accuracy_metric = Accuracy(task="multiclass", num_classes=net.num_classes).to(device)

    for epoch in range(epochs):
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

        # Later, we'll use this code to log the parameters!
        # self.client_state = (
        #     context.state
        # )  # add a reference to the state of your ClientApp

        # # Here we create all the metrics!
        # if "eval_metrics" not in self.client_state.metrics_records:
        #     self.client_state.metrics_records["eval_metrics"] = MetricsRecord()

        # if "fit_metrics" not in self.client_state.metrics_records:
        #     self.client_state.metrics_records["fit_metrics"] = MetricsRecord()
            
    def get_parameters(self, config):
        return get_parameters(self._net)

    def fit(self, parameters, config):
        ''' Function utilized by the server'''
        set_parameters(self._net, parameters)
        train_loss, train_acc = train(net=self._net,
                                    dataloader=self._trainloader,
                                    epochs=self._epochs,
                                    device=self._device,
                                    optimizer=self._optimizer,
                                    criterion=self._criterion)

        metrics = {"train_loss": train_loss, "train_accuracy": train_acc}
        return get_parameters(self._net), len(self._trainloader), metrics

    def evaluate(self, parameters, config):
        ''' Function utilized by the server'''
        set_parameters(self._net, parameters)
        loss, accuracy = test(net=self._net,
                              dataloader=self._valloader,
                              device=self._device,
                              criterion=self._criterion)
        
        metrics = {"validation_accuracy": float(accuracy)} 
        return float(loss), len(self._valloader), metrics