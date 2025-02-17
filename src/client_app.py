from flwr.client import  NumPyClient
import torch
from collections import OrderedDict
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader

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
    """Train the network on the training set."""
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in dataloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        #     # Metrics
        #     epoch_loss += loss
        #     total += labels.size(0)
        #     correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        # epoch_loss /= len(dataloader.dataset)
        # epoch_acc = correct / total

def test(net, dataloader, device, criterion):
    """Evaluate the network on the entire test set."""
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(dataloader.dataset)
    accuracy = correct / total
    return loss, accuracy

class FlowerClient(NumPyClient):
    
    def __init__(
        self,
        net : torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        epochs: int,
        device: str,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer
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

    def get_parameters(self, config):
        return get_parameters(self._net)

    def fit(self, parameters, config):
        ''' Function utilized by the server'''
        set_parameters(self._net, parameters)
        train(net=self._net,
              dataloader=self._trainloader,
              epochs=self._epochs,
              device=self._device,
              optimizer=self._optimizer,
              criterion=self._criterion)
        
        metric_dict = {}
        return get_parameters(self._net), len(self._trainloader), metric_dict

    def evaluate(self, parameters, config):
        ''' Function utilized by the server'''
        set_parameters(self._net, parameters)
        loss, accuracy = test(net=self._net,
                              dataloader=self._valloader,
                              device=self._device,
                              criterion=self._criterion)
        
        metric_dict = {"accuracy": float(accuracy)}
        return float(loss), len(self._valloader), metric_dict