from flwr.client import  NumPyClient
import torch
from collections import OrderedDict
from typing import List
import numpy as np
import torch


# How do we inlcude epochs???????
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

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    
    def __init__(
        self,
        net,
        trainloader,
        valloader,
        epochs: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        criterion=None,
        optim_method=None,
    ):
        # Parameters needed from flwr
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

        # Parameters used in project
        self.epochs = epochs
        self.device = device

        # Ensure criterion is properly initialized
        self.criterion = criterion if criterion else torch.nn.CrossEntropyLoss()

        # Ensure optimizer is properly initialized
        self.optim_method = optim_method if optim_method else torch.optim.Adam
        self.optimizer = self.optim_method(self.net.parameters())

        # Move model to correct device
        self.net.to(self.device)

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        ''' Function utilized by the server'''
        set_parameters(self.net, parameters)
        train(net=self.net,
              dataloader=self.trainloader,
              epochs=self.epochs,
              device=self.device,
              optimizer=self.optimizer,
              criterion=self.criterion)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        ''' Function utilized by the server'''
        set_parameters(self.net, parameters)
        loss, accuracy = test(net=self.net,
                              dataloader=self.valloader,
                              device=self.device,
                              criterion=self.criterion)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}