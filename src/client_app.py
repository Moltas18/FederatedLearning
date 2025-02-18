from flwr.client import  NumPyClient
from flwr.common import Context, ConfigsRecord, MetricsRecord
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
    total_correct, total_loss, total_samples = 0, 0.0, 0
    for epoch in range(epochs):
        correct, epoch_loss, total = 0, 0.0, 0
        for batch in dataloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        
        total_loss += epoch_loss
        total_correct += correct
        total_samples += total

    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy


    #     if total != len(dataloader.dataset):
    #         print(len(dataloader.dataset))
    #         print(total)
    #         print(len(dataloader))
    #         ValueError('Nu är det kört!')

    #     metric['accuracy'].append(float(epoch_acc))
    #     metric['loss'].append(float(epoch_loss))
    #     metric['samples'] = int(total)

    # return metric

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
        
        metric_dict = {"validation_accuracy": float(accuracy)} 
        return float(loss), len(self._valloader), metric_dict