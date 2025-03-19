import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# torch.cuda.manual_seed_all(42)  # If using multi-GPU
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)  # Specific for CIFAR-10
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)  

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x