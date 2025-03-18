import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # If using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(CNN, self).__init__()
        self.num_classes = num_classes
        # Define the convolutional layers with more filters
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # More filters and smaller kernel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # More filters
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # More filters

        # Fully connected layers with more neurons
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)  # Increased number of neurons
        self.fc2 = nn.Linear(1024, 512)          # Increased number of neurons
        self.fc3 = nn.Linear(512, 256)           # Increased number of neurons
        self.fc4 = nn.Linear(256, self.num_classes)            # Output layer for CIFAR-10 (10 classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply convolutional layers and pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output: (batch_size, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # Output: (batch_size, 64, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))  # Output: (batch_size, 128, 4, 4)

        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 128 * 4 * 4)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Final output layer
        
        return x