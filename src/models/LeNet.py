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

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)  # Output: (6, 32, 32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output size halves
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)  # No padding now
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # Corrected input size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (6, 32, 32) -> (6, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # (16, 12, 12) -> (16, 6, 6)
        x = torch.flatten(x, 1)  # Flatten to (batch_size, 16*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Logits output
        return x