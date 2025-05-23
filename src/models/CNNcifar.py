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

class CNNcifar(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNcifar, self).__init__()
        self.num_classes = num_classes
        self.act = nn.ReLU()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            self.act,
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            self.act,
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(8192, 200)
        self.fc2 = nn.Linear(200, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        return self.fc2(x)