import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # If using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# from src.utils import denormalize
# from src.models.models import LeNet5


class Dlg:

    '''
    Deep leakage from gradients
    '''
    
    def __init__(self,
                 ):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # These should come from a dummy client i think!
        self._model = LeNet5().to(self._device)
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.SGD(self._model.parameters(), lr=0.1)
        self._iterations = 100

        self._create_data()
        self._simulate_theft()

    def _create_data(self):
        # Load CIFAR-10 dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

        real_data, real_label = next(iter(trainloader))
        self._real_data, self._real_label = real_data.to(self._device), real_label.to(self._device)

    def _simulate_theft(self):
        
        self._model.zero_grad()
        output = self._model(self._real_data)
        loss = self._criterion(output, self._real_label)
        loss.backward()
        self._real_gradients = [p.grad.clone() for p in self._model.parameters()]

    def run_single_attack(self
                          ):
        '''
        Run a single attack, specifically on only one gradient observation
        ''' 


        # Initialize fake input and label
        fake_data = torch.randn_like(self._real_data, requires_grad=True, device=self._device)
        # fake_label = torch.randn((1, 10), requires_grad=True, device=device)
        fake_label = self._real_label

        # Optimizer for reconstruction
        reconstruction_optimizer = optim.LBFGS([fake_data], lr=1)

        # Closure function for optimization
        closure = self._create_closure(reconstruction_optimizer=reconstruction_optimizer, fake_data=fake_data, fake_label=fake_label)

        # Perform gradient matching
        for step in range(self._iterations):  # Iterations for optimization
            reconstruction_optimizer.step(closure)
            if step % 10 == 0:
                print(f"Step {step}: Gradient Matching Loss: {closure().item()}")

        display_images(fake_data, self._real_data)

    def _create_closure(self, reconstruction_optimizer, fake_data, fake_label):
        def closure():
            reconstruction_optimizer.zero_grad()
            fake_output = self._model(fake_data)
            loss = self._criterion(fake_output, fake_label)

            dummy_gradients = torch.autograd.grad(loss, self._model.parameters(), create_graph=True, allow_unused=True)

            gradient_loss = sum(F.mse_loss(fg, rg) for fg, rg in zip(dummy_gradients, self._real_gradients))
            gradient_loss.backward(retain_graph=True)
            return gradient_loss
        return closure
    
# Display real vs reconstructed image
def display_images(fake_data, real_data):
    with torch.no_grad():
        fake_image = denormalize(fake_data.clone().detach().cpu().squeeze())
        real_image = denormalize(real_data.clone().detach().cpu().squeeze())

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(real_image.permute(1, 2, 0))  # Convert tensor to image
        ax[0].set_title("Real Image")
        ax[1].imshow(fake_image.permute(1, 2, 0))
        ax[1].set_title("Reconstructed Image")
        plt.show()

if __name__ == '__main__':
    
    # Add the root directory to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    from src.utils import denormalize
    from src.models.models import LeNet5

    attack = Dlg()
    attack.run_single_attack()


