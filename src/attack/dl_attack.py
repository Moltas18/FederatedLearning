# Python imports
from abc import ABC, abstractmethod
from typing import List, Dict

# PyTorch imports
import torch

# Other imports
from tqdm import tqdm

# Local imports
from src.attack import dummy_client
from src.attack.dummy_client import DummyClient

'''
DLAttack is an abstract class for all optimization based deep learning attacks.

'''
class DLAttack(ABC):
    
    def __init__(self,
                 dummy_model: torch.nn.Module,
                 dummy_criterion: torch.nn.Module,
                 dummy_lr: float,
                 reconstruction_optimizer: torch.optim.Optimizer,
                 reconstruction_lr: float,
                 device: str,
                 global_parameters: List[Dict[str, torch.Tensor]] = None,
                 locally_updated_parameters: List[Dict[str, torch.Tensor]] = None,
                 real_labels: torch.Tensor = None
                 ):
        
        # Create a dummy client to compute gradients w.r.t the provided hyperparameters.
        # The client is initiated with the global parameters of the model. This is very important to be aware of as 
        # the gradients are computed w.r.t the global parameters and not the locally updated parameters.
        # Change this if you want to compute the gradients w.r.t the locally updated parameters or any other parameters!
        
        self._dummy_client = DummyClient(model=dummy_model,
                                         criterion=dummy_criterion,
                                         device=device,
                                         initial_parameters=global_parameters[0])
        
        # Initialize the global parameters list
        if global_parameters is None:
            global_parameters = []  # Initialize as an empty list to avoid mutable default issues
        self._global_parameters = global_parameters

        # Initialize the locally updated parameters list
        if locally_updated_parameters is None:
            locally_updated_parameters = []  # Initialize as an empty list to avoid mutable default issues
        self._locally_updated_parameters = locally_updated_parameters

        # Set the device
        self._device = device

        # Set the reconstruction optimizer and lr
        self._reconstruction_optimizer = reconstruction_optimizer
        self._reconstruction_lr = reconstruction_lr

        # Set the real labels
        self._real_label = real_labels
        
        # Do this step in the attack!
        # reconstruction_optimizer = reconstruction_optimizer([dummy_data], lr=0.01)

    @abstractmethod
    def generate_dummy_data_and_dummy_label(self):
        '''Create a dummy data and label for the attack
        example: dummy_data = torch.randn_like(real_data, requires_grad=True, device=self._device)
                 dummy_label = self._real_label

        The dummy_label can also be a random tensor initialized with random values
        # fake_label = torch.randn((1, 10), requires_grad=True, device=device)

        The function should return the dummy_data and dummy_label and make sure to set the requires_grad=True!
        
        '''
        pass

    @abstractmethod
    def create_closure_fn(self, reconstruction_optimizer: torch.optim.Optimizer, optimization_variables: List[torch.Tensor]):
        '''Create a closure function for the optimization'''
        pass

    def initialize_optimization_variables(self, dummy_data: torch.Tensor):
        '''Initialize the optimization variables for the attack
           rewrite this function if you have more than one optimization variable!
        '''
        return [dummy_data]

    def run_attack(self, iterations: int):
        '''Run the attack'''

        # Now the issue is that we might want to extend the attack to include more than one optimization variable
        
        # Initialize fake input and label
        dummy_data, dummy_label = self.generate_dummy_data_and_dummy_label()
        
        # Put optimization variables in a list, could be several that needs tracking!
        optimization_variables = self.initialize_optimization_variables(dummy_data)
        
        # Initialize the optimizer
        reconstruction_optimizer = self._initialize_optimizer(optimization_variables)

        # Closure function for optimization
        closure = self.create_closure_fn(reconstruction_optimizer=reconstruction_optimizer, 
                                         optimization_variables=optimization_variables)
        loss_history = []   
        # Perform gradient matching
        for _ in tqdm(range(iterations)):  # Iterations for optimization
            reconstruction_optimizer.step(closure)
            loss_history.append(closure().item())
            # if step % 10 == 0:
            #     print(f"Step {step}: Gradient Matching Loss: {closure().item()}")
    
        return dummy_data, dummy_label
    
    def _initialize_optimizer(self, optimization_variables: List[torch.Tensor]):
        '''Initialize the optimizer for the optimization'''
        return self._reconstruction_optimizer(optimization_variables, lr=self._reconstruction_lr)
        
    def _compute_gradients(self, x, y):
        '''Computes the gradients of the loss w.r.t the model parameters and input x'''
        return self._dummy_client.compute_gradients(x, y)