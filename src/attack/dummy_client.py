import torch

class DummyClient:
    ''' Creates an instance of a dummy client that only returns gradients of the model w.r.t the data
    ''' 
    def __init__(self, model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 device: str,
                 initial_parameters: dict):
        
        self._model = model.to(device)
        self._criterion = criterion()
        self._device = device

        # Set the initial parameters, must be done!
        self._set_parameters(initial_parameters)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass of the model'''
        return self._model(x)
    
    def _loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        '''Calculate the loss'''
        return self._criterion(y_pred, y_true)

    def compute_gradients(self, x: torch.Tensor, y: torch.Tensor):
        '''Computes the gradients of the loss w.r.t the model parameters and input x'''
    
        outputs = self._forward(x)
        loss = self._loss(outputs, y)

        # # Compute gradients
        # self._model.zero_grad()  # Clear any existing gradients
        # loss.backward(retain_graph=True)

        # Compute dummy gradient (gradients of loss w.r.t model parameters)
        dummy_gradients = torch.autograd.grad(loss, self._model.parameters(), create_graph=True, allow_unused=True)

        # # Collect gradients of the model parameters
        # param_grads = {name: param.grad.clone() for name, param in self._model.named_parameters() if param.grad is not None}

        # # Compute gradients of the loss w.r.t. x
        # x_grad = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]

        return dummy_gradients

    def _set_parameters(self, parameters: dict):
        '''Sets the model parameters from a dictionary'''
        self._model.load_state_dict(parameters)