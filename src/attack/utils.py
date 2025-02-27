import matplotlib.pyplot as plt
import torch
import pandas as pd 
import numpy as np
from collections import OrderedDict
from flwr.common import Metrics
import yaml
from typing import List, Tuple, Union
from pathlib import Path
from time import time 
import json
import torch
import torch.nn.functional as F
from typing import List

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # If using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GradientApproximation:
    def __init__(self, W0: List[torch.Tensor], W1: List[torch.Tensor], lr: float, epochs: int):
        self.W0 = W0
        self.W1 = W1
        self.lr = lr
        self.epochs = epochs

    def approximate_gradient(self) -> List[torch.Tensor]:
        '''
        Approximate the gradient of the model parameters using the straight away method from https://arxiv.org/abs/2110.09074
        '''
        # Compute the approximate gradient for each pair of tensors in W0 and W1
        approcimated_gradient = [(w0 - w1) / (self.lr * self.epochs) for w0, w1 in zip(self.W0, self.W1)]
        
        return approcimated_gradient

    
class ParameterDifference:
    def __init__(self, W0: List[torch.Tensor], W1: List[torch.Tensor]):
        self.W0 = W0
        self.W1 = W1

    def elementwise_difference(self):
        """Calculate the element-wise difference between corresponding tensors"""
        return [w1 - w0 for w1, w0 in zip(self.W1, self.W0)]

    def frobenius_norm(self):
        """Calculate the Frobenius norm (L2 norm) of the difference between corresponding tensors"""
        diff_norms = [torch.norm(w1 - w0).item() for w1, w0 in zip(self.W1, self.W0)]
        return sum(diff_norms)

    def mean_squared_error(self):
        """Calculate the Mean Squared Error (MSE) between corresponding tensors"""
        mse = [torch.mean((w1 - w0) ** 2).item() for w1, w0 in zip(self.W1, self.W0)]
        return sum(mse)

    def cosine_similarity(self):
        """Calculate the cosine similarity between corresponding tensors"""
        cosine_similarities = [F.cosine_similarity(w1.view(-1), w0.view(-1), dim=0).item() for w1, w0 in zip(self.W1, self.W0)]
        return sum(cosine_similarities) / len(cosine_similarities)

    def sum_of_absolute_differences(self):
        """Calculate the sum of absolute differences between corresponding tensors"""
        sum_abs_diff = [torch.sum(torch.abs(w1 - w0)).item() for w1, w0 in zip(self.W1, self.W0)]
        return sum(sum_abs_diff)

    def plot_difference(self):
        """Plots the difference measures in a bar chart"""
        # Collect the difference measures
        measures = {
            "Element-wise Diff": self.frobenius_norm(),
            "Frobenius Norm": self.frobenius_norm(),
            "Mean Squared Error": self.mean_squared_error(),
            "Cosine Similarity": self.cosine_similarity(),
            "Sum of Abs Diff": self.sum_of_absolute_differences(),
        }
        
        # Plot the differences
        plt.figure(figsize=(10, 6))
        plt.bar(measures.keys(), measures.values(), color='skyblue')
        plt.xlabel('Difference Measures')
        plt.ylabel('Value')
        plt.title('Comparison of Parameter Differences')
        plt.tight_layout()
        plt.show()

