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
import os
from torchvision.transforms import ToPILImage
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List

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

def require_grad(net, flag):
    for p in net.parameters():
        p.require_grad = flag

def prior_boundary(data, low, high):
    with torch.no_grad():
        data.data = torch.clamp(data, low, high)

def compute_norm(inputs):
    squared_sum = sum([p.square().sum() for p in inputs])
    norm = squared_sum.sqrt()
    return norm

def total_variation(x):
    dh = (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean()
    dw = (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()
    return (dh + dw) / 2

def psnr(data, rec, sort=False):
    assert data.max().item() <= 1.0001 and data.min().item() >= -0.0001
    assert rec.max().item() <= 1.0001 and rec.min().item() >= -0.0001
    cost_matrix = []
    if sort:
        for x_ in rec:
            cost_matrix.append(
                [(x_ - d).square().mean().item() for d in data]
            )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assert np.all(row_ind == np.arange(len(row_ind)))
        data = data[col_ind]
    psnr_list = [10 * np.log10(1 / (d - r).square().mean().item()) for d, r in zip(data, rec)]
    return np.mean(psnr_list)

def save_args(**kwargs):
    if os.path.exists(os.path.join(kwargs["path_to_res"], "args.json")):
        os.remove(os.path.join(kwargs["path_to_res"], "args.json"))

    with open(os.path.join(kwargs["path_to_res"], "args.json"), "w") as f:
        json.dump(kwargs, f, indent=4)

def save_figs(tensors, path, subdir=None, dataset=None):
    def save(imgs, path):
        for name, im in imgs:
            plt.figure()
            plt.imshow(im, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(path, f'{name}.png'), bbox_inches='tight')
            plt.close()
    tensor2image = ToPILImage()
    path = os.path.join(path, subdir)
    os.makedirs(path, exist_ok=True)
    if dataset == "FEMNIST":
        tensors = 1 - tensors
    imgs = [
        [i, tensor2image(tensors[i].detach().cpu().squeeze())] for i in range(len(tensors))
    ]
    save(imgs, path)

def allign_images(ground_truth_images: torch.Tensor,
                  reconstructed_images: torch.Tensor
                  ):
    
    cost_matrix = []

    for x_ in reconstructed_images:
        cost_matrix.append(
            [(x_ - d).square().mean().item() for d in ground_truth_images]
        )
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assert np.all(row_ind == np.arange(len(row_ind)))
    ground_truth_images = ground_truth_images[col_ind]
    return ground_truth_images, reconstructed_images