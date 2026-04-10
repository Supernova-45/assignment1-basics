"""Linear class that performs a linear transformation."""

import math

import torch

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = torch.empty(size=(self.out_features, self.in_features), device=self.device, dtype=self.dtype)
        weights_std = math.sqrt(2 / (self.in_features + self.out_features))
        self.W = torch.nn.Parameter(torch.nn.init.trunc_normal_(tensor=self.W, mean=0, std=weights_std, a = -3 * weights_std, b = 3 * weights_std))
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T