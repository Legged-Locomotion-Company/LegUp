from typing import Iterable, List

import torch

class TensorWrapper:
    def __init__(self, tensor: torch.Tensor, end_shape: List[int]):
        self.tensor = tensor
        self.end_shape = end_shape
        self.pre_shape = tensor.shape[:-len(self.end_shape)]
        self.device = tensor.device

    def __getitem__(self, index):
        return self.tensor[index]

    def __setitem__(self, index, value):
        self.tensor[index] = value

    def reshape(self, pre_shape: List[int]):
        return self.tensor.reshape(pre_shape + self.end_shape)

    def to(self, device: torch.device):
        self.tensor = self.tensor.to(device)
        self.device = device
