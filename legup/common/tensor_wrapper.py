import torch


class TensorWrapper:
    def __init__(self, tensor: torch.Tensor, end_dims: int):
        self.tensor = tensor
        self.pre_shape = tensor.shape[:-end_dims]
        self.device = tensor.device

    def __getitem__(self, index):
        return self.tensor[index]

    def __setitem__(self, index, value):
        self.tensor[index] = value
