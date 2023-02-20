from typing import Iterable, List

import torch


class TensorWrapper:
    def __init__(self, tensor: torch.Tensor, end_shape: List[int]):
        self.tensor = tensor
        self.end_shape = end_shape
        self.device = tensor.device

    def __getitem__(self, index):
        return self.tensor[index]

    def __setitem__(self, index, value):
        self.tensor[index] = value

    def pre_shape(self) -> List[int]:
        return list(self.tensor.shape[:-len(self.end_shape)])

    def reshape(self, pre_shape: List[int]):
        return self.tensor.reshape(pre_shape + self.end_shape)

    # No return type annotation because of dynamic return type.
    def to(self, device: torch.device):
        self.tensor = self.tensor.to(device)
        self.device = device

        return self

    def unsqueeze_to_broadcast(self, new_pre_shape: List[int]) -> "TensorWrapper":
        """Unsqueezes a tensor to broadcast to a shape which can be broadcast with another pre_shape."""

        raise NotImplementedError(
            "This method is not implemented for this class.")

    @staticmethod
    def unsqueeze_to_broadcast_tensors(a: "TensorWrapper", b: "TensorWrapper"):
        """Unsqueezes a tensor to broadcast with another tensor."""

        new_pre_shape = torch.broadcast_shapes(a.pre_shape(), b.pre_shape())

        new_pre_shape = list(new_pre_shape)

        a_broad = a.unsqueeze_to_broadcast(new_pre_shape)
        b_broad = b.unsqueeze_to_broadcast(new_pre_shape)

        return a_broad, b_broad

    @staticmethod
    def _default_device():
        """Returns the default device for the current context."""

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
