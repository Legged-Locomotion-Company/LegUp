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
        return self.tensor.shape[:-len(self.end_shape)]

    def reshape(self, pre_shape: List[int]):
        return self.tensor.reshape(pre_shape + self.end_shape)

    def to(self, device: torch.device):
        self.tensor = self.tensor.to(device)
        self.device = device

    def unsqeeze_to_broadcast(self, new_pre_shape: List[int]) -> "TensorWrapper":
        """Unsqueezes a tensor to broadcast with another tensor."""

        # Unsqueeze the tensor.
        new_tensor = TensorWrapper._raw_tensor_unsqueeze_to_broadcast(
            self.tensor, new_pre_shape, self.end_shape)

        return type(self).__init__(new_tensor)

    @staticmethod
    def unsqueeze_to_broadcast_tensors(a: "TensorWrapper", b: "TensorWrapper"):
        """Unsqueezes a tensor to broadcast with another tensor."""

        new_pre_shape = torch.broadcast_shapes(a.pre_shape(), b.pre_shape())

        a_broad = a.unsqueeze_to_broadcast(new_pre_shape)
        b_broad = b.unsqueeze_to_broadcast(new_pre_shape)

        return a_broad, b_broad

    @staticmethod
    @torch.jit.script
    def _raw_tensor_unsqueeze_to_broadcast(tensor: torch.Tensor, new_pre_shape: List[int], end_shape: List[int]) -> torch.Tensor:
        """Unsqueezes a tensor to broadcast with another tensor."""

        # Unsqueeze the tensor.
        for _ in range(len(new_pre_shape) - len(tensor.pre_shape)):
            for i in range(len(tensor.pre_shape)):
                if tensor.pre_shape[i] != new_pre_shape[i]:
                    tensor.tensor = tensor.tensor.unsqueeze(i)

        return tensor
