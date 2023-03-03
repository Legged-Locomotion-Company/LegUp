from typing import Iterable, List, TypeVar

import torch

TensorWrapperSubclass = TypeVar(
    "TensorWrapperSubclass", bound="TensorWrapper")


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

    def to(self: TensorWrapperSubclass, device: torch.device) -> TensorWrapperSubclass:
        """Moves the tensor to a device. This modifies the tensor, so other references to the tensor will be on the new device as well."""
        self.tensor = self.tensor.to(device)
        self.device = device

        return self

    def view(self: TensorWrapperSubclass, shape: Iterable[int]) -> TensorWrapperSubclass:
        if not isinstance(shape, list):
            shape = list(shape)

        new_shape = shape + self.end_shape
        new_raw_tensor = self.tensor.view(new_shape)

        return self.__class__(new_raw_tensor)  # type: ignore

    @staticmethod
    def _default_device():
        """Returns the default device for the current context."""

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
