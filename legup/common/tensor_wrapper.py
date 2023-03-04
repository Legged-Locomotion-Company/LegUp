from typing import Iterable, List, TypeVar

from abc import ABC, abstractmethod

import torch

TensorWrapperSubclass = TypeVar(
    "TensorWrapperSubclass", bound="TensorWrapper")


class TensorWrapper(ABC):

    @abstractmethod
    def __init__(self, wrapped_tensor: torch.Tensor) -> None:
        pass

    def initialize_base(self, tensor: torch.Tensor, end_shape: List[int]):
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

    def unsqueeze_to_broadcast(self: TensorWrapperSubclass, pre_shape: Iterable[int]) -> TensorWrapperSubclass:
        """This function unsqueezes this tensor until it is broadcastable with pre_shape,
        or it throws an error if it cannot be broadcasted.

        Args:
            pre_shape (Iterable(int)): This is the shape to broadcast to

        Raises:
            ValueError: This is thrown if the tensor cannot be broadcasted to shape

        Returns:
            TensorWrapperSubclass: This is the tensor with the correct shape
        """

        pre_shape = list(pre_shape)

        new_shape = pre_shape + self.end_shape

        new_tensor_wrapper = self.__class__(self.tensor)

        next_shape_idx = -len(self.pre_shape())

        while len(new_tensor_wrapper.pre_shape()) < len(pre_shape) and next_shape_idx >= -len(new_shape):
            if new_tensor_wrapper.pre_shape()[next_shape_idx] != pre_shape[next_shape_idx]:
                new_tensor_wrapper = \
                    new_tensor_wrapper.unsqueeze(
                        next_shape_idx - len(new_tensor_wrapper.end_shape))
            next_shape_idx -= 1

        if len(new_tensor_wrapper.pre_shape()) != len(pre_shape):
            raise ValueError(
                f"Cannot broadcast tensor of shape {self.pre_shape()} to shape {pre_shape}")

        return new_tensor_wrapper

    @ staticmethod
    def stack(tensors: Iterable[TensorWrapperSubclass]) -> TensorWrapperSubclass:
        """Stacks a list of tensors into a single tensor. This is a static method, so it can be called on the class itself."""

        raw_tensors = [tensor.tensor for tensor in tensors]
        stacked_tensor = torch.stack(raw_tensors)

        return tensors[0].__class__(stacked_tensor)  # type: ignore

    def view(self: TensorWrapperSubclass, shape: Iterable[int]) -> TensorWrapperSubclass:
        if not isinstance(shape, list):
            shape = list(shape)

        new_shape = shape + self.end_shape
        new_raw_tensor = self.tensor.view(new_shape)

        return self.__class__(new_raw_tensor)  # type: ignore

    def unsqueeze(self: TensorWrapperSubclass, idx: int) -> TensorWrapperSubclass:
        new_raw_tensor = self.tensor.unsqueeze(idx)

        return self.__class__(new_raw_tensor)

    @ staticmethod
    def get_broadcast_pre_shape(tensor_wrappers: Iterable["TensorWrapper"]) -> List[int]:
        """This function broadcasts a list of tensors to the same shape.

        Args:
            tensor_wrappers (Iterable(TensorWrapperSubclass)): This is the list of tensors to broadcast

        Returns:
            List[int] This is a the new pre_shape
        """

        broadcasted_pre_shape = torch.broadcast_shapes(
            *[tensor_wrapper.pre_shape()
              for tensor_wrapper in tensor_wrappers])

        return list(broadcasted_pre_shape)

    @ staticmethod
    def _default_device():
        """Returns the default device for the current context."""

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
