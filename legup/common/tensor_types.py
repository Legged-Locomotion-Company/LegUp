from typing import Iterable, List, TypeVar, Dict, Union, Generic, get_args, Callable

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

    def __getitem__(self: TensorWrapperSubclass, *index) -> TensorWrapperSubclass:

        raw_index = [*index, *self.end_shape]

        raw_slice_tensor = self.tensor[raw_index]

        return self.__class__(raw_slice_tensor)

    def __setitem__(self, index, value):
        self.tensor[index] = value

    def pre_shape(self) -> List[int]:
        return list(self.tensor.shape[: -len(self.end_shape)])

    def reshape(self, pre_shape: List[int]):
        return self.tensor.reshape(pre_shape + self.end_shape)

    def to(self: TensorWrapperSubclass, device: torch.device) -> TensorWrapperSubclass:
        """Moves the tensor to a device. This modifies the tensor, so other references to the tensor will be on the new device as well."""
        self.tensor = self.tensor.to(device)
        self.device = device

        return self

    def unsqueeze_to_broadcast(self: TensorWrapperSubclass, pre_shape: Iterable[int]) -> TensorWrapperSubclass:
        """Creates a new tensor which is unsqueezed until it is broadcastable with pre_shape, or throws an error if it cannot be broadcasted.

        Args:
            pre_shape (Iterable[int]): The desired shape to broadcast the tensor to.

        Raises:
            ValueError: If the tensor cannot be broadcasted to the desired shape.

        Returns:
            TensorWrapperSubclass: The tensor with the desired shape.
        """

        pre_shape = list(pre_shape)

        target_shape = torch.Size(pre_shape) + \
            self.tensor.shape[-len(self.end_shape):]
        broadcast_shape = [1] * len(target_shape)

        for i in range(len(pre_shape)-1, -1, -1):
            if pre_shape[i] != target_shape[i]:
                if target_shape[i] == 1:
                    broadcast_shape[i] = self.tensor.shape[-len(
                        self.end_shape)+i]
                else:
                    raise ValueError(
                        f"Cannot broadcast tensor of shape {self.pre_shape()} to shape {pre_shape}")

        new_tensor = self.tensor.view(
            self.pre_shape() + [1] * len(self.end_shape))
        new_tensor = new_tensor.expand(self.pre_shape() + broadcast_shape)

        try:
            new_tensor = new_tensor.reshape(target_shape)
        except RuntimeError:
            raise ValueError(
                f"Cannot broadcast tensor of shape {self.pre_shape()} to shape {pre_shape}")

        return self.__class__(new_tensor)

    @staticmethod
    def stack(tensors: Iterable[TensorWrapperSubclass], dim: int = 0) -> TensorWrapperSubclass:
        """Stacks a list of tensors into a single tensor. This is a static method, so it can be called on the class itself."""

        raw_tensors = [tensor.tensor for tensor in tensors]

        # Adjust dim so that it works given the end shape
        if dim < 0:
            dim = dim - len(next(iter(tensors)).end_shape)

        stacked_tensor = torch.stack(raw_tensors, dim=dim)

        return next(iter(tensors))[0].__class__(stacked_tensor)

    def view(self: TensorWrapperSubclass, shape: Iterable[int]) -> TensorWrapperSubclass:
        if not isinstance(shape, list):
            shape = list(shape)

        new_shape = shape + self.end_shape
        new_raw_tensor = self.tensor.view(new_shape)

        return self.__class__(new_raw_tensor)  # type: ignore

    def unsqueeze(self: TensorWrapperSubclass, idx: int) -> TensorWrapperSubclass:
        new_raw_tensor = self.tensor.unsqueeze(idx)

        return self.__class__(new_raw_tensor)

    def broadcast_to(self: TensorWrapperSubclass, pre_shape: Iterable[int]) -> TensorWrapperSubclass:
        end_shape = list(pre_shape) + self.end_shape

        return self.__class__(self.tensor.broadcast_to(end_shape))

    @staticmethod
    def get_broadcast_pre_shape(tensor_wrappers: Iterable[Union["TensorWrapper", torch.Tensor]]) -> List[int]:
        """This function broadcasts a list of tensors to the same shape.

        Args:
            tensor_wrappers (Iterable(TensorWrapperSubclass)): This is the list of tensors to broadcast

        Returns:
            List[int] This is a the new pre_shape
        """

        shapes = [tensor_wrapper.pre_shape()
                  if isinstance(tensor_wrapper, TensorWrapper)
                  else list(tensor_wrapper.shape)
                  for tensor_wrapper in tensor_wrappers]

        broadcasted_pre_shape = torch.broadcast_shapes(*shapes)

        return list(broadcasted_pre_shape)

    def __copy__(self: TensorWrapperSubclass) -> TensorWrapperSubclass:
        new_raw_tensor = self.tensor.clone()

        return self.__class__(new_raw_tensor)

    @staticmethod
    def _default_device():
        """Returns the default device for the current context."""

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WrappedScalar(TensorWrapper):

    def __init__(self, wrapped_tensor: torch.Tensor) -> None:
        self.initialize_base(wrapped_tensor, [])


class TensorIndexer(Generic[TensorWrapperSubclass]):
    def __init__(self, tensor_wrapper: TensorWrapperSubclass, idx_dict: Dict[str, int]):

        self.tensor_wrapper = tensor_wrapper
        self.idx_dict = idx_dict
        self.device = tensor_wrapper.device

    def __getitem__(self, key: Union[str, int, slice]) -> TensorWrapperSubclass:
        if isinstance(key, str):
            return self.tensor_wrapper[..., self.idx_dict[key]]
        elif isinstance(key, int):
            return self.tensor_wrapper[..., key]
        else:
            raise TypeError(
                f"Key must be either str or int, but is {type(key)}.")

    def __contains__(self, key: Union[str, int]) -> bool:
        if isinstance(key, str):
            return key in self.idx_dict
        elif isinstance(key, int):
            return key in self.idx_dict.values()
        else:
            raise TypeError(
                f"Key must be either str or int, but is {type(key)}.")

    def get_idx_names(self) -> List[str]:
        names_and_idxs = [(idx_name, idx)
                          for idx_name, idx in self.idx_dict.items()]

        return [idx_name for idx_name, _
                in sorted(names_and_idxs, key=lambda x: x[1])]

    def get_idx(self, key: str) -> int:
        return self.idx_dict[key]

    def num_idxs(self) -> int:
        return len(self.idx_dict)

    def to_tensor(self, idx_order: Iterable[str]) -> TensorWrapperSubclass:

        ordered_tensors = [self[idx_name] for idx_name in idx_order]

        return TensorWrapper.stack(ordered_tensors)

    def reordered(self: "TensorIndexer[TensorWrapperSubclass]", idx_names: List[str]) -> "TensorIndexer[TensorWrapperSubclass]":
        """Returns a TensorIndexer made of reordered slices of this one"""

        self_names = self.get_idx_names()

        if self_names == idx_names:
            return self

        idx_dict = {idx_name: self[idx_name]for idx_name in idx_names}
        return TensorIndexer.from_dict(idx_dict)

    TensorWrapperSubclassApplyTarget = TypeVar(
        "TensorWrapperSubclassApplyTarget", bound=TensorWrapper)

    def apply(self, func: Callable[..., TensorWrapperSubclassApplyTarget], *args) -> "TensorIndexer[TensorWrapperSubclassApplyTarget]":
        return TensorIndexer(func(self.tensor_wrapper, *args), self.idx_dict)

    def to_dict(self) -> Dict[str, TensorWrapperSubclass]:
        return {idx_name: self[idx_name] for idx_name in self.get_idx_names()}

    def to_raw_dict(self) -> Dict[str, torch.Tensor]:
        return {idx_name: tensor_wrapper.tensor for idx_name, tensor_wrapper in self.to_dict().items()}

    @property
    def pre_shape(self) -> List[int]:
        return self.tensor_wrapper.pre_shape()[:-1]

    @staticmethod
    def from_dict(tensor_dict: Dict[str, TensorWrapperSubclass]) -> "TensorIndexer[TensorWrapperSubclass]":

        # Extract specific values type from tensor_dict
        tensor_type = list(tensor_dict.values())[0].__class__

        idx_dict = {idx_name: idx for idx, idx_name in enumerate(
                    tensor_dict)}

        pre_shapes = [tensor_wrapper.pre_shape()
                      for tensor_wrapper in tensor_dict.values()]

        target_shape = torch.broadcast_shapes(*pre_shapes)

        broadcasted_tensors = [tensor_wrapper.broadcast_to(target_shape)
                               for tensor_wrapper in tensor_dict.values()]

        stacked_tensor = tensor_type.stack(broadcasted_tensors)

        return TensorIndexer(stacked_tensor, idx_dict)
