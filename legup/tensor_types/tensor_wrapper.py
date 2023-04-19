import torch
from abc import ABC, abstractmethod
from typing import Iterable, List, Union, Type, Optional, TypeVar


TensorWrapperDerived = TypeVar(
    "TensorWrapperDerived", bound="TensorWrapper")


class TensorWrapper(ABC):

    @abstractmethod
    def __init__(self, wrapped_tensor: torch.Tensor) -> None:
        pass

    def initialize_base(self, tensor: torch.Tensor, end_shape: List[int]):
        self.tensor = tensor
        self.end_shape = end_shape
        self.device = tensor.device

    def __getitem__(self: TensorWrapperDerived, index) -> TensorWrapperDerived:

        slices = tuple(index) + (slice(None),) * len(self.end_shape)

        raw_slice_tensor = self.tensor[slices]

        return self.__class__(raw_slice_tensor)

    def __setitem__(self, index, value):
        self.tensor[index] = value

    def pre_shape(self) -> List[int]:
        return list(self.tensor.shape[: -len(self.end_shape)])

    def reshape(self, pre_shape: List[int]):
        return self.tensor.reshape(pre_shape + self.end_shape)

    def to(self: TensorWrapperDerived, device: torch.device) -> TensorWrapperDerived:
        """Moves the tensor to a device. This modifies the tensor, so other references to the tensor will be on the new device as well."""
        self.tensor = self.tensor.to(device)
        self.device = device

        return self

    @staticmethod
    def stack(tensors: Iterable[TensorWrapperDerived], dim: int = 0) -> TensorWrapperDerived:
        """Stacks a list of tensors into a single tensor. This is a static method, so it can be called on the class itself."""

        raw_tensors = [tensor.tensor for tensor in tensors]

        wrapper_class = next(iter(tensors)).__class__

        # Adjust dim so that it works given the end shape
        if dim < 0:
            dim = dim - len(next(iter(tensors)).end_shape)

        stacked_tensor = torch.stack(raw_tensors, dim=dim)

        return wrapper_class(stacked_tensor)

    def view(self: TensorWrapperDerived, shape: Iterable[int]) -> TensorWrapperDerived:
        if not isinstance(shape, list):
            shape = list(shape)

        new_shape = shape + self.end_shape
        new_raw_tensor = self.tensor.view(new_shape)

        return self.__class__(new_raw_tensor)  # type: ignore

    def unsqueeze(self: TensorWrapperDerived, idx: int) -> TensorWrapperDerived:

        if idx < 0:
            idx = idx - len(self.end_shape)

        new_raw_tensor = self.tensor.unsqueeze(idx)

        return self.__class__(new_raw_tensor)

    def broadcast_to(self: TensorWrapperDerived, pre_shape: Iterable[int]) -> TensorWrapperDerived:
        full_shape = list(pre_shape) + self.end_shape

        return self.__class__(self.tensor.broadcast_to(full_shape))

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

    def __copy__(self: TensorWrapperDerived) -> TensorWrapperDerived:
        new_raw_tensor = self.tensor.clone()

        return self.__class__(new_raw_tensor)

    @staticmethod
    def _default_device():
        """Returns the default device for the current context."""

        # type: ignore
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def make_wrapper_tensor_from_list(
            TensorClass: Type[TensorWrapperDerived],
            list: Union[List[float], List[int]],
            device: Optional[torch.device] = None
    ) -> TensorWrapperDerived:

        if device is None:
            device = TensorClass._default_device()

        raw_float_tensor = torch.tensor(list, dtype=torch.float, device=device)

        return TensorClass(raw_float_tensor)


class WrappedScalar(TensorWrapper):

    def __init__(self, wrapped_tensor: torch.Tensor) -> None:
        self.initialize_base(wrapped_tensor, [])
