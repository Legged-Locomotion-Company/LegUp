from typing import Iterable, List, TypeVar, Dict, Union, Generic, get_args, Callable

from .tensor_wrapper import TensorWrapper

import torch

TensorWrapperDerived = TypeVar(
    "TensorWrapperDerived", bound="TensorWrapper")


class TensorIndexer(Generic[TensorWrapperDerived]):
    def __init__(self, tensor_wrapper: TensorWrapperDerived, idx_dict: Dict[str, int]):

        self.tensor_wrapper = tensor_wrapper
        self.idx_dict = idx_dict
        self.device = tensor_wrapper.device

    def __getitem__(self, key: Union[str, int, slice]) -> TensorWrapperDerived:
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

    def to_tensor(self, idx_order: Iterable[str]) -> TensorWrapperDerived:

        ordered_tensors = [self[idx_name] for idx_name in idx_order]

        return TensorWrapper.stack(ordered_tensors)

    def reordered(self: "TensorIndexer[TensorWrapperDerived]", idx_names: List[str]) -> "TensorIndexer[TensorWrapperDerived]":
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

    def to_dict(self) -> Dict[str, TensorWrapperDerived]:
        return {idx_name: self[idx_name] for idx_name in self.get_idx_names()}

    def to_raw_dict(self) -> Dict[str, torch.Tensor]:
        return {idx_name: tensor_wrapper.tensor for idx_name, tensor_wrapper in self.to_dict().items()}

    @property
    def pre_shape(self) -> List[int]:
        return self.tensor_wrapper.pre_shape()[:-1]

    @staticmethod
    def from_dict(tensor_dict: Dict[str, TensorWrapperDerived]) -> "TensorIndexer[TensorWrapperDerived]":

        # Extract specific values type from tensor_dict
        tensor_type = list(tensor_dict.values())[0].__class__

        idx_dict = \
            {idx_name: idx for idx, idx_name in enumerate(tensor_dict)}

        pre_shapes = [tensor_wrapper.pre_shape()
                      for tensor_wrapper in tensor_dict.values()]

        target_shape = torch.broadcast_shapes(*pre_shapes)

        broadcasted_tensors = [tensor_wrapper.broadcast_to(target_shape)
                               for tensor_wrapper in tensor_dict.values()]

        stacked_tensor = tensor_type.stack(broadcasted_tensors)

        return TensorIndexer(stacked_tensor, idx_dict)
