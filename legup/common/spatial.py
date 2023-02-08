from tensor_wrapper import TensorWrapper

from typing import Optional

import torch


class Position(TensorWrapper):
    def __init__(self, position_tensor: torch.Tensor):
        if position_tensor.shape[-1] != (3,):
            raise ValueError("Position vector must be of shape (3,).")
        super().__init__(position_tensor, end_dims=1)


class Transform(TensorWrapper):
    def __init__(self, transform_tensor: torch.Tensor):
        if transform_tensor.shape[-2:] != (4, 4):
            raise ValueError("Transform matrix must be of shape (4, 4).")
        super().__init__(transform_tensor, end_dims=2)

    def compose(self, other: "Transform", out: Optional["Transform"]) -> "Transform":
        """Composes two transforms.
        """

        if out is None:
            out_shape = torch.broadcast_shapes(
                self.tensor.shape, other.tensor.shape)
            out = Transform(torch.empty(out_shape, device=self.device))

        torch.matmul(self.tensor, other.tensor, out=out.tensor)

        return out


class Direction(TensorWrapper):
    def __init__(self, direction_vec_tensor: torch.Tensor):
        if direction_vec_tensor.shape[-1] != (3,):
            raise ValueError("Direction vector must be of shape (3,).")

        super().__init__(direction_vec_tensor, end_dims=1)


class Screw(TensorWrapper):
    def __init__(self, screw_vec_tensor: torch.Tensor):
        if screw_vec_tensor.shape[-1] != (6,):
            raise ValueError("Screw vector must be of shape (6,).")
        super().__init__(screw_vec_tensor, end_dims=1)

    @staticmethod
    def empty_screw(*shape, device):
        screw_tensor = torch.empty((*shape, 6), device=device)
        return Screw(screw_tensor)


"""
Raw Torch Operations
"""


@torch.jit.script
def normalize_tensor(in_tensor: torch.Tensor, dim: int):
    """Normalizes a tensor along a given dimension.

    Args:
        in_tensor: The tensor to normalize.
        dim: The dimension to normalize along.
    """

    norms = torch.norm(in_tensor, dim=dim, keepdim=True)

    return in_tensor / norms
