from legup.common.tensor_wrapper import TensorWrapper

from typing import Optional, List

import torch


class Position(TensorWrapper):
    def __init__(self, position_tensor: torch.Tensor):
        if position_tensor.shape[-1] != (3,):
            raise ValueError("Position vector must be of shape (3,).")
        super().__init__(position_tensor, end_shape=[3])

    def compose(self, other: "Position") -> "Position":
        """Composes two positions."""

        return Position(self.tensor + other.tensor)


class Rotation(TensorWrapper):
    def __init__(self, rotation_tensor: torch.Tensor):
        if rotation_tensor.shape[-2:] != (3, 3):
            raise ValueError("Rotation matrix must be of shape (3, 3).")
        super().__init__(rotation_tensor, end_shape=[3, 3])

    @staticmethod
    def empty_rotation(*shape, device):
        rotation_tensor = torch.empty((*shape, 3, 3), device=device)
        return Rotation(rotation_tensor)

    def compose(self, other: "Rotation") -> "Rotation":
        """Composes two rotations."""

        return Rotation(self.tensor @ other.tensor)

    def inverse(self) -> "Rotation":
        """Computes the inverse of a rotation."""

        return Rotation(self.tensor.transpose(-1, -2))

    @staticmethod
    @torch.jit.script  # type: ignore
    def _raw_log_map(rotation_tensor: torch.Tensor):
        """Computes the log map of a rotation."""

        # Compute the trace of the rotation matrix.
        trace = torch.trace(rotation_tensor)

        # Compute the angle of rotation.
        angle = torch.acos((trace - 1) / 2)

        # Compute the skew matrix of the rotation.
        skew = (rotation_tensor - rotation_tensor.transpose(-1, -2)) / 2

        # Compute the axis of rotation.
        axis = skew / torch.sin(angle)

        # Compute the log map.
        log_map = angle * axis

        return log_map


class Twist(TensorWrapper):
    def __init__(self, twist_tensor: torch.Tensor):
        if twist_tensor.shape[-1] != (3,):
            raise ValueError("Twist vector must be of shape (3,).")
        super().__init__(twist_tensor, end_shape=[3])

    def skew(self) -> "TwistSkew":
        """Computes the skew matrix of a twist."""

        skew_matrix_tensor = self._raw_tensor_skew()

        return TwistSkew(skew_matrix_tensor)

    @staticmethod
    @torch.jit.script  # type: ignore
    def _raw_tensor_skew(twist_tensor: torch.Tensor):
        """Converts the raw twist tensor to a skew matrix tensor"""

        # Construct the skew matrix.
        skew = torch.zeros(
            list(twist_tensor.shape[:-1]) + [3, 3], device=twist_tensor.device)
        skew[..., 0, 1] = -twist_tensor[..., 2]
        skew[..., 0, 2] = twist_tensor[..., 1]
        skew[..., 1, 0] = twist_tensor[..., 2]
        skew[..., 1, 2] = -twist_tensor[..., 0]
        skew[..., 2, 0] = -twist_tensor[..., 1]
        skew[..., 2, 1] = twist_tensor[..., 0]

        return skew


class TwistSkew(TensorWrapper):
    def __init__(self, twist_skew_tensor: torch.Tensor):
        if twist_skew_tensor.shape[-2:] != (3, 3):
            raise ValueError("Twist skew matrix must be of shape (3, 3).")
        super().__init__(twist_skew_tensor, end_shape=[3, 3])

    def unskew(self) -> Twist:
        """Computes the unskew vector of a twist skew matrix."""

        # Construct the unskew vector.
        unskew_vec = TwistSkew._raw_tensor_unskew(self.tensor)

        return Twist(unskew_vec)

    @staticmethod
    @torch.jit.script  # type: ignore
    def _raw_tensor_unskew(twist_skew_tensor: torch.Tensor) -> torch.Tensor:
        """Computes the unskew vector of a twist skew matrix."""

        # Construct the unskew vector.
        twist_unskew_tensor = torch.zeros(
            list(twist_skew_tensor.shape[:-2]) + [3], device=twist_skew_tensor.device)

        twist_unskew_tensor[..., 0] = twist_skew_tensor[..., 2, 1]
        twist_unskew_tensor[..., 1] = twist_skew_tensor[..., 0, 2]
        twist_unskew_tensor[..., 2] = twist_skew_tensor[..., 1, 0]

        return twist_unskew_tensor

    @torch.jit.script  # type: ignore
    def _raw_tensor_exp_map(twist_skew_tensor: torch.Tensor) -> torch.Tensor:
        """Computes the exponential map of a twist skew matrix."""

        # Extract the translation and rotation components of the twist skew matrix.
        translation = twist_skew_tensor[..., :3, 3]
        rotation = twist_skew_tensor[..., :3, :3]

        # Compute the rotation matrix.
        raise NotImplementedError(
            "Rotation matrix computation not implemented.")


class Transform(TensorWrapper):
    def __init__(self, transform_tensor: torch.Tensor):
        if transform_tensor.shape[-2:] != (4, 4):
            raise ValueError("Transform matrix must be of shape (4, 4).")
        super().__init__(transform_tensor, end_shape=[4, 4])

    def compose(self, other: "Transform", out: Optional["Transform"]) -> "Transform":
        """Composes two transforms."""

        if out is None:
            out_shape = torch.broadcast_shapes(
                self.tensor.shape, other.tensor.shape)
            out = Transform(torch.empty(out_shape, device=self.device))

        torch.matmul(self.tensor, other.tensor, out=out.tensor)

        return out

    def extract_translation(self) -> Position:
        """Extracts the translation component of a transform."""

        return Position(self.tensor[..., :3, 3])

    def log_map(self) -> "Screw":
        raise NotImplementedError("Log map not implemented.")


class Direction(TensorWrapper):
    def __init__(self, direction_vec_tensor: torch.Tensor):
        if direction_vec_tensor.shape[-1] != (3,):
            raise ValueError("Direction vector must be of shape (3,).")

        super().__init__(direction_vec_tensor, end_shape=[3])


class Screw(TensorWrapper):
    def __init__(self, screw_vec_tensor: torch.Tensor):
        if screw_vec_tensor.shape[-1] != (6,):
            raise ValueError("Screw vector must be of shape (6,).")
        super().__init__(screw_vec_tensor, end_shape=[6])

    def empty_screw(*shape, device):
        screw_tensor = torch.empty(list((*shape, 6)), device=device)
        return Screw(screw_tensor)

    def exp_map(self) -> Transform:
        raise NotImplementedError("Exponential map not implemented.")


"""
Raw Torch Operations
"""


@torch.jit.script  # type: ignore
def normalize_tensor(in_tensor: torch.Tensor, dim: int):
    """Normalizes a tensor along a given dimension.

    Args:
        in_tensor: The tensor to normalize.
        dim: The dimension to normalize along.
    """

    norms = torch.norm(in_tensor, dim=dim, keepdim=True)

    return in_tensor / norms
