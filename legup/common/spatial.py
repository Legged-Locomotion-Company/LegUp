from legup.common.tensor_wrapper import TensorWrapper

from typing import Optional, Sequence

import torch

"""
Raw tensor functions
"""


@torch.jit.script  # type: ignore
def _raw_twist_skew(twist_tensor: torch.Tensor):
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


@ torch.jit.script  # type: ignore
def _raw_screw_skew(screw_tensor: torch.Tensor):
    """Converts the raw screw tensor to a skew matrix tensor"""

    screw_rotation = screw_tensor[..., :3]
    screw_translation = screw_tensor[..., 3:]

    pre_shape = screw_tensor.shape[:-1]
    out_shape = list(pre_shape) + [4, 4]

    rotation_skew = _raw_twist_skew(screw_rotation)

    result = torch.zeros(out_shape, device=screw_tensor.device)

    result[..., :3, :3] = rotation_skew
    result[..., :3, 3] = screw_translation

    return result


@ torch.jit.script  # type: ignore
def _raw_twist_unskew(twist_skew_tensor: torch.Tensor) -> torch.Tensor:
    """Computes the unskew vector of a twist skew matrix."""

    # Construct the unskew vector.
    twist_unskew_tensor = torch.zeros(
        list(twist_skew_tensor.shape[:-2]) + [3], device=twist_skew_tensor.device)

    twist_unskew_tensor[..., 0] = twist_skew_tensor[..., 2, 1]
    twist_unskew_tensor[..., 1] = twist_skew_tensor[..., 0, 2]
    twist_unskew_tensor[..., 2] = twist_skew_tensor[..., 1, 0]

    return twist_unskew_tensor


@torch.jit.script  # type: ignore
def _raw_rotation_log_map(rotation_tensor: torch.Tensor):
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


@ torch.jit.script  # type: ignore
def _raw_normalize_tensor(in_tensor: torch.Tensor, dim: int):
    """Normalizes a tensor along a given dimension.

    Args:
        in_tensor: The tensor to normalize.
        dim: The dimension to normalize along.
    """

    norms = torch.norm(in_tensor, dim=dim, keepdim=True)

    return in_tensor / norms


@ torch.jit.script  # type: ignore
def _raw_twist_skew_exp_map(twist_skew_tensor: torch.Tensor) -> torch.Tensor:
    """Computes the exponential map of a twist skew matrix.
        This is an implementation of equation 3.51 from Modern Robotics by Kevin Lynch."""

    # Get omega * theta from [omega] * theta
    omega_theta = _raw_twist_unskew(twist_skew_tensor)

    # Since ||omega|| = 1. ||omega * theta|| = theta
    theta = torch.norm(omega_theta, dim=-1)

    # divide [omega] * theta by theta to get [omega]
    omega_skew = \
        torch.einsum('...,...ij->...ij', 1/theta, twist_skew_tensor)

    omega_skew_squared = torch.matmul(
        omega_skew, omega_skew)

    # Compute term2 sin(theta) * [omega_hat]
    term2 = torch.einsum('...,...ij->...ij', torch.sin(theta), omega_skew)

    # Compute term3 (1 - cos(theta)) * [omega_hat]^2
    term3 = torch.einsum('...,...ij->...ij',
                         (1 - torch.cos(theta)), omega_skew_squared)

    exponential_map = \
        torch.eye(3, 3, device=twist_skew_tensor.device) + term2 + term3

    return exponential_map


@ torch.jit.script  # type: ignore
def _raw_screw_skew_exp_map(screw_skew_tensor: torch.Tensor):
    """Converts the raw screw skew tensor to a log tensor
    Implementation of equation 3.88 from Modern Robotics by Kevin Lynch"""

    # create the result tensor
    pre_shape = screw_skew_tensor.shape[:-2]
    out_shape = list(pre_shape) + [4, 4]

    # Create a tensor to hold the result
    result = torch.zeros(out_shape, device=screw_skew_tensor.device)

    result[..., 3, 3] = 1

    # Extract the rotation skew matrix and translation vector
    omega_skew_theta = screw_skew_tensor[..., :3, :3]
    v_theta = screw_skew_tensor[..., :3, 3]

    # Create a mask for where ||omega|| > 0 and ||omega|| == 0
    omega_theta = _raw_twist_unskew(omega_skew_theta)

    # Since either omega == 1 or omega == 0, if omega * theta > 0, then omega == 1
    omega_norm_theta = torch.norm(omega_theta, dim=-1)

    omega_norm_nonzero_mask = omega_norm_theta > 0
    omega_norm_zero_mask = omega_norm_theta == 0

    # Compute the exponential map for case where ||omega|| > 0
    if omega_norm_nonzero_mask.any():

        # Get the values for the relevant indices
        masked_omega_skew_theta = omega_skew_theta[omega_norm_nonzero_mask]
        masked_v_theta = v_theta[omega_norm_nonzero_mask]

        # Compute the rotation matrix for e^([omega] * theta)
        masked_rotation_matrix = _raw_twist_skew_exp_map(
            masked_omega_skew_theta)

        # Since we know that in all of these indeces ||omega|| == 1, we can just
        # say that theta = ||omega|| * theta
        masked_theta = omega_norm_theta[omega_norm_nonzero_mask]

        # Now we divide [omega] * theta by theta to get [omega]
        masked_omega_skew = torch.einsum('...,...ij->...ij',
                                         1 / masked_theta,
                                         masked_omega_skew_theta)

        # Now we divide v * theta by theta to get v
        masked_translation_axis = torch.einsum('...,...i->...i',
                                               1 / masked_theta,
                                               masked_v_theta)

        masked_omega_skew_square = torch.matmul(
            masked_omega_skew, masked_omega_skew)

        term1 = torch.einsum('...B,ij->...Bij',
                             masked_theta,
                             torch.eye(3, device=screw_skew_tensor.device))

        term2 = torch.einsum('...,...ij->...ij',
                             (1 - torch.cos(masked_theta)),
                             masked_omega_skew)

        term3 = torch.einsum('...,...ij->...ij',
                             (masked_theta - torch.sin(masked_theta)),
                             masked_omega_skew_square)

        masked_translation = torch.einsum(
            '...ij,...j->...i', (term1 + term2 + term3), masked_translation_axis)

        result[..., :3, :3][omega_norm_nonzero_mask] = masked_rotation_matrix
        result[..., :3, 3][omega_norm_nonzero_mask] = masked_translation

    # Compute the exponential map for case where ||omega|| == 0
    if omega_norm_zero_mask.any():
        result[omega_norm_zero_mask][..., :3, :3] = torch.eye(
            3, device=screw_skew_tensor.device)
        result[omega_norm_zero_mask][..., :3,
                                     3] = v_theta[omega_norm_zero_mask]

    return result


"""
Spatial classes
"""


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

    @ staticmethod
    def empty_rotation(*shape, device=None):
        if device is None:
            device = Rotation._default_device()

        rotation_tensor = torch.empty((*shape, 3, 3), device=device)
        return Rotation(rotation_tensor)

    def compose(self, other: "Rotation") -> "Rotation":
        """Composes two rotations."""

        return Rotation(self.tensor @ other.tensor)

    def inverse(self) -> "Rotation":
        """Computes the inverse of a rotation."""

        return Rotation(self.tensor.transpose(-1, -2))


class Twist(TensorWrapper):
    def __init__(self, twist_tensor: torch.Tensor):
        if twist_tensor.shape[-1] != 3:
            raise ValueError("Twist vector must be of shape (3,).")
        super().__init__(twist_tensor, end_shape=[3])

    def skew(self) -> "TwistSkew":
        """Computes the skew matrix of a twist."""

        skew_matrix_tensor = _raw_twist_skew(self.tensor)

        return TwistSkew(skew_matrix_tensor)

    def exp_map(self) -> "Rotation":
        """Computes the exponential map of a twist."""

        # Compute the skew matrix of the twist.
        skew = self.skew()

        # Compute the exponential map.
        exp_map = skew.exp_map()

        return exp_map

    @ staticmethod
    def rand(*shape, device=None):
        if device is None:
            device = Twist._default_device()

        twist_tensor = torch.rand((*shape, 3), device=device)
        return Twist(twist_tensor)


class TwistSkew(TensorWrapper):
    def __init__(self, twist_skew_tensor: torch.Tensor):
        if twist_skew_tensor.shape[-2:] != (3, 3):
            raise ValueError("Twist skew matrix must be of shape (3, 3).")
        super().__init__(twist_skew_tensor, end_shape=[3, 3])

    def unskew(self) -> Twist:
        """Computes the unskew vector of a twist skew matrix."""

        # Construct the unskew vector.
        unskew_vec = _raw_twist_unskew(self.tensor)

        return Twist(unskew_vec)

    @staticmethod
    def rand(shape: Sequence[int], device=None):
        if device is None:
            device = TwistSkew._default_device()

        random_twist = Twist.rand(*shape, device=device)
        twist_skew = random_twist.skew()
        return twist_skew

    def exp_map(self) -> "Rotation":
        """Computes the exponential map of a twist skew matrix."""

        # Compute the exponential map.
        raw_tensor_exponential_map = _raw_twist_skew_exp_map(
            self.tensor)

        exponential_map = Rotation(raw_tensor_exponential_map)

        return exponential_map


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
        if screw_vec_tensor.shape[-1] != 6:
            raise ValueError("Screw vector must be of shape (6,).")
        super().__init__(screw_vec_tensor, end_shape=[6])

    @ staticmethod
    def empty_screw(shape: Sequence[int], device: Optional[torch.device] = None) -> "Screw":
        """Creates an empty screw.

        Args:
            shape (tuple): Shape of the screw.
            device (torch.Device, optional): The device on which to create the empty screw. Defaults to None.

        Returns:
            Screw: an empty screw
        """

        shape = list(shape)

        if device is None:
            device = Screw._default_device()

        screw_tensor = torch.empty(list((*shape, 6)), device=device)
        return Screw(screw_tensor)

    @ staticmethod
    def rand(*shape, device: Optional[torch.device] = None) -> "Screw":
        """Creates a random screw.

        Args:
            shape (tuple): Shape of the screw.
            device (torch.Device, optional): The device on which to create the random screw. Defaults to None.

        Returns:
            Screw: a random screw
        """

        if device is None:
            device = Screw._default_device()

        screw_tensor = torch.rand(list((*shape, 6)), device=device)
        return Screw(screw_tensor)

    def exp_map(self) -> Transform:
        skew = self.skew()
        transform = skew.exp_map()
        return transform

    def skew(self) -> "ScrewSkew":
        """Computes the skew matrix of a screw."""

        skew_matrix_tensor = _raw_screw_skew(self.tensor)

        return ScrewSkew(skew_matrix_tensor)


class ScrewSkew(TensorWrapper):
    def __init__(self, screw_skew_tensor: torch.Tensor):
        if screw_skew_tensor.shape[-2:] != (4, 4):
            raise ValueError("Screw skew matrix must be of shape (4, 4).")
        super().__init__(screw_skew_tensor, end_shape=[4, 4])

    def exp_map(self) -> Transform:
        """Computes the log map of a screw skew matrix which is a transform."""

        transform_tensor = _raw_screw_skew_exp_map(self.tensor)

        return Transform(transform_tensor)
