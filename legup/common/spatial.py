from legup.common.tensor_types import TensorWrapper

from typing import Optional, Sequence, List

import torch

"""
Raw tensor functions
"""


@torch.jit.script  # type: ignore
class RawSpatialMethods:

    @staticmethod
    def screw_from_axis_origin(axis: torch.Tensor, origin: torch.Tensor):
        """Constructs a raw screw tensor from an axis and origin."""

        axis = axis.to(dtype=torch.float)
        origin = origin.to(dtype=torch.float)

        pre_shape = axis.shape[:-1]
        screw = torch.empty(pre_shape + (6,), device=axis.device)

        screw[..., :3] = axis
        screw[..., 3:] = torch.cross(-axis, origin)

        return screw

    @staticmethod
    def twist_skew(twist_tensor: torch.Tensor):
        """Converts the raw twist tensor to a skew matrix tensor"""

        twist_tensor = twist_tensor.to(dtype=torch.float)

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

    @staticmethod
    def screw_skew(screw_tensor: torch.Tensor):
        """Converts the raw screw tensor to a skew matrix tensor"""

        screw_tensor = screw_tensor.to(dtype=torch.float)

        screw_rotation = screw_tensor[..., :3]
        screw_translation = screw_tensor[..., 3:]

        pre_shape = screw_tensor.shape[:-1]
        out_shape = list(pre_shape) + [4, 4]

        rotation_skew = RawSpatialMethods.twist_skew(screw_rotation)

        result = torch.zeros(out_shape, device=screw_tensor.device)

        result[..., :3, :3] = rotation_skew
        result[..., :3, 3] = screw_translation

        return result

    @staticmethod
    def twist_unskew(twist_skew_tensor: torch.Tensor) -> torch.Tensor:
        """Computes the unskew vector of a twist skew matrix."""

        twist_skew_tensor = twist_skew_tensor.to(dtype=torch.float)

        # Construct the unskew vector.
        twist_unskew_tensor = torch.zeros(
            list(twist_skew_tensor.shape[:-2]) + [3], device=twist_skew_tensor.device)

        twist_unskew_tensor[..., 0] = twist_skew_tensor[..., 2, 1]
        twist_unskew_tensor[..., 1] = twist_skew_tensor[..., 0, 2]
        twist_unskew_tensor[..., 2] = twist_skew_tensor[..., 1, 0]

        return twist_unskew_tensor

    @staticmethod
    def rotation_log_map(rotation_tensor: torch.Tensor):
        """Computes the log map of a rotation."""

        rotation_tensor = rotation_tensor.to(dtype=torch.float)

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

    @staticmethod
    def normalize_tensor(in_tensor: torch.Tensor, dim: int):
        """Normalizes a tensor along a given dimension.

        Args:
            in_tensor: The tensor to normalize.
            dim: The dimension to normalize along.
        """

        in_tensor = in_tensor.to(dtype=torch.float)

        norms = torch.norm(in_tensor, dim=dim, keepdim=True)

        return in_tensor / norms

    @staticmethod
    def twist_skew_exp_map(twist_skew_tensor: torch.Tensor) -> torch.Tensor:
        """Computes the exponential map of a twist skew matrix.
            This is an implementation of equation 3.51 from Modern Robotics by Kevin Lynch."""

        twist_skew_tensor = twist_skew_tensor.to(dtype=torch.float)

        # Get omega * theta from [omega] * theta
        omega_theta = RawSpatialMethods.twist_unskew(twist_skew_tensor)

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

    @staticmethod
    def screw_skew_exp_map(screw_skew_tensor: torch.Tensor):
        """Converts the raw screw skew tensor to a log tensor
        Implementation of equation 3.88 from Modern Robotics by Kevin Lynch"""

        screw_skew_tensor = screw_skew_tensor.to(dtype=torch.float)

        # create the result tensor
        pre_shape = screw_skew_tensor.shape[:-2]
        out_shape = list(pre_shape) + [4, 4]

        # Create a tensor to hold the result
        result = torch.zeros(
            out_shape, device=screw_skew_tensor.device, dtype=torch.float)

        result[..., 3, 3] = 1

        # Extract the rotation skew matrix and translation vector
        omega_skew_theta = screw_skew_tensor[..., :3, :3]
        v_theta = screw_skew_tensor[..., :3, 3]

        # Create a mask for where ||omega|| > 0 and ||omega|| == 0
        omega_theta = RawSpatialMethods.twist_unskew(omega_skew_theta)

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
            masked_rotation_matrix = RawSpatialMethods.twist_skew_exp_map(
                masked_omega_skew_theta)

            # Since we know that in all of these indices ||omega|| == 1, we can just
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

    @staticmethod
    def transform_compose(transform_tensors: List[torch.Tensor]):
        """Composes a list of transforms together"""

        return torch.chain_matmul(*transform_tensors)

    @staticmethod
    def transform_invert(transform_tensor: torch.Tensor):
        """This function computes the inverse of a transform efficiently"""

        # Create a tensor to hold the result
        result = torch.zeros_like(transform_tensor)

        # Extract the rotation matrix and translation vector
        rotation_matrix = transform_tensor[..., :3, :3]
        translation_vector = transform_tensor[..., :3, 3]

        # Compute the inverse rotation matrix
        result[..., :3, :3] = torch.transpose(rotation_matrix, -1, -2)

        # Compute the inverse translation vector
        result[..., :3, 3] = torch.einsum('...ij,...j->...i',
                                          -result[..., :3, :3], translation_vector)

        result[..., 3, 3] = 1

        return result

    @staticmethod
    def transform_adjoint(transform_tensor: torch.Tensor):
        """This function computes the adjoint of a transform
        This is an implementation of Definition 3.20 in Modern Robotics"""

        # Create a tensor to hold the result
        result = torch.zeros((*transform_tensor.shape, 6, 6),
                             device=transform_tensor.device)

        # Extract the rotation matrix and translation vector
        rotation_matrix = transform_tensor[..., :3, :3]
        translation_vector = transform_tensor[..., :3, 3]

        # Assign top left and bottom right to rotation matrix
        result[..., :3, :3] = rotation_matrix.unsqueeze(-4)
        result[..., 3:, 3:] = rotation_matrix.unsqueeze(-4)

        # Compute bottom left part [p]R
        result[..., 3:, :3] = torch.einsum('...ij,...jk->...ij',
                                           RawSpatialMethods.twist_skew(
                                               translation_vector),
                                           rotation_matrix)

        return result


"""
Spatial classes
"""


class Position(TensorWrapper):
    def __init__(self, position_tensor: torch.Tensor):
        if position_tensor.shape[-1] != 3:
            raise ValueError(
                f"Last dim of position_tensor must be 3 not {position_tensor.shape[-1]}.")
        self.initialize_base(position_tensor, end_shape=[3])

    def compose(self, other: "Position") -> "Position":
        """Composes two positions."""

        return Position(self.tensor + other.tensor)


class Rotation(TensorWrapper):
    def __init__(self, rotation_tensor: torch.Tensor):
        if rotation_tensor.shape[-2:] != (3, 3):
            raise ValueError("Rotation matrix must be of shape (3, 3).")
        self.initialize_base(rotation_tensor, end_shape=[3, 3])

    @staticmethod
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
        self.initialize_base(twist_tensor, end_shape=[3])

    def skew(self) -> "TwistSkew":
        """Computes the skew matrix of a twist."""

        skew_matrix_tensor = RawSpatialMethods.twist_skew(self.tensor)

        return TwistSkew(skew_matrix_tensor)

    def exp_map(self) -> "Rotation":
        """Computes the exponential map of a twist."""

        # Compute the skew matrix of the twist.
        skew = self.skew()

        # Compute the exponential map.
        exp_map = skew.exp_map()

        return exp_map

    @staticmethod
    def rand(*shape, device=None):
        if device is None:
            device = Twist._default_device()

        twist_tensor = torch.rand((*shape, 3), device=device)
        return Twist(twist_tensor)


class TwistSkew(TensorWrapper):
    def __init__(self, twist_skew_tensor: torch.Tensor):
        if twist_skew_tensor.shape[-2:] != (3, 3):
            raise ValueError("Twist skew matrix must be of shape (3, 3).")
        self.initialize_base(twist_skew_tensor, end_shape=[3, 3])

    def unskew(self) -> Twist:
        """Computes the unskew vector of a twist skew matrix."""

        # Construct the unskew vector.
        unskew_vec = RawSpatialMethods.twist_unskew(self.tensor)

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
        raw_tensor_exponential_map = RawSpatialMethods.twist_skew_exp_map(
            self.tensor)

        exponential_map = Rotation(raw_tensor_exponential_map)

        return exponential_map


class Transform(TensorWrapper):
    def __init__(self, transform_tensor: torch.Tensor):
        if transform_tensor.shape[-2:] != (4, 4):
            raise ValueError("Transform matrix must be of shape (4, 4).")
        self.initialize_base(transform_tensor, end_shape=[4, 4])

    @staticmethod
    def compose(*transforms: "Transform") -> "Transform":
        """Composes a list of transforms."""

        if len(transforms) == 0:
            raise ValueError("Must provide at least one transform.")

        for transform in transforms:
            if transform.tensor.shape[-2:] != (4, 4):
                raise ValueError("Transform matrix must be of shape (4, 4).")

        # broadcast the transforms
        broadcast_shape = TensorWrapper.get_broadcast_pre_shape(transforms)
        broadcasted_transforms = [transform.broadcast_to(broadcast_shape)
                                  for transform in transforms]

        raw_composed_tensor = torch.chain_matmul(broadcasted_transforms)
        return Transform(raw_composed_tensor)

    def adjoint(self) -> "TransformAdjoint":
        """Computes the adjoint of a transform."""

        return TransformAdjoint(RawSpatialMethods.transform_adjoint(self.tensor))

    @staticmethod
    def zero(*shape, device=None):
        if device is None:
            device = Transform._default_device()

        transform_tensor = torch.eye(4, device=device).repeat(*shape, 1, 1)
        return Transform(transform_tensor)

    def extract_translation(self) -> Position:
        """Extracts the translation component of a transform."""

        return Position(self.tensor[..., :3, 3])

    def log_map(self) -> "Screw":
        raise NotImplementedError("Log map not implemented.")

    def get_position(self) -> Position:
        return Position(self.tensor[..., :3, 3])

    def __mul__(self, other: "Transform") -> "Transform":
        """Composes two transforms."""

        return Transform(torch.matmul(self.tensor, other.tensor))

    def invert(self) -> "Transform":
        """Inverts a transform."""

        return Transform(RawSpatialMethods.transform_invert(self.tensor))


class Direction(TensorWrapper):
    def __init__(self, direction_vec_tensor: torch.Tensor):
        if direction_vec_tensor.shape[-1] != 3:
            raise ValueError(
                f"Last dim of direction_vec_tensor must be 3 not {direction_vec_tensor.shape[-1]}.")

        self.initialize_base(direction_vec_tensor, end_shape=[3])


class Screw(TensorWrapper):
    def __init__(self, screw_vec_tensor: torch.Tensor):
        if screw_vec_tensor.shape[-1] != 6:
            raise ValueError("Screw vector must be of shape (6,).")
        self.initialize_base(screw_vec_tensor, end_shape=[6])

    @staticmethod
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

    @staticmethod
    def from_axis_and_origin(axis: Direction, origin: Position, device: Optional[torch.device] = None) -> "Screw":
        """Creates a screw from an axis and an origin.

        Args:
            axis (Direction): The axis of the screw.
            origin (Position): The origin of the screw.
            device (torch.Device, optional): The device on which to create the screw. Defaults to None.

        Returns:
            Screw: a screw
        """

        if device is None:
            device = Screw._default_device()

        origin = origin.to(device)

        if axis.pre_shape() != origin.pre_shape():
            raise ValueError(
                f"Axis and origin must have the same pre-shape. Got {axis.pre_shape()} and {origin.pre_shape()} respectively.")

        screw_tensor = RawSpatialMethods.screw_from_axis_origin(
            axis.tensor.to(device), origin.tensor.to(device))

        return Screw(screw_tensor)

    @staticmethod
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

        skew_matrix_tensor = RawSpatialMethods.screw_skew(self.tensor)

        return ScrewSkew(skew_matrix_tensor)

    def __mul__(self, factor: torch.Tensor) -> "Screw":
        """Multiplies a screw by a tensor."""

        broadcast_shape = TensorWrapper.get_broadcast_pre_shape([self, factor])

        self_broadcast = self.broadcast_to(broadcast_shape)
        factor_broadcast = factor.broadcast_to(broadcast_shape)

        # Multiply
        multiplied_screw_tensor = torch.einsum("...ij,...->...ij",
                                               self_broadcast.tensor, factor_broadcast)

        return Screw(multiplied_screw_tensor)

    def apply(self, theta: torch.Tensor) -> "Transform":
        """Applies a rotation to a screw.

        Args:
            theta (torch.Tensor): The rotation angle.

        Returns:
            Screw: The rotated screw.
        """

        multiplied_screw = self * theta

        return multiplied_screw.exp_map()


class TransformAdjoint(TensorWrapper):
    def __init__(self, transform_adjoint_tensor: torch.Tensor):
        if transform_adjoint_tensor.shape[-2:] != (6, 6):
            raise ValueError(
                "Transform adjoint matrix must be of shape (6, 6).")
        self.initialize_base(transform_adjoint_tensor, end_shape=[6, 6])

    def __mul__(self, screw: Screw) -> Screw:
        """Composes two transform adjoints."""

        broadcast_shape = TensorWrapper.get_broadcast_pre_shape([self, screw])

        self_broadcast = self.broadcast_to(broadcast_shape)
        screw_broadcast = screw.broadcast_to(broadcast_shape)

        return Screw(torch.einsum("...ij,...j->...i",
                                  self_broadcast.tensor, screw_broadcast.tensor))


class ScrewSkew(TensorWrapper):
    def __init__(self, screw_skew_tensor: torch.Tensor):
        if screw_skew_tensor.shape[-2:] != (4, 4):
            raise ValueError("Screw skew matrix must be of shape (4, 4).")
        self.initialize_base(screw_skew_tensor, end_shape=[4, 4])

    def exp_map(self) -> Transform:
        """Computes the log map of a screw skew matrix which is a transform."""

        transform_tensor = RawSpatialMethods.screw_skew_exp_map(self.tensor)

        return Transform(transform_tensor)

    def apply(self, theta: torch.Tensor) -> "Transform":
        """Applies a rotation to a screw skew matrix.

        Args:
            theta (torch.Tensor): The rotation angle.

        Returns:
            Transform: The rotated screw skew matrix.
        """

        broadcast_shape = TensorWrapper.get_broadcast_pre_shape([self, theta])

        self_broadcast = self.broadcast_to(broadcast_shape)
        theta_broadcast = theta.broadcast_to(broadcast_shape)

        # Multiply
        multiplied_screw_skew_tensor = torch.einsum("...ij,...->...ij",
                                                    self_broadcast.tensor, theta_broadcast)

        return ScrewSkew(multiplied_screw_skew_tensor).exp_map()


class ScrewJacobian(TensorWrapper):
    def __init__(self, screw_jacobian_tensor: torch.Tensor):
        if screw_jacobian_tensor.shape[-2] != 6:
            raise ValueError("Screw jacobian must have second to last dim 6")

        self.num_dofs = screw_jacobian_tensor.shape[-1]

        self.initialize_base(screw_jacobian_tensor,
                             end_shape=[6, self.num_dofs])
