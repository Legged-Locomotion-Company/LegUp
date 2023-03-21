from legup.common.tensor_types import TensorWrapper

from typing import Optional, Sequence, List, Union, Iterable

from . import raw_spatial_methods

import torch


class Position(TensorWrapper):
    def __init__(self, position_tensor: torch.Tensor):
        if position_tensor.shape[-1] != 3:
            raise ValueError(
                f"Last dim of position_tensor must be 3 not {position_tensor.shape[-1]}.")
        self.initialize_base(position_tensor, end_shape=[3])

    def compose(self, other: "Position") -> "Position":
        """Composes two positions."""

        return Position(self.tensor + other.tensor)

    def make_transform(self):
        """This function creates a pure translation Transform from this position

        Returns:
            Transform: A transform that translates to this position
        """

        result_transform = \
            Transform.zero(*self.pre_shape(), device=self.device)

        result_transform.tensor[..., :3, 3] = self.tensor

        return result_transform

    def __add__(self, other: Union["Position", Union[Iterable[float], Iterable[int]]]) -> "Position":
        if not isinstance(other, Position):
            other = Position.from_iter(list(other), device=self.device)
        return self.compose(other)

    def __sub__(self, other: Union["Position", Union[Iterable[float], Iterable[int]]]) -> "Position":
        if not isinstance(other, Position):
            other = Position.from_iter(list(other), device=self.device)
        return self + (-other)

    def __neg__(self) -> "Position":
        return Position(-self.tensor)

    def __mul__(self, other: Union[float, int]) -> "Position":
        return Position(self.tensor * other)

    def norm(self) -> torch.Tensor:
        return torch.norm(self.tensor, dim=-1)

    @staticmethod
    def from_iter(iterable:  Union[Iterable[float], Iterable[int]], device: Optional[torch.device] = None):
        """This function creates a position from a list of numbers

        Args:
            list (Union[Iterable[float], Iterable[int]]): This is a list of numbers that will be assigned into this Position
            device (Optional[torch.device], optional): This defines the device for the new position. Defaults to None.

        Returns:
            Position: The created position object
        """

        return TensorWrapper.make_wrapper_tensor_from_list(Position, list=list(iterable), device=device)


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

        skew_matrix_tensor = raw_spatial_methods.twist_skew(self.tensor)

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
        unskew_vec = raw_spatial_methods.twist_unskew(self.tensor)

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
        raw_tensor_exponential_map = raw_spatial_methods.twist_skew_exp_map(
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

        return TransformAdjoint(raw_spatial_methods.transform_adjoint(self.tensor))

    @staticmethod
    def zero(*shape: int, device: Optional[torch.device] = None):
        if device is None:
            device = Transform._default_device()

        transform_tensor = torch.eye(4, device=device).repeat(*shape, 1, 1)
        return Transform(transform_tensor)

    @staticmethod
    def from_rotation_translation(rotation: Rotation, translation: Position) -> "Transform":
        """Constructs a transform from a rotation and translation."""

        if (translation_pre_shape := translation.pre_shape()) != (rotation_pre_shape := rotation.pre_shape()):
            raise ValueError(
                f"Rotation and translation must have the same pre-shape. Got {rotation_pre_shape} and {translation_pre_shape}.")

        return Transform(raw_spatial_methods
                         .transform_from_rotation_translation(rotation.tensor, translation.tensor))

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

        return Transform(raw_spatial_methods.transform_invert(self.tensor))


class Direction(TensorWrapper):
    def __init__(self, direction_vec_tensor: torch.Tensor):
        if direction_vec_tensor.shape[-1] != 3:
            raise ValueError(
                f"Last dim of direction_vec_tensor must be 3 not {direction_vec_tensor.shape[-1]}.")

        self.initialize_base(direction_vec_tensor, end_shape=[3])

    @ staticmethod
    def from_list(list: Union[List[float], List[int]], device: Optional[torch.device] = None):
        return TensorWrapper.make_wrapper_tensor_from_list(Direction, list=list, device=device)


class Screw(TensorWrapper):
    def __init__(self, screw_vec_tensor: torch.Tensor):
        if screw_vec_tensor.shape[-1] != 6:
            raise ValueError("Screw vector must be of shape (6,).")
        self.initialize_base(screw_vec_tensor, end_shape=[6])

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

        screw_tensor = raw_spatial_methods.screw_from_axis_origin(
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

        skew_matrix_tensor = raw_spatial_methods.screw_skew(self.tensor)

        return ScrewSkew(skew_matrix_tensor)

    def __mul__(self, factor: torch.Tensor) -> "Screw":
        """Multiplies a screw by a tensor."""

        broadcast_shape = TensorWrapper.get_broadcast_pre_shape([self, factor])

        self_broadcast = self.broadcast_to(broadcast_shape)
        factor_broadcast = factor.broadcast_to(broadcast_shape)

        # Multiply
        multiplied_screw_tensor = torch.einsum("...i,...->...i",
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

        transform_tensor = raw_spatial_methods.screw_skew_exp_map(self.tensor)

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
