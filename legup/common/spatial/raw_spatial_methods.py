import torch

from typing import List


@torch.jit.script  # type: ignore
def screw_from_axis_origin(axis: torch.Tensor, origin: torch.Tensor):
    """Constructs a raw screw tensor from an axis and origin."""

    axis = axis.to(dtype=torch.float)
    origin = origin.to(dtype=torch.float)

    pre_shape = axis.shape[:-1]
    screw = torch.empty(pre_shape + (6,), device=axis.device)

    screw[..., :3] = axis
    screw[..., 3:] = torch.cross(-axis, origin)

    return screw


@torch.jit.script  # type: ignore
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


@torch.jit.script  # type: ignore
def screw_skew(screw_tensor: torch.Tensor):
    """Converts the raw screw tensor to a skew matrix tensor"""

    screw_tensor = screw_tensor.to(dtype=torch.float)

    screw_rotation = screw_tensor[..., :3]
    screw_translation = screw_tensor[..., 3:]

    pre_shape = screw_tensor.shape[:-1]
    out_shape = list(pre_shape) + [4, 4]

    rotation_skew = twist_skew(screw_rotation)

    result = torch.zeros(out_shape, device=screw_tensor.device)

    result[..., :3, :3] = rotation_skew
    result[..., :3, 3] = screw_translation

    return result


@torch.jit.script  # type: ignore
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


@torch.jit.script  # type: ignore
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


@torch.jit.script  # type: ignore
def normalize_tensor(in_tensor: torch.Tensor, dim: int):
    """Normalizes a tensor along a given dimension.

    Args:
        in_tensor: The tensor to normalize.
        dim: The dimension to normalize along.
    """

    in_tensor = in_tensor.to(dtype=torch.float)

    norms = torch.norm(in_tensor, dim=dim, keepdim=True)

    return in_tensor / norms


@torch.jit.script  # type: ignore
def twist_skew_exp_map(twist_skew_tensor: torch.Tensor) -> torch.Tensor:
    """Computes the exponential map of a twist skew matrix.
        This is an implementation of equation 3.51 from Modern Robotics by Kevin Lynch."""

    twist_skew_tensor = twist_skew_tensor.to(dtype=torch.float)

    # Get omega * theta from [omega] * theta
    omega_theta = twist_unskew(twist_skew_tensor)

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


@torch.jit.script  # type: ignore
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
    omega_theta = twist_unskew(omega_skew_theta)

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
        masked_rotation_matrix = twist_skew_exp_map(
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
        result[..., :3, :3][omega_norm_zero_mask] = \
            torch.eye(3, device=screw_skew_tensor.device)
        result[..., :3, 3][omega_norm_zero_mask] = \
            v_theta[omega_norm_zero_mask]

    return result


@torch.jit.script  # type: ignore
def transform_compose(transform_tensors: List[torch.Tensor]):
    """Composes a list of transforms together"""

    return torch.chain_matmul(transform_tensors)


@torch.jit.script  # type: ignore
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


@torch.jit.script  # type: ignore
def transform_adjoint(transform_tensor: torch.Tensor):
    """This function computes the adjoint of a transform
    This is an implementation of Definition 3.20 in Modern Robotics"""

    # Here we strip away the last 2 elements of transform_tensor.shape
    # which should be [4, 4] since it is a transform, and instead add
    # two new dims of [6, 6] to get the shape of an adjoint with
    # the same pre_shape
    result_shape = list(transform_tensor.shape[:-2]) + [6, 6]

    # Create a tensor to hold the result
    result = torch.zeros(result_shape, device=transform_tensor.device)

    # Extract the rotation matrix and translation vector
    rotation_matrix = transform_tensor[..., :3, :3]
    translation_vector = transform_tensor[..., :3, 3]

    # Assign top left and bottom right to rotation matrix
    result[..., :3, :3] = rotation_matrix
    result[..., 3:, 3:] = rotation_matrix

    translation_skew = twist_skew(translation_vector)

    # Compute bottom left part [p]R
    result[..., 3:, :3] = translation_skew @ rotation_matrix

    return result
