import torch

from typing import NamedTuple, Callable, List, Optional, Union

from legup.spatial import Transform, Screw, Position
from . import FKFunction, FKResult

# Design ideas

# kinematics interface that essentially wraps a Callable that takes an error jacobian and returns a kinematics result


class IKFunction:
    def __init__(self, ik_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.ik_function = ik_function

    def __call__(self, error: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        return self.ik_function(error, jacobian)

    def apply(self, target: Union[Position, Screw, Transform],
              dof_angles: torch.Tensor, fk_func: FKFunction,
              max_abs_change: Union[torch.Tensor, float] = 0.25) -> torch.Tensor:

        if isinstance(max_abs_change, float):
            max_abs_change = \
                torch.tensor(max_abs_change, device=dof_angles.device)

        fk_result = fk_func(dof_angles)

        raw_jacobian = fk_result.jacobian.tensor
        raw_current = None
        raw_target = None

        if isinstance(target, Position):
            raw_jacobian = raw_jacobian[..., 3:, :]
            raw_current = fk_result.transform.extract_translation().tensor
            raw_target = target.tensor

        else:
            raw_current = fk_result.transform.log_map().tensor

        if isinstance(target, Transform):
            raw_target = target.log_map().tensor

        error = raw_target - raw_current

        deltas = self.ik_function(error, raw_jacobian)

        # Warning: This does not take into account multi-objective IK.
        # This will explode if you have multiple ee's on one dof
        accumulated_deltas = deltas.sum(dim=-2)

        return dof_angles + accumulated_deltas


@torch.jit.script  # type: ignore
def _raw_pinverse_ik_delta(error: torch.Tensor, j: torch.Tensor, factor: float = 0.25) -> torch.Tensor:
    """Executes the pseudo inverse algorithm on inputs (Do not use this alg it sucks, just here for testing)
    Args:
        error_jacobian (ErrorJacobian): the error and jacobian to to run inverse kinematics on
        factor (float, optional): a scalar weight on the result to try to reduce the instability inherent to this algorithm
    Returns:
        torch.Tensor: a (NUM_ENVS x 4 x 3) tensor with the command delta to be applied to joint from each jacobian
    """

    return torch.linalg.lstsq(j, error).solution * factor


pinverse_ik = IKFunction(_raw_pinverse_ik_delta)


@torch.jit.script  # type: ignore
def _raw_dls_ik_delta(error: torch.Tensor, j: torch.Tensor, lam: float = 0.05) -> torch.Tensor:

    # angle_shape = j.shape[:-2]
    # num_dofs = j.shape[-1]
    # num_dims = j.shape[-2]

    # j = j.reshape((-1, num_dims, num_dofs))
    # error = error.reshape((-1, num_dims))

    j_T = j.transpose(-1, -2)
    lam_eye = (torch.eye(j_T.shape[-1], device=j_T.device) * (lam ** 2))

    jj_T_lameye = j @ j_T + lam_eye

    # This computes jj_T_lameye^-1 @ error
    inv_jj_T_lameye_x_error = torch.linalg.lstsq(jj_T_lameye, error).solution

    u = torch.einsum('...ij,...j->...i', j_T, inv_jj_T_lameye_x_error)

    return u


dls_ik = IKFunction(_raw_dls_ik_delta)
