from legup.robots.mini_cheetah.kinematics.mini_cheetah_kin_torch import mini_cheetah_dls_invkin

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def walk_half_circle_line(q_vec, pos_phase_deltas, phases):
    """This function takes the robot positions, desired foot deltas and theta deltas, and returns the desired foot position along the half_circle_line profile with a walk gait.

    Args:
        q_vec (torch.Tensor): a (NUM_ENVS x 12) tensor with the current joint angles of the joints in each env
        pos_phase_deltas (torch.Tensor): a (NUM_ENVS x 16) tensor with the (x,y,z) foot deltas and phase deltas for each foot in each env
        phases (torch.Tensor): a (NUM_ENVS x 4) tensor with the scalar robot phase for each environment and foot

    Returns:
        torch.Tensor: a (NUM_ENVS x 12) tensor with the new target joint angles of each foot in the robot
    """

    phase_offsets = torch.tensor([0, 0, 0, 0], device=device)

    result = use_gait_footpath(q_vec, pos_phase_deltas, phases, half_circle_line, phase_offsets)

    return result


def use_gait_footpath(q_vec, pos_phase_deltas, phase, path_func, phase_offsets):
    """This function takes the joint positions, desired foot deltas, path function, and phase,
    gives the new robot q_vec along that path with that phase

    Args:
        q_vec (torch.Tensor): a (NUM_ENVS x 12) vector giving the joint positions of each joint of the robots in each env
        pos_phase_deltas (torch.Tensor): a (NUM_ENVS x 16) vector giving (x,y,z) deltas from the current position along the preplanned path for the foot, stacked with a theta delta for each of the foot phases
        phase (torch.Tensor): a (NUM_ENVS x NUM_FEET) vector giving the gait phase for the robot in each of the envs for each foot
        path_func (function): This function takes a phase, and outputs an (x,y,z) position along the path
        phase_offsets (torch.tensor) a (4) vector giving an offset for each of the feet's phase.

    Returns:
        torch.Tensor: a (NUM_ENVS x 12) vector which gives the new target joint positions for each of the robots in each of the envs
    """

    pos_deltas, phase_deltas = torch.split(pos_phase_deltas, (12, 4), dim=-1)
    per_foot_pos_deltas = torch.reshape(pos_deltas, (-1, 4, 3))

    phases = phase_deltas + phase + phase_offsets.to(device)
    goal_positions = path_func(phases) + per_foot_pos_deltas

    # print(f"path[0]: {path_func(phases)[0]}")
    # print(f"phase[0]: {phase[0]}")

    new_positions = mini_cheetah_dls_invkin(q_vec, goal_positions)

    return new_positions


def half_circle_line(phase):
    """Gives an (x,y,z) position based on phase given, for the first half of pi part of the phase it does a half circle.
    For the second half it just follows a line back to the start

    Args:
        phase (torch.Tensor): a (NUM_ENVS x 4) tensor for the phase of each foot

    Returns:
        torch.Tensor: a (NUM_ENVS x 4 x 3) tensor for the target position of each foot
    """

    NUM_ENVS, _ = phase.shape

    x_amp = 0.1
    z_amp = 0.1

    phase = torch.remainder(phase, torch.pi*2)

    result = torch.zeros((NUM_ENVS, 4, 3)).to(device)

    result[:, :, 0] = -torch.cos(phase) * x_amp
    result[:, :, 1] = torch.zeros_like(phase).to(device)
    result[:, :, 2] = torch.sin(phase) * z_amp

    mask = phase > torch.pi
    result[:, :, 2][mask] = 0.

    return result
