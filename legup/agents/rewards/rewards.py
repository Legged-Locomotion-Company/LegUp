
from legup.agents.rewards.util import squared_norm

from typing import Union

import torch

"""
This file contains the reward functions for the agents.
Currently implemented functions all come from the Wild Anymal paper https://leggedrobotics.github.io/rl-perceptiveloco/assets/pdf/wild_anymal.pdf (pages 18+19)
"""


def lin_velocity(v_des: torch.Tensor, v_act: torch.Tensor, scale: float = 1.0):
    """If the norm of the desired velocity is 0, then the reward is exp(-norm(v_act)^2)
    If the dot product of the desired velocity and the actual velocity is greater than the norm of the desired velocity, then the reward is 1.
    Otherwise, the reward is exp(-(dot(v_des, v_act) - norm(v_des))^2)


    Args:
        v_des (torch.Tensor): Desired X,Y velocity of shape (num_envs, 2)
        v_act (torch.Tensor): Actual  X,Y velocity of shape (num_envs, 2)
        rewards (dict, optional): Dictionary to store the reward. Defaults to None.
        scale (float, optional): Scale the reward. Defaults to 1.0.

    Returns:
        torch.Tensor: the reward for each env of shape (num_envs,)
    """
    # we need the norm of v_des 3 times, so we calculate it once and store it
    # remove the last dimension of v_des_norm, since it is 1
    v_des_norm = torch.norm(v_des, dim=-1).squeeze()  # (num_envs,)

    dots = torch.einsum('Bi,Bj->B', v_des, v_act)

    result = torch.exp(-(dots - v_des_norm)**2)

    mask = v_des_norm == 0.0
    result[mask] = torch.exp(-torch.einsum('Bi,Bi->B', v_act, v_act)[mask])

    mask = dots > v_des_norm
    result[mask] = 1.0

    return result


def ang_velocity(w_des_yaw: torch.Tensor, w_act_yaw: torch.Tensor, scale: float = 1.0):
    """If the desired angular velocity is 0, then the reward is exp(-(w_act_yaw)^2).
    If the dot product of the desired angular velocity and the actual angular velocity is greater than the desired angular velocity, then the reward is 1.
    Otherwise, the reward is exp(-(dot(w_des_yaw,w_act_yaw) - w_des_yaw)^2)

    Args:
        w_des_yaw (torch.Tensor): Desired yaw velocity of shape (num_envs, 1)
        w_act_yaw (torch.Tensor): Actual yaw velocity of shape (num_envs, 1)
        rewards (dict, optional): Dictionary to store the reward. Defaults to None.
        scale (float, optional): Scale the reward. Defaults to 1.0.

    Returns:
        torch.Tensor: the reward for each env of shape (num_envs,)
    """

    dots = w_des_yaw * w_act_yaw

    result = torch.exp(-(dots - w_des_yaw)**2)

    mask = w_des_yaw == 0.0
    result[mask] = torch.exp(-w_act_yaw[mask]**2)

    mask = dots > w_des_yaw
    result[mask] = 1.0

    return result.squeeze()


def linear_ortho_velocity(v_des: torch.Tensor, v_act: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    This term penalizes the velocity orthogonal to the target direction .
    Reward is exp(-3 * norm(v_0)^2), where v_0 is v_act-(dot(v_des,v_act))*v_des.

    Args:
        v_des (torch.Tensor): Desired X,Y,Z velocity of shape (num_envs, 3)
        v_act (torch.Tensor): Actual  X,Y,Z velocity of shape (num_envs, 3)
        rewards (dict, optional): Dictionary to store the reward. Defaults to None.
        scale (float, optional): Scale the reward. Defaults to 1.0.

    Returns:
        torch.Tensor: the reward for each env of shape (num_envs,)
    """
    dot_v_des_v_act = torch.sum(
        v_des * v_act, dim=1).unsqueeze(1)  # (num_envs,1)
    v_0 = v_act - dot_v_des_v_act * v_des  # (num_envs,3)

    x = torch.exp(-3 * squared_norm(v_0))

    x *= scale

    return x


def body_motion(v_z: torch.Tensor, w_x: torch.Tensor, w_y: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """This term penalizes the body velocity in directions not part of the command\n
    Reward is -1.25*v_z^2 - 0.4 * abs(w_x) - 0.4 * abs(w_y)

    Args:
        v_z (torch.Tensor): Velocity in the z direction of shape (num_envs, 1)
        w_x (torch.Tensor): Current angular velocity in the x direction of shape (num_envs, 1)
        w_y (torch.Tensor): Current angular velocity in the y direction of shape (num_envs, 1)
        rewards (dict, optional): Dictionary to store the reward. Defaults to None.
        scale (float, optional): Scale the reward. Defaults to 1.0.

    Returns:
        torch.Tensor: the reward for each env of shape (num_envs,)
    """

    x = (-1.25*torch.pow(v_z, 2) - 0.4 * torch.abs(w_x) - 0.4 * torch.abs(w_y))

    x *= scale

    return x.squeeze()


def foot_clearance(h: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Penalizes the model if the foot is more than 0.2 meters above the ground, with a reward of -1 per foot that is not in compliance.

    Args:
        h (torch.Tensor): Sampled heights around each foot of shape (num_envs, num_feet, num_heights_per_foot). 
        rewards (dict, optional): Dictionary to store the reward. Defaults to None.
        scale (float, optional): Scale the reward. Defaults to 1.0.

    Returns:
        torch.Tensor: the reward for each env of shape (num_envs,)
    """
    x = torch.sum(-1.0 * (h > 0.2), dim=1)

    x *= scale

    return x


def shank_or_knee_col(is_col: torch.Tensor, curriculum_factor: float, scale: float = 1.0) -> torch.Tensor:
    """If any of the shanks or knees are in contact with the ground, the reward is -curriculum_factor

    Args:
        is_col (torch.Tensor): tensor of shape (num_envs, num_shank_and_knee) indicating whether each shank or knee is in contact with the ground
        curriculum_factor (float): the curriculum factor that increases monotonically and converges to 1
        rewards (dict, optional): Dictionary to store the reward. Defaults to None.
        scale (float, optional): Scale the reward. Defaults to 1.0.

    Returns:
        torch.Tensor: the reward for each env of shape (num_envs,)
    """
    x = -curriculum_factor * torch.any(is_col, dim=1)

    x *= scale

    return x


def joint_motion(j_vel:  torch.Tensor, j_vel_t_1: torch.Tensor, dt: float, curriculum_factor: float, scale: float = 1.0) -> torch.Tensor:
    """This term penalizes the joint velocity and acceleration to avoid vibrations

    Args:
        j_vel (torch.Tensor): Joint velocity of shape (num_envs, num_joints)
        j_vel_t_1 (torch.Tensor): Joint velocity at previous time step of shape (num_envs, num_joints)
        dt (float): change in time since previous time step
        curriculum_factor (float): the curriculum factor that increases monotonically and converges to 1
        rewards (dict, optional): Dictionary to store the reward. Defaults to None.
        scale (float, optional): Scale the reward. Defaults to 1.0.

    Returns:
        torch.Tensor: the reward for each env of shape (num_envs,)
    """
    accel = (j_vel - j_vel_t_1)/dt
    x = -curriculum_factor * torch.sum(0.01*(j_vel)**2 + accel**2, dim=1)

    x *= scale

    return x


def joint_constraint(q: torch.Tensor, q_th: Union[float, torch.Tensor], scale: float = 1.0) -> torch.Tensor:
    """This term penalizes the joint position if it is outside of the joint limits.

    Args:
        q (torch.Tensor): Joint position of shape (num_envs, num_joints)
        q_th (float, torch.Tensor): Joint position threshold. Can be scalar, 1d (num_joints,) or 2d (num_envs, num_joints).
        rewards (dict, optional): Dictionary to store the reward. Defaults to None.
        scale (float, optional): Scale the reward. Defaults to 1.0.

    Returns:
        torch.Tensor: the reward for each env of shape (num_envs,)
    """

    mask = q > q_th
    x = torch.sum(-torch.pow(q-q_th, 2) * mask, dim=1)

    x *= scale

    return x


def target_smoothness(joint_target_t: torch.Tensor, joint_t_1_des: torch.Tensor, joint_t_2_des: torch.Tensor, curriculum_factor: float, scale: float = 1.0) -> torch.Tensor:
    """This term penalizes the smoothness of the target foot trajectories

    Args:
        joint_t_des (torch.Tensor): Current joint target position of shape (num_envs, num_joints)
        joint_t_1_des (torch.Tensor): Joint target position at previous time step of shape (num_envs, num_joints)
        joint_t_2_des (torch.Tensor): Joint target position two time steps ago of shape (num_envs, num_joints)
        curriculum_factor float: the curriculum factor that increases monotonically and converges to 1
        rewards (dict, optional): Dictionary to store the reward. Defaults to None.
        scale (float, optional): Scale the reward. Defaults to 1.0.

    Returns:
        torch.Tensor: the reward for each env of shape (num_envs,)
    """
    x = -curriculum_factor * torch.sum((joint_target_t - joint_t_1_des)**2 + (
        joint_target_t - 2*joint_t_1_des + joint_t_2_des)**2, dim=1)

    x *= scale

    return x


def torque_reward(tau: torch.Tensor, curriculum_factor: float, scale: float = 1.0) -> torch.Tensor:
    """This term penalizes the torque to reduce energy consumption

    Args:
        tau (torch.Tensor): Joint torque of shape (num_envs, num_joints)
        curriculum_factor (float): the curriculum factor that increases monotonically and converges to 1
        rewards (dict, optional): Dictionary to store the reward. Defaults to None.
        scale (float, optional): Scale the reward. Defaults to 1.0.

    Returns:
        torch.Tensor: the reward for each env of shape (num_envs,)
    """
    x = -curriculum_factor * torch.sum(tau**2, dim=1)

    x *= scale

    return x


def slip(foot_is_in_contact: torch.Tensor, feet_vel: torch.Tensor, curriculum_factor: float, scale: float = 1.0) -> torch.Tensor:
    """We penealize the foot velocity if the foot is in contact with the ground to reduce slippage

    Args:
        foot_is_in_contact (torch.Tensor): boolean tensor of shape (num_envs, num_feet) indicating whether each foot is in contact with the ground
        feet_vel (torch.Tensor): Foot velocity of shape (num_envs, num_feet, 3)
        curriculum_factor (float): the curriculum factor that increases monotonically and converges to 1
        rewards (dict, optional): Dictionary to store the reward. Defaults to None.
        scale (float, optional): Scale the reward. Defaults to 1.0.

    Returns:
        torch.Tensor: the reward for each env of shape (num_envs,)
    """

    x = -curriculum_factor * \
        torch.sum((foot_is_in_contact * squared_norm(feet_vel)), dim=1)

    x *= scale

    return x


def clip(actions: torch.Tensor, clip_value: float, curriculum_factor: float = 1.0, scale: float = 1.0) -> torch.Tensor:
    """Give a negative reward if the model outputs are greater than the clip value. 
       This is to encourage the model to output values in the range [-clip_value, clip_value]
       formula is -sum((clamp(actions, -clip_value, clip_value) - actions)**2)

    Args:
        actions (torch.Tensor): actions of shape (num_envs, num_joints)
        clip_value (float): the value to clip the reward to
        scale (float, optional): Scale the reward. Defaults to 1.0.

    Returns:
        torch.Tensor: the reward for each env of shape (num_envs,)
    """

    clip_value *= curriculum_factor

    deltas = torch.clamp(actions, -clip_value,
                         clip_value) - actions

    x = -torch.sum(torch.pow(deltas, 2), dim=1)

    x *= scale

    return x
