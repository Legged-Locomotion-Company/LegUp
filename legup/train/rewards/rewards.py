
from legup.train.rewards.reward_helpers import squared_norm

from typing import Union

import torch

"""
This file contains the reward functions for the agents.
Currently implemented functions all come from the Wild Anymal paper https://leggedrobotics.github.io/rl-perceptiveloco/assets/pdf/wild_anymal.pdf (pages 18+19)
"""


class BaseReward:
    def __init__(self, env, robot_config, weight, dt):
        self.env = env
        self.robot_config = robot_config
        self.weight = weight

    def __call__(self, rewards: dict = None) -> torch.Tensor:
        reward_value = self.calculate_reward() * self.weight

        if rewards is not None:
            rewards[type(self).reward_name] = reward_value

        return reward_value


class CommandVelocityReward(BaseReward):
    """Reward function which encourages the robot to move at the command velocity."""

    reward_name = 'lin_velocity_reward'

    def calculate_reward(self) -> torch.Tensor:
        v_des = self.env.get_command_velocity()
        v_act = self.env.get_rb_velocity()[:, self.robot_config.base_index, :2]

        v_des_norm = self.env.get_command_velocity().norm(dim=1)
        dots = torch.einsum('Bi,Bj->B', v_des, v_act)

        result = torch.exp(-dots - v_des_norm**2)

        mask = v_des_norm == 0.0
        result[mask] = torch.exp(-v_act.norm(dim=-1)[mask])

        mask = dots > v_des_norm
        result[mask] = 1.0

        return result


def lin_velocity(v_des: torch.Tensor, v_act: torch.Tensor, rewards: dict = None, scale: float = 1.0):
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
    v_des_norm = torch.norm(v_des, dim=1).squeeze()  # (num_envs,)

    # calculate the dot product of v_des and v_act across dim 1
    dot_v_des_v_act = torch.sum(v_des * v_act, dim=1)  # (num_envs,)

    # use masks to select the correct reward function for a given env
    x = torch.exp(squared_norm(v_act)) * (v_des_norm == 0)
    x = 1 * (dot_v_des_v_act > v_des_norm)
    x = torch.exp(-torch.pow(dot_v_des_v_act - v_des_norm, 2)) * \
        ~((v_des_norm == 0) + dot_v_des_v_act > v_des_norm)

    x *= scale

    if rewards is not None:
        rewards['lin_velocity_reward'] = x

    return x


def ang_velocity(w_des_yaw, w_act_yaw, rewards: dict = None, scale: float = 1.0):
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

    # elementwise multiplication since w_des_yaw and w_act_yaw are 1D
    dot_w_des_w_act = w_des_yaw * w_act_yaw

    x = torch.exp(-torch.pow(dot_w_des_w_act - w_des_yaw, 2))

    # This is the case for when w_des_yaw is 0
    x[w_des_yaw == 0] = torch.exp(-torch.pow(w_act_yaw, 2)) * (w_des_yaw == 0)

    # This is the case for when dot_w_des_w_act > w_des_yaw
    x[dot_w_des_w_act > w_des_yaw] = 1.0

    x *= scale

    if rewards is not None:
        rewards['ang_velocity_reward'] = x

    return x


def linear_ortho_velocity(v_des: torch.Tensor, v_act: torch.Tensor, rewards: dict = None, scale: float = 1.0) -> torch.Tensor:
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

    if rewards is not None:
        rewards['lin_ortho_velocity_reward'] = x

    return x


def body_motion(v_z: torch.Tensor, w_x: torch.Tensor, w_y: torch.Tensor, rewards: dict = None, scale: float = 1.0) -> torch.Tensor:
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

    if rewards is not None:
        rewards['body_motion_reward'] = x

    return x


def foot_clearance(h: torch.Tensor, rewards: dict = None, scale: float = 1.0) -> torch.Tensor:
    """Penalizes the model if the foot is more than 0.2 meters above the ground, with a reward of -1 per foot that is not in compliance.

    Args:
        h (torch.Tensor): Sampled heights around each foot of shape (num_envs, num_feet, num_heights_per_foot). 
        rewards (dict, optional): Dictionary to store the reward. Defaults to None.
        scale (float, optional): Scale the reward. Defaults to 1.0.

    Returns:
        torch.Tensor: the reward for each env of shape (num_envs,)
    """
    x = torch.sum(-1 * (h > 0.2), dim=1)

    x *= scale

    if rewards is not None:
        rewards['foot_clearance_reward'] = x

    return x


def shank_or_knee_col(is_col: torch.Tensor, curriculum_factor: float, rewards: dict = None, scale: float = 1.0) -> torch.Tensor:
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

    if rewards is not None:
        rewards['shank_or_knee_col_reward'] = x

    return x


def joint_motion(j_vel:  torch.Tensor, j_vel_t_1: torch.Tensor, dt: float, curriculum_factor: float, rewards: dict = None, scale: float = 1.0) -> torch.Tensor:
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

    if rewards is not None:
        rewards['joint_motion_reward'] = x

    return x


def joint_constraint(q: torch.Tensor, q_th: Union[float, torch.Tensor], rewards: dict = None, scale: float = 1.0) -> torch.Tensor:
    """This term penalizes the joint position if it is outside of the joint limits.

    Args:
        q (torch.Tensor): Joint position of shape (num_envs, num_joints)
        q_th (float, torch.Tensor): Joint position threshold. Can be scalar, 1d (num_joints,) or 2d (num_envs, num_joints).
        rewards (dict, optional): Dictionary to store the reward. Defaults to None.
        scale (float, optional): Scale the reward. Defaults to 1.0.

    Returns:
        torch.Tensor: the reward for each env of shape (num_envs,)
    """

    q_th = torch.tensor(q_th).to(q.device).expand(q.shape)

    mask = q > q_th
    x = torch.sum(-torch.pow(q-q_th, 2) * mask, dim=1)

    x *= scale

    if rewards is not None:
        rewards['joint_constraint_reward'] = x

    return x


def target_smoothness(joint_target_t: torch.Tensor, joint_t_1_des: torch.Tensor, joint_t_2_des: torch.Tensor, curriculum_factor: float, rewards: dict = None, scale: float = 1.0) -> torch.Tensor:
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

    if rewards is not None:
        rewards['target_smoothness_reward'] = x

    return x


def torque_reward(tau: torch.Tensor, curriculum_factor: float, rewards: dict = None, scale: float = 1.0) -> torch.Tensor:
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

    if rewards is not None:
        rewards['torque_reward'] = x

    return x


def slip(foot_is_in_contact: torch.Tensor, feet_vel: torch.Tensor, curriculum_factor: float, rewards: dict = None, scale: float = 1.0) -> torch.Tensor:
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

    if rewards is not None:
        rewards['slip_reward'] = x

    return x
