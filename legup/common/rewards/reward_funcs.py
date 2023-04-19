from legup.common.abstract_dynamics import AbstractDynamics
from legup.common.legged_robot import LeggedRobot
from legup.common.rewards.rewards import reward, RewardArgs

from omegaconf import DictConfig

from typing import Iterable, Callable, Tuple, Dict

import torch


@reward
def lin_velocity(reward_args: RewardArgs) -> torch.Tensor:
    """If the norm of the desired velocity is 0, then the reward is exp(-norm(v_act) ^ 2)
    If the dot product of the desired velocity and the actual velocity is greater than the norm of the desired velocity, then the reward is 1.
    Otherwise, the reward is exp(-(dot(v_des, v_act) - norm(v_des)) ^ 2)
    """

    # we need the norm of v_des 3 times, so we calculate it once and store it
    # remove the last dimension of v_des_norm, since it is 1
    v_des = reward_args.dynamics.command
    v_act = reward_args.dynamics.get_linear_velocity()

    v_act_norm_sq = torch.einsum('Bi,Bi->B', v_act, v_act)
    v_des_norm = torch.norm(v_des, dim=-1).squeeze()  # (num_envs,)

    act_dot_des = torch.einsum('Bi,Bi->B', v_des, v_act)

    result = (-(act_dot_des - v_des_norm).square()).exp()

    mask = v_des_norm == 0.0
    result[mask] = (-torch.einsum(v_act_norm_sq[mask])).exp()

    mask = act_dot_des > v_des_norm
    result[mask] = 1.0

    return result


@reward
def ang_velocity(reward_args: RewardArgs) -> torch.Tensor:
    """If the desired angular velocity is 0, then the reward is exp(-(w_act_yaw) ^ 2).
    If the dot product of the desired angular velocity and the actual angular velocity is greater than the desired angular velocity, then the reward is 1.
    Otherwise, the reward is exp(-(dot(w_des_yaw, w_act_yaw) - w_des_yaw) ^ 2)
    """

    if (command := getattr(reward_args, 'command')) is None:
        command = torch.zeros((reward_args.dynamics.get_num_agents(), 3),
                              device=reward_args.dynamics.device())

    w_des_yaw = command[:, :2]
    w_act_yaw = reward_args.dynamics.get_angular_velocity()[:, :2]

    dots = w_des_yaw * w_act_yaw

    result = torch.exp(-(dots - w_des_yaw)**2)

    mask = w_des_yaw == 0.0
    result[mask] = torch.exp(-w_act_yaw[mask]**2)

    mask = dots > w_des_yaw
    result[mask] = 1.0

    result = result.squeeze()

    return result


class Rewards:
    def __init__(self,
                 dynamics: AbstractDynamics,
                 robot: LeggedRobot,
                 scale: DictConfig):
        """This class contains all of the reward functions.
        Instantiating it for a dynamics and robot allows rewards to be computed for that robot in that environment

        Args:
            dynamics (AbstractDynamics): This is a dynamics object which allows the reward functions to query the environment.
            robot (Robot): This is a robot object which contains robot features and kinematics
            scale (DictConfig): This is an object which contains scales for different reward functions.
        """

        self.reward_info = {}
        self.curriculum_factor = 0.0

        self.dynamics = dynamics
        self.robot = robot

        self.scale = scale

    def update_rewards(self,
                       command: torch.Tensor,
                       rewards: Iterable[Callable[[], None]] = {},
                       curriculum_factor: float = 1.0
                       ) -> Dict[str, torch.Tensor]:
        """This function updates the rewards for the current timestep.

        Args:
            command (torch.Tensor): This is the desired velocity of the robot
            rewards (Iterable[Callable[[], None]], optional): This is a list of reward functions. Defaults to {}.
            curriculum_factor (float, optional): This is the current curriculum factor. Defaults to 1.0.

        Returns:
            Dict[str, torch.Tensor]: This is a dictionary of keys are reward names, values are (num_envs) shaped tensors containing per env rewards
        """

        self.command = command

        self.reward_info = {}
        self.curriculum_factor = curriculum_factor

        for reward in rewards:
            reward()

        return self.reward_info

    def get_total_reward(self):
        rewards_list = list(self.reward_info.values())
        stacked_rewards = torch.stack(rewards_list)
        return stacked_rewards.sum(dim=0)

    def lin_velocity(self):
        """If the norm of the desired velocity is 0, then the reward is exp(-norm(v_act) ^ 2)
        If the dot product of the desired velocity and the actual velocity is greater than the norm of the desired velocity, then the reward is 1.
        Otherwise, the reward is exp(-(dot(v_des, v_act) - norm(v_des)) ^ 2)
        """

        # we need the norm of v_des 3 times, so we calculate it once and store it
        # remove the last dimension of v_des_norm, since it is 1
        v_des = self.command
        v_act = self.dynamics.get_linear_velocity()
        v_des_norm = torch.norm(v_des, dim=-1).squeeze()  # (num_envs,)

        dots = torch.einsum('Bi,Bj->B', v_des, v_act)

        result = torch.exp(-(dots - v_des_norm)**2)

        mask = v_des_norm == 0.0
        result[mask] = torch.exp(-torch.einsum('Bi,Bi->B', v_act, v_act)[mask])

        mask = dots > v_des_norm
        result[mask] = 1.0

        self.reward_info['linear_velocity_reward'] = result * \
            self.scale['velocity']

    def ang_velocity(self):
        """If the desired angular velocity is 0, then the reward is exp(-(w_act_yaw) ^ 2).
        If the dot product of the desired angular velocity and the actual angular velocity is greater than the desired angular velocity, then the reward is 1.
        Otherwise, the reward is exp(-(dot(w_des_yaw, w_act_yaw) - w_des_yaw) ^ 2)
        """

        command = self.command
        w_des_yaw = command[:, :2]
        w_act_yaw = self.dynamics.get_angular_velocity()[:, :2]

        dots = w_des_yaw * w_act_yaw

        result = torch.exp(-(dots - w_des_yaw)**2)

        mask = w_des_yaw == 0.0
        result[mask] = torch.exp(-w_act_yaw[mask]**2)

        mask = dots > w_des_yaw
        result[mask] = 1.0

        result = result.squeeze()

        self.reward_info['angular_velocity_reward'] = result * \
            self.scale['velocity']

    def linear_ortho_velocity(self):
        """
        This term penalizes the velocity orthogonal to the target direction .
        Reward is exp(-3 * norm(v_0) ^ 2), where v_0 is v_act-(dot(v_des, v_act))*v_des.
        """

        v_des = self.command
        v_act = self.dynamics.get_linear_velocity()

        dot_v_des_v_act = torch.sum(
            v_des * v_act, dim=1).unsqueeze(1)  # (num_envs,1)
        v_0 = v_act - dot_v_des_v_act * v_des  # (num_envs,3)

        x = torch.exp(-3 * squared_norm(v_0))

        self.reward_info['linear_ortho_velocity_reward'] = x * \
            self.scale['velocity']

    def body_motion(self):
        """This term penalizes the body velocity in directions not part of the command
        Reward is -1.25*v_z ^ 2 - 0.4 * abs(w_x) - 0.4 * abs(w_y)
        """
        v_z = self.dynamics.get_linear_velocity()[:, 2]
        w_x = self.dynamics.get_angular_velocity()[:, 0]
        w_y = self.dynamics.get_angular_velocity()[:, 1]

        x = (-1.25*torch.pow(v_z, 2) - 0.4 *
             torch.abs(w_x) - 0.4 * torch.abs(w_y))

        self.reward_info['body_motion_reward'] = x * self.scale['body_motion']

    def foot_clearance(self):
        """Penalizes the model if the foot is more than 0.2 meters above the ground, with a reward of - 1 per foot that is not in compliance.
        """

        h = self.dynamics.get_rb_position(
        )[:, self.robot.foot_link_indices, 2]

        x = torch.sum(-1.0 * (h > 0.2), dim=1)

        self.reward_info['foot_clearance_reward'] = x * \
            self.scale['foot_clearance']

    def shank_or_knee_col(self):
        """If any of the shanks or knees are in contact with the ground, the reward is -curriculum_factor
        """
        is_col = self.dynamics.get_contact_states(
        )[:, self.robot.shank_link_indices]

        x = -self.curriculum_factor * torch.any(is_col, dim=1)
        self.reward_info['shank_or_knee_col_reward'] = x * \
            self.scale['shank_knee_col']

    def joint_motion(self):
        """This term penalizes the joint velocity and acceleration to avoid vibrations
        """

        j_vel_hist = self.dynamics.get_joint_velocity_hist()
        j_vel = j_vel_hist[0]
        j_vel_t_1 = j_vel_hist[1]

        accel = (j_vel - j_vel_t_1)/self.dynamics.get_dt()
        x = -self.curriculum_factor * \
            torch.sum(0.01*(j_vel)**2 + accel**2, dim=1)

        self.reward_info['joint_motion_reward'] = x * \
            self.scale['joint_velocities']

    def joint_constraint(self):
        """This term penalizes the joint position if it is outside of the joint limits.

        Args:
            dynamics(Dynamics): Dynamics object
            scale(float, optional): Scale the reward. Defaults to 1.0.
            info_dict(dict): Dictionary for logging information

        Returns:
            torch.Tensor: the reward for each env of shape(num_envs,)
        """

        joint_pos = self.dynamics.get_joint_position()
        q = joint_pos[:, self.robot.knee_joint_indices]
        q_th = self.robot.knee_joint_limits

        mask = q > q_th
        x = torch.sum(-torch.pow(q-q_th, 2) * mask, dim=1)

        self.reward_info['joint_constraint_reward'] = x * \
            self.scale['joint_constraints']

    def target_smoothness(self):
        """This term penalizes the smoothness of the target foot trajectories
        """
        joint_target_hist = self.dynamics.get_joint_position_hist()
        joint_target_t = joint_target_hist[0]
        joint_target_tm1 = joint_target_hist[1]
        joint_target_tm2 = joint_target_hist[2]

        x = -self.curriculum_factor * torch.sum((joint_target_t - joint_target_tm1)**2 + (
            joint_target_t - 2*joint_target_tm1 + joint_target_tm2)**2, dim=1)

        self.reward_info['target_smoothness_reward'] = x * \
            self.scale['target_smoothness']

    def torque_reward(self):
        """This term penalizes the torque to reduce energy consumption

        Args:
            dynamics(Dynamics): Dynamics object
            scale(float, optional): Scale the reward. Defaults to 1.0.
            info_dict(dict): Dictionary for logging information

        Returns:
            torch.Tensor: the reward for each env of shape(num_envs,)
        """
        tau = self.dynamics.get_joint_torque()

        x = -self.curriculum_factor * torch.sum(tau**2, dim=1)

        self.reward_info['torque_reward'] = x * self.scale['torque']

    def slip(self):
        """We penealize the foot velocity if the foot is in contact with the ground to reduce slippage
        """
        foot_is_in_contact = self.dynamics.get_contact_states()[
            :, self.robot.foot_link_indices]

        feet_vel = self.dynamics.get_rb_linear_velocity()[
            :, self.robot.foot_link_indices]

        x = -self.curriculum_factor * \
            torch.sum((foot_is_in_contact * squared_norm(feet_vel)), dim=1)

        self.reward_info['slip_reward'] = x * self.scale['slip']


def squared_norm(x: torch.Tensor, dim=-1) -> torch.Tensor:
    """Calculates the squared norm of a tensor
    Args:
        x(torch.Tensor): Arbitrarily shaped tensor
        dim(int, optional): Dimension to calculate the norm over. Defaults to - 1.
    Returns:
        torch.Tensor: The squared norm of the tensor across the given dimension
    """
    return torch.sum(torch.pow(x, 2), dim=dim)
