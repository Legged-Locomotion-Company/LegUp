from omegaconf import DictConfig
import torch
from typing import List

from legup.train.agents.base import BaseAgent
from legup.train.rewards.anymal_rewards import WildAnymalReward
from legup.robots.Robot import Robot


class AnymalAgent(BaseAgent):
    """Specific agent implementation for https://leggedrobotics.github.io/rl-perceptiveloco/assets/pdf/wild_anymal.pdf
    """

    def __init__(self, robot_cfg: Robot, num_environments: int, asset_path: str, asset_name: str, train_cfg: DictConfig):
        """Initialize wild anymal agent.

        Args:
            robot_cfg (Robot): Robot interface configuration class
            num_environments (int): Number of environments
            asset_path (str): Path to the robot asset
            asset_name (str): Name of the robot asset
            train_cfg (DictConfig): Configuration dictionary for training
        """
        super().__init__(robot_cfg, num_environments, asset_path, asset_name)

        # need more stuff for reward function like train config
        self.dt = 1/60
        # (self, env, robot_config, train_config, dt: float):
        self.reward_fn = WildAnymalReward(
            self.env, robot_cfg, train_cfg, dt=self.dt)
        self.robot_cfg = robot_cfg
        self.train_cfg = train_cfg

        self.reset_history_vec()

    def reset_history_vec(self, idx=None):
        # 3 timesteps for history, 2 for velocity
        if idx is not None:
            self.joint_pos_history[idx] = torch.zeros(
                self.robot_cfg.num_joints, 3).to(self.device)
            self.joint_vel_history[idx] = torch.zeros(
                self.robot_cfg.num_joints, 2).to(self.device)
            self.joint_target_history[idx] = torch.zeros(
                self.robot_cfg.num_joints, 2).to(self.device)
        else:
            self.joint_pos_history = torch.zeros(
                self.num_envs, self.robot_cfg.num_joints, 3).to(self.device)
            self.joint_vel_history = torch.zeros(
                self.num_envs, self.robot_cfg.num_joints, 2).to(self.device)
            self.joint_target_history = torch.zeros(
                self.num_envs, self.robot_cfg.num_joints, 2).to(self.device)

    def make_observation(self, idx=None):
        if idx is None:
            idx = torch.arange(self.num_envs)

        proprio = torch.zeros(self.num_envs, 133).to(self.device)
        extro = torch.zeros(self.num_envs, 208).to(self.device)
        privil = torch.zeros(self.num_envs, 50).to(self.device)

        # TODO: talk to rohan about the command

        proprio[idx, :3] = torch.tensor([1., 0., 0.]).to(
            self.device)  # self.command[idx]

        proprio[idx, 3:6] = self.env.get_position()[idx]
        proprio[idx, 6:9] = self.env.get_linear_velocity()[idx]

        proprio[idx, 9:12] = self.env.get_angular_velocity()[idx]
        proprio[idx, 12:24] = self.env.get_joint_position()[idx]
        proprio[idx, 24:36] = self.env.get_joint_velocity()[idx]

        proprio[idx, 36:72] = self.joint_pos_history[idx].flatten(start_dim=1)
        proprio[idx, 72:96] = self.joint_vel_history[idx].flatten(start_dim=1)
        proprio[idx, 96:120] = self.joint_target_history[idx].flatten(
            start_dim=1)

        # TODO: talk to misha about getting the CPG phase generation
        # proprio[:, 120:133] = self.phase_gen(idx)

        privil[idx, :4] = self.env.get_contact_states(
        )[idx][:, self.robot_cfg.foot_indices].to(torch.float)

        privil[idx, 4:16] = self.env.get_contact_forces(
        )[idx][:, self.robot_cfg.foot_indices, :].flatten(start_dim=1)
        # privil[idx, 16:28] = self.env.get_contact_normals()
        # privil[idx, 28, 32] = self.enc.get_frivtion_coeffs()
        privil[idx, 32:40] = self.env.get_contact_states()[idx][
            :, self.robot_cfg.shank_indices + self.robot_cfg.thigh_indices].to(torch.float)

        # TODO: add airtime

        self.joint_pos_history[idx, :, 1:] = self.joint_pos_history[idx, :, :2]
        self.joint_pos_history[idx, :, 0] = self.env.get_joint_position()[idx]

        self.joint_vel_history[idx, :, 1] = self.joint_vel_history[idx, :, 0]
        self.joint_vel_history[idx, :, 0] = self.env.get_joint_velocity()[idx]

        self.joint_target_history[idx, :,
                                  1] = self.joint_target_history[idx, :, 0]
        self.joint_target_history[idx, :, 0] = self.env.get_joint_position()[
            idx]

        return (torch.cat([proprio, extro, privil], dim=1)[idx]).cpu().numpy()

    # TODO: talk to Rohan - Base class passes in actions, unecessary for this reward function
    def make_reward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.reward_fn(self.joint_vel_history[:, :, 0],
                              self.joint_target_history[:, :, 0],
                              self.joint_target_history[:, :, 1])

    def reset_envs(self, envs):
        self.reset_history_vec(envs)

    def check_termination(self) -> torch.Tensor:
        # Check if any rigidbodies are hitting the ground
        # 0 = rb index of body
        is_collided = torch.any(self.env.get_contact_states(), dim=-1)

        # Check if the robot is tilted too much
        is_tilted = torch.any(
            torch.abs(self.env.get_rotation()) > self.train_cfg["max_tilt"], dim=-1)

        # Check if the robot's movements exceed the torque limits
        is_exceeding_torque = torch.any(
            torch.abs(self.env.get_joint_torque()) > self.train_cfg["max_torque"], dim=-1)

        return (is_collided + is_tilted + is_exceeding_torque).bool()
