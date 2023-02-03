from legup.train.agents.base import BaseAgent
from legup.train.rewards.anymal_rewards import WildAnymalReward
from legup.robots.Robot import Robot
from legup.robots.mini_cheetah.kinematics.mini_cheetah_footpaths import walk_half_circle_line
from legup.train.rewards.util import HistoryBuffer
from omegaconf import DictConfig
from typing import List
from typing import Tuple
import numpy as np
import gym

import torch


class AnymalAgent(BaseAgent):
    """Specific agent implementation for https://leggedrobotics.github.io/rl-perceptiveloco/assets/pdf/wild_anymal.pdf
    """

    def __init__(self, robot_cfg: Robot, num_environments: int, curriculum_exponent: int, asset_path: str, asset_name: str, train_cfg: DictConfig):
        """Initialize wild anymal agent.

        Args:
            robot_cfg (Robot): Robot interface configuration class
            num_environments (int): Number of environments
            asset_path (str): Path to the robot asset
            asset_name (str): Name of the robot asset
            train_cfg (DictConfig): Configuration dictionary for training
        """

        super().__init__(robot_cfg, num_environments,
                         curriculum_exponent, asset_path, asset_name)

        # need more stuff for reward function like train config
        self.dt = 1/60
        # (self, env, robot_config, train_config, dt: float):
        self.reward_fn = WildAnymalReward(
            self.env, robot_cfg, train_cfg, dt=self.dt)
        self.robot_cfg = robot_cfg
        self.train_cfg = train_cfg

        self.hit_factor = 0.0

        self.clip_factor = 10**(-self.train_cfg.clip_exponent)

        self.reset_history_vec()

        self.prev_obs = HistoryBuffer(
            num_environments, self.dt, self.dt, 5, 133 + 208 + 50, self.device)
        self.prev_action = HistoryBuffer(
            num_environments, self.dt, self.dt, 5, 12, self.device)

        self.final_hit_max_mag = 0.5
        self.final_hit_min_mag = 0.25

        self.clip_high_max = torch.tensor(
            [self.train_cfg.pos_delta_clip] * 12 +
            [self.train_cfg.phase_delta_clip] * 4,
            dtype=torch.float32, device=self.device)

        self.clip_low_max = self.clip_high_max.neg()

        self.clip_high = torch.zeros_like(self.clip_high_max)
        self.clip_low = torch.zeros_like(self.clip_low_max)

        self.action_space = gym.spaces.Box(
            low=self.clip_low_max.cpu().numpy(),
            high=self.clip_high_max.cpu().numpy(),
            dtype=np.float32)

        self.update_factors()

    def step_curriculum(self):
        """Empty implementation to prevent curriculum from being stepped by the base class"""
        return

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

    def phase_gen(self):
        cpg_freq = 4.0
        base_frequencies = torch.tensor([cpg_freq] * 4).to(self.device)
        phase_offsets = torch.tensor(
            [0, torch.pi, torch.pi, 0]).to(self.device)

        phase = (self.ep_lens * self.dt).expand(4, self.num_envs).T * \
            base_frequencies * torch.pi * 2
        phase += phase_offsets.expand(phase.shape)

        phase = phase % (2 * torch.pi)

        return phase

    def make_phase_observation(self):
        # Make some clever way to get the phase offsets, for now we just hardcode it
        cpg_freq = 4.0

        phase = self.phase_gen()

        phase_cos = torch.cos(phase)
        phase_sin = torch.sin(phase)

        stacked = torch.cat([phase, phase_cos, phase_sin], dim=1)

        cpg_freq_expanded = torch.tensor([cpg_freq]).to(
            self.device).expand(self.ep_lens.shape)

        return torch.cat([cpg_freq_expanded.unsqueeze(-1), stacked], dim=1)

    def dump_log(self):
        # check for nans
        nan_idx = torch.unique(torch.argwhere(
            torch.isnan(self.prev_obs.get(0)))[:, 0])
        for ni in nan_idx:
            for ts in range(5):
                # print(f'OBS AT TIMESTEP {ts} FOR ENV {ni}: ')
                # self.explain_observation(self.prev_obs.get(ts)[ni])
                print(f'ACTIONS AT TIMESTEP {ts} FOR ENV {ni}: ')
                print([round(i.item(), 4)
                      for i in self.prev_action.get(ts)[ni]])
                print()

    def explain_observation(self, obs):
        def round_list_nice(vec):
            return [round(i.item(), 4) for i in vec]

        proprio, privil = obs[:133], obs[341:391]

        command = proprio[0:3]
        pos = proprio[3:6]
        linvel = proprio[6:9]
        angvel = proprio[9:12]
        jointpos = proprio[12:24]
        jointvel = proprio[24:36]
        jointposhist = proprio[36:72]
        jointvelhist = proprio[72:96]
        jointtarghist = proprio[96:120]
        phaseobs = proprio[120:133]

        feetcontacts = privil[:4]
        feetcontactforces = privil[4:16]
        shankthighcontacts = privil[32:40]

        print(f'Command: {round_list_nice(command)}')
        print(f'Position: {round_list_nice(pos)}')
        print(f'Linear Velocity: {round_list_nice(linvel)}')
        print(f'Angular Velocity: {round_list_nice(angvel)}')
        print(f'Joint Pos: {round_list_nice(jointpos)}')
        print(f'Joint Vel: {round_list_nice(jointvel)}')
        print(f'Joint Pos Hist: {round_list_nice(jointposhist)}')
        print(f'Joint Vel Hist: {round_list_nice(jointvelhist)}')
        print(f'Joint Target Hist: {round_list_nice(jointtarghist)}')
        print(f'Phase Observation: {round_list_nice(phaseobs)}')
        print(f'Feet Contacts: {round_list_nice(feetcontacts)}')
        print(f'Feet Contact Forces: {round_list_nice(feetcontactforces)}')
        print(f'ShankThighContacts: {round_list_nice(shankthighcontacts)}')
        print()

    def make_observation(self, idx=None):
        if idx is None:
            idx = torch.arange(self.num_envs)

        proprio = torch.zeros(self.num_envs, 133).to(self.device)
        extro = torch.zeros(self.num_envs, 208).to(self.device)
        privil = torch.zeros(self.num_envs, 50).to(self.device)

        # TODO: talk to rohan about the command
        # ill work on this - Rohan

        proprio[idx, :3] = self.commands[idx]

        proprio[idx, 3:6] = self.env.get_position()[idx]
        proprio[idx, 6:9] = self.env.get_linear_velocity()[idx]

        proprio[idx, 9:12] = self.env.get_angular_velocity()[idx]
        proprio[idx, 12:24] = self.env.get_joint_position()[idx]
        proprio[idx, 24:36] = self.env.get_joint_velocity()[idx]

        proprio[idx, 36:72] = self.joint_pos_history[idx].flatten(start_dim=1)
        proprio[idx, 72:96] = self.joint_vel_history[idx].flatten(start_dim=1)
        proprio[idx, 96:120] = self.joint_target_history[idx].flatten(
            start_dim=1)

        proprio[:, 120:133] = self.make_phase_observation()[idx]

        privil[idx, :4] = self.env.get_contact_states(
        )[idx][:, self.robot_cfg.foot_indices].to(torch.float)

        privil[idx, 4:16] = self.env.get_contact_forces(
        )[idx][:, self.robot_cfg.foot_indices, :].flatten(start_dim=1)
        # privil[idx, 16:28] = self.env.get_contact_normals()
        # privil[idx, 28:32] = self.enc.get_frivtion_coeffs()
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

        obs = torch.cat([proprio, extro, privil], dim=1)

        self.prev_obs.step(obs)

        if torch.any(torch.isnan(obs)):
            self.dump_log()

        # return (torch.cat([proprio, extro, privil], dim=1)[idx]).cpu().numpy()
        return (torch.cat([proprio, extro, privil], dim=1)[idx])

    def make_actions(self, actions: torch.Tensor) -> torch.Tensor:

        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions).to(self.device)

        actions.clip_(self.clip_low, self.clip_high)

        actions = walk_half_circle_line(
            self.env.get_joint_position(), actions, self.phase_gen())

        self.prev_action.step(actions)

        return actions

    def make_reward(self, actions: torch.Tensor) -> torch.Tensor:
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions).to(self.device)

        total_reward, reward_keys, reward_vals = self.reward_fn(previous_joint_velocities=self.joint_vel_history[:, :, 0],
                                                                joint_target_t_1=self.joint_target_history[:, :, 0],
                                                                joint_target_t_2=self.joint_target_history[:, :, 1],
                                                                actions=actions,
                                                                command=self.commands,
                                                                clip_low=self.clip_low,
                                                                clip_high=self.clip_high,
                                                                curriculum_factor=self.curriculum_factor)

        if reward_vals[list(reward_keys).index("lin_velocity_reward")].mean() > self.train_cfg.reward_scales.velocity * 0.8:
            curriculum_step = 0.01
            hit_factor_step = 0.01

            self.curriculum_factor = min(
                self.curriculum_factor + curriculum_step, 1.0)
            if self.curriculum_factor >= 1.0:
                self.hit_factor = min(self.hit_factor + hit_factor_step, 1.0)

        self.clip_factor **= 1-10**(-self.train_cfg.clip_exponent)

        self.update_factors()

        return total_reward, reward_keys, reward_vals

    def update_factors(self):

        self.push_mag_upper = self.push_mag_upper_max * self.hit_factor
        self.push_mag_lower = self.push_mag_lower_max * self.hit_factor

        biased_clip_factor = ((1 - self.train_cfg.clip_bias) *
                              self.clip_factor + self.train_cfg.clip_bias)

        clip_avg = (self.clip_high_max + self.clip_low_max) / 2
        clip_half_range = (self.clip_high_max - self.clip_low_max) / 2

        self.clip_high = clip_avg + clip_half_range * biased_clip_factor
        self.clip_low = clip_avg - clip_half_range * biased_clip_factor

    def make_logs(self) -> dict:
        return {
            "curriculum_factor": self.curriculum_factor,
            "hit_factor": self.hit_factor,
            "clip_factor": self.clip_factor, }

    def reset_envs(self, envs):
        self.reset_history_vec(envs)

    def check_termination(self) -> torch.Tensor:
        # Check if any rigidbodies are hitting the ground
        # 0 = rb index of body
        is_collided = torch.any(self.env.get_contact_states()[
            :, [0, 2, 5, 8, 11]], dim=-1)

        # # Check if the robot is tilted too much
        # is_tilted = torch.any(
        #     torch.abs(self.env.get_rotation()) > self.train_cfg["max_tilt"], dim=-1)

        # # Check if the robot's movements exceed the torque limits
        is_exceeding_torque = torch.any(
            torch.abs(self.env.get_joint_torque()) > self.train_cfg["max_torque"], dim=-1)

        # return (is_collided + is_tilted + is_exceeding_torque).bool()
        return (is_collided + is_exceeding_torque).bool()
