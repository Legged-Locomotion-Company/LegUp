import torch

from typing import Optional, Tuple

from omegaconf import DictConfig

from legup.agents import AbstractAgent
from legup.abstract.abstract_dynamics import AbstractDynamics
from legup.robot.legged_robot import LeggedRobot
from legup.rewards import calculate_reward, RewardArgs
from legup.agents.wild_anymal.wild_anymal_config import WildAnymalConfig


class WildAnymalAgent(AbstractAgent):
    def __init__(self,
                 config: WildAnymalConfig,
                 robot: LeggedRobot,
                 device: torch.device,
                 **kwargs):
        
        # TODO: add type hints here once the types are implemented
        # TODO: figure out how to get the device

        # TODO: add reward function shit here
        
        self.robot = robot
        self.device = device
        self.config = config

        print("wild anymal created")


    def make_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return super().make_actions(actions)

    def reset_history_vec(self, idx=None):
        # 3 timesteps for history, 2 for velocity

        if idx is not None:
            self.joint_pos_history[idx] = torch.zeros(
                self.robot.num_dofs, 3, device=self.device)
            self.joint_vel_history[idx] = torch.zeros(
                self.robot.num_dofs, 2, device=self.device)
            self.joint_target_history[idx] = torch.zeros(
                self.robot.num_dofs, 2, device=self.device)

        else:
            self.joint_pos_history = torch.zeros(
                self.num_agents, self.robot.num_dofs, 3, device=self.device)
            self.joint_vel_history = torch.zeros(
                self.num_agents, self.robot.num_dofs, 2, device=self.device)
            self.joint_target_history = torch.zeros(
                self.num_agents, self.robot.num_dofs, 2, device=self.device)

    def phase_gen(self):
        cpg_freq = 4.0
        base_frequencies = torch.tensor([cpg_freq] * 4).to(self.device)
        phase_offsets = torch.tensor(
            [0, torch.pi, torch.pi, 0]).to(self.device)

        phase = (self.ep_lens * self.dynamics.get_dt()).expand(4, self.num_agents).T * \
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

    def reset_agents(self, terminated_agents: torch.Tensor) -> None:
        self.reset_history_vec(terminated_agents)
        self.ep_lens[terminated_agents] = 0

    def post_physics_step(self) -> None:
        self.ep_lens += 1        

    def make_observation(self, dynamics: AbstractDynamics) -> torch.Tensor:
        proprio = torch.zeros(self.num_agents, 133).to(self.device)
        extro = torch.zeros(self.num_agents, 208).to(self.device)
        privil = torch.zeros(self.num_agents, 50).to(self.device)

        proprio[: , :3] = self.command

        proprio[:, 3:6] = dynamics.get_position()
        proprio[:, 6:9] = dynamics.get_linear_velocity()

        proprio[:, 9:12] = dynamics.get_angular_velocity()
        proprio[:, 12:24] = dynamics.get_joint_position()
        proprio[:, 24:36] = dynamics.get_joint_velocity()

        proprio[:, 36:72] = self.joint_pos_history.flatten(start_dim=1)
        proprio[:, 72:96] = self.joint_vel_history.flatten(start_dim=1)
        proprio[:, 96:120] = self.joint_target_history.flatten(
            start_dim=1)

        proprio[:, 120:133] = self.make_phase_observation()

        privil[:, :4] = dynamics.get_contact_states(
        )[:, self.robot.get_foot_link_indices()].to(torch.float)

        privil[:, 4:16] = dynamics.get_contact_forces(
        )[:, self.robot.foot_link_indices, :].flatten(start_dim=1)
        # privil[idx, 16:28] = self.env.get_contact_normals()
        # privil[idx, 28:32] = self.enc.get_frivtion_coeffs()
        privil[:, 32:40] = dynamics.get_contact_states()[
            :, self.robot.get_shank_link_indices() + self.robot.get_thigh_link_indices()].to(torch.float)

        # TODO: add airtime

        self.joint_pos_history[:, :, 1:] = self.joint_pos_history[:, :, :2]
        self.joint_pos_history[:, :, 0] = dynamics.get_joint_position()

        self.joint_vel_history[:, :, 1] = self.joint_vel_history[:, :, 0]
        self.joint_vel_history[:, :, 0] = dynamics.get_joint_velocity()

        self.joint_target_history[:, :,
                                  1] = self.joint_target_history[:, :, 0]
        self.joint_target_history[:, :, 0] = dynamics.get_joint_position()

        return (torch.cat([proprio, extro, privil], dim=1))


    def make_reward(self, dynamics: AbstractDynamics) -> Tuple[torch.Tensor, dict]:
        reward_args = RewardArgs()
        rewards_dict = 
            self.reward_fn.update_rewards(self.command,
                                          [
                                            Rewards.ang_velocity,
                                            Rewards.lin_velocity,
                                            Rewards.foot_clearance,
                                          ])
        
        rewards = torch.stack(list(rewards_dict.values()), dim=0).sum(dim=-1)
        return rewards, rewards_dict

    def find_terminated(self, dynamics: AbstractDynamics) -> torch.Tensor:
         # Check if any rigidbodies are hitting the ground
        # 0 = rb index of body
        is_collided = torch.any(dynamics.get_contact_states()[
                                :, [0, 2, 5, 8, 11]], dim=-1)

        # # Check if the robot is tilted too much
        # is_tilted = torch.any(
        #     torch.abs(self.env.get_rotation()) > self.train_cfg["max_tilt"], dim=-1)

        # # Check if the robot's movements exceed the torque limits
        is_exceeding_torque = torch.any(
            torch.abs(dynamics.get_joint_torque()) > self.config.agent["max_torque"], dim=-1)

        # return (is_collided + is_tilted + is_exceeding_torque).bool()
        return (is_collided + is_exceeding_torque).bool()

    def sample_new_position(self, num_positions: int, pos_lower: Tuple[int, int, int], pos_upper: Tuple[int, int, int]) -> torch.Tensor:
        return super().sample_new_position(num_positions, pos_lower, pos_upper)

    def sample_new_quaternion(self, num_quaternions: int) -> torch.Tensor:
        return super().sample_new_quaternion(num_quaternions)

    def sample_new_joint_pos(self, num_pos: int) -> torch.Tensor:
        return super().sample_new_joint_pos(num_pos)
    