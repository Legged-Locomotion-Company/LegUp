from isaacgym import gymtorch

from legup.isaac.environment import IsaacGymEnvironment
from legup.robots.Robot import Robot

from typing import Union, List, Tuple

import gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

import torch


class BaseAgent(VecEnv):
    """Default agent that other agents will inherit from. To extend this to a new agent, you must implement the following functions:
    - post_physics_step (not necessary)
    - check_termination
    - make_actions (not necessary)
    - make_observation
    - make_reward
    - reset_envs (not necessary)

    Read the documentation for each of these functions to learn how to implement them!
    """

    def __init__(self, robot: Robot, num_environments: int, curriculum_exponent: int, asset_path: str, asset_name: str):
        """
        Args:
            num_environments (int): number of parallel environments to create in simulator
            asset_path (str): path to URDF file from `asset_root`
            asset_root (str): root folder where the URDF to load is
        """
        self.device = robot.device

        self.all_envs = list(range(num_environments))
        self.num_envs = num_environments
        self.term_idx = self.all_envs.copy()

        self.ep_lens = torch.zeros(num_environments).to(self.device)

        self.env = IsaacGymEnvironment(
            num_environments, True, asset_path, asset_name, robot.home_position)
        self.robot = robot

        self.dt = 1. / 60.  # TODO: make this config
        self.max_ep_len = 1000 / self.dt  # TODO: make this config

        self.curriculum_factor = 10**(-curriculum_exponent)
        # self.env_curriculum_factor
        # self.curriculum_exponent = curriculum_exponent

        # domain randomization parameters

        # linear velocity command limits in meters per second
        self.command_mag_lower = torch.tensor(0., device=self.device)
        self.command_mag_upper = torch.tensor(2., device=self.device)

        # angular velocity command limits in radians per second
        self.command_ang_vel_lower = torch.tensor(-1., device=self.device)
        self.command_ang_vel_upper = torch.tensor(1., device=self.device)

        # angle limits for robot linear command wrt the robot frame
        self.command_ang_lower = torch.tensor(-torch.pi, device=self.device)
        self.command_ang_upper = torch.tensor(torch.pi, device=self.device)

        self.commands = torch.zeros((self.num_envs, 3), device=self.device)

        self.obs_noise_mean = 0
        # self.obs_noise_var = 0.1
        self.obs_noise_var = 0.0

        self.should_push = True
        self.push_prob = 1./1024.
        self.push_mag_upper_max = 0.5
        self.push_mag_lower_max = 0.25

        self.push_mag_upper = 0.0
        self.push_mag_lower = 0.0

        self.push_vel_upper = torch.tensor(
            [0.5, 0.5, 0.1], device=self.device, dtype=torch.float)
        self.push_idx = torch.zeros(self.num_envs, device=self.device)

        # OpenAI Gym Environment required fields
        # TODO: custom observation space/action space bounds, this would help with clipping!
        self.observation_space = gym.spaces.Box(low=np.ones(
            391) * -10000000, high=np.ones(391) * 10000000, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.ones(
            12) * -10000000, high=np.ones(12) * 10000000, dtype=np.float32)
        self.metadata = {"render_modes": ['rgb_array']}
        self.reward_range = (-float("inf"), float("inf"))
        self.spec = None

    def make_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Converts the raw model output into joint target positions. If this isn't implemented, the raw network output is sent to the environment

        Args:
            actions (torch.Tensor): shape `(num_envs, num_dof)`

        Returns:
            torch.Tensor: joint position targets of shape `(num_envs, num_dof)`
        """
        return actions

    def make_observation(self) -> torch.Tensor:
        """Computes a new observation from the environment

        Raises:
            NotImplementedError: throws exception if it isn't implemented

        Returns:
            torch.Tensor: new observations of shape `(num_envs, observation_space)`
        """
        raise NotImplementedError("agent::make_observation not implemented!")

    def make_reward(self, actions: torch.Tensor) -> torch.Tensor:
        """Computes a new reward based on the environment

        Args:
            actions (torch.Tensor): actions that were sent to environment of shape `(num_envs, action_space)`

        Raises:
            NotImplementedError: throws exception if it isn't implemented

        Returns:
            torch.Tensor: agent rewards of shape `(num_envs)`
        """
        raise NotImplementedError("agent::make_reward not implemented!")

    def post_physics_step(self):
        """Called by agent after the environment is updated every step, used for tracking any agent-specific information. Nothing happens if not implemented
        """
        pass

    def check_termination(self) -> List[int]:
        """Checks if any agents have satisfied your termination conditions. Must return a truthy tensor of shape `(num_envs)`

        Raises:
            NotImplementedError: throws exception if it isn't implemented

        Returns:
            List[int]: truthy tensor of shape `(num_envs)` that explains which agents/environments have terminated
        """
        raise NotImplementedError("agent::check_termination not implemented!")

    def reset_envs(self, idxs: Union[torch.Tensor, List[int], int]):
        """This is called right before any of the environments are reset, use it to reset any information you are tracking. Nothing happens if not implemented.

        Args:
            idxs (Union[torch.Tensor, List[int], int]): idxs of environments that are being reset
        """
        pass

    def get_termination_list(self, reward: torch.Tensor) -> List[int]:
        """Gets list of environment idxs that have either terminated or truncated

        Args:
            reward (torch.Tensor): reward buffer of shape `(num_envs)`, used to penalize for terminated environments

        Returns:
            List[int]: list of environment idxs that terminated
        """
        term = self.check_termination()
        trunc = self.ep_lens > self.max_ep_len

        # reward[torch.where(term)] = -10
        # reset_idx = torch.logical_or(term, trunc)
        # return reset_idx.long().tolist()

        # TODO: can probably make this faster
        reset_idx = set()
        for term_idx in torch.where(trunc)[0]:
            reset_idx.add(term_idx.item())

        for term_idx in torch.where(term)[0]:
            reward[term_idx] = -20  # TODO: add a config for this
            reset_idx.add(term_idx.item())

        return list(reset_idx)

    def add_noise(self, tensor, noise_mean, noise_var):
        if isinstance(tensor, np.ndarray):
            # why are we returning numpy, keeping this here for a very short time because will refactor to only return torch soon
            return tensor + (np.random.randn(*tensor.shape) * np.sqrt(noise_var) + noise_mean)

        return tensor + (torch.randn_like(tensor) * np.sqrt(noise_var) + noise_mean)

    def create_random_commands(self, count: int) -> torch.Tensor:
        """Randomizes the commands for the agents

        Args:
            idxs (Union[torch.Tensor, List[int], int]): idxs of environments whose commands should be randomized
        """
        result = torch.zeros((count, 3), device=self.device)

        ang_range = (self.command_ang_upper -
                     self.command_ang_lower) * self.curriculum_factor
        ang_avg = (self.command_ang_upper + self.command_ang_lower) / 2
        ang_upper = ang_avg + ang_range / 2
        ang_lower = ang_avg - ang_range / 2

        mag_range = (self.command_mag_upper -
                     self.command_mag_lower) * self.curriculum_factor
        mag_avg = (self.command_mag_upper + self.command_mag_lower) / 2
        mag_upper = mag_avg + mag_range / 2
        mag_lower = mag_avg - mag_range / 2

        ang_vel_range = (self.command_ang_vel_upper -
                         self.command_ang_vel_lower) * self.curriculum_factor
        ang_vel_avg = (self.command_ang_vel_upper +
                       self.command_ang_vel_lower) / 2
        ang_vel_upper = ang_vel_avg + ang_vel_range / 2
        ang_vel_lower = ang_vel_avg - ang_vel_range / 2

        # use idx 0 as scratch space
        command_angles_scratch = result[:, 0]
        # generate random angle commands and write into scratch space
        command_angles_scratch.uniform_(ang_lower, ang_upper)
        # write cos and sin of angle commands into commands tensor
        torch.cos(command_angles_scratch, out=result[:, 1])
        torch.sin(command_angles_scratch, out=result[:, 2])
        # generate random magnitude commands and write into scratch space
        command_angles_scratch.uniform_(mag_lower, mag_upper)
        # multiply cos and sin by magnitude commands and write into commands tensor
        result[:, 1] *= command_angles_scratch
        result[:, 2] *= command_angles_scratch

        # use idx 0 as scratch space
        command_ang_vel_scratch = result[:, 0]
        # generate random anglular velocity commands
        command_ang_vel_scratch.uniform_(ang_vel_lower, ang_vel_upper)
        # write scratch into tensor (they are the same but I think this is clearer)
        result[:, 0] = command_ang_vel_scratch

        # now make some of the commands zero

        # make 4% of resets a stand still command (all zeros)
        stand_still_command = torch.rand(count, device=self.device) < 0.04
        result[stand_still_command, :] = 0

        # make 3% of resets a no turn command (command[:, 0] = 0)
        zero_turn_command = torch.rand(count, device=self.device) < 0.03
        result[zero_turn_command, 0] = 0

        # make 3% of resets an only turn command (command:, 1:] = 0)
        only_turn_command = torch.rand(count, device=self.device) < 0.03
        result[only_turn_command, 1:] = 0

        return result

    def reset_partial(self) -> List[int]:
        """Resets a subset of all environments, if they need to be reset

        Returns:
            List[int]: idxs of environments that were reset
        """

        done_idxs = torch.tensor(self.term_idx, dtype=torch.long)

        if len(done_idxs) > 0:
            self.ep_lens[self.term_idx] = 0
            self.reset_envs(self.term_idx)
            self.env.reset(self.term_idx)

            self.commands[self.term_idx] = self.create_random_commands(
                len(done_idxs))

        dones = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        dones[done_idxs] = True

        self.term_idx.clear()
        return dones, list(done_idxs)

    def reset(self):
        """Resets all environments, should only be called once at the beginning of training. Undefined behavior if not called only once at the beginning of training

        Returns:
            torch.Tensor: new observations after reset, shape `(num_envs, observation_space)`
        """

        self.reset_partial()
        self.env.step(None)
        self.env.refresh_buffers()

        return self.add_noise(self.make_observation(), self.obs_noise_mean, self.obs_noise_var)

    def make_logs(self) -> dict:
        return {}

    def step_curriculum(self):
        """Overwrite this function in the child class to implement curriculum learning. This function is called at the end of every episode"""
        self.curriculum_factor **= 1-10**(-self.curriculum_exponent)
        self.push_mag_upper = self.push_mag_upper * self.curriculum_factor
        self.push_mag_lower = self.push_mag_lower * self.curriculum_factor

    def generate_pushes(self, count: int) -> torch.Tensor:
        out = torch.zeros((int(count), 3), device=self.device)

        out[:, 2].uniform_(0, 2*torch.pi)
        torch.cos(out[:, 2], out=out[:, 0])
        torch.sin(out[:, 2], out=out[:, 1])
        out[:, 2].uniform_(self.push_mag_lower, self.push_mag_upper)
        out[:, 0:2].mul_(out[:, 2].unsqueeze(1))
        out[:, 2].zero_()

        return out

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Does a single step of the simulation environment based on a given command, and computes new state information and rewards

        Args:
            actions (torch.Tensor): Commanded joint position, shape `(num_envs, num_degrees_of_freedom)`

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]: new observation, corresponding reward, truthy tensor of which
            environments have terminated/truncated, and additional metadata (currently empty). Observation tensor has shape `(num_envs, observation_space)`
            and all other tensors have shape `(num_envs)`
        """

        # self.curriculum_factor **= 1-10**(-self.curriculum_exponent)
        self.step_curriculum()
        # send actions through the network
        reward, reward_keys, reward_vals = self.make_reward(actions)

        actions = self.make_actions(actions)
        self.env.step(actions)

        # reset any terminated environments and update buffers
        dones, done_idxs = self.reset_partial()

        # TODO: move this into reset, refactor coming soon so it wont be here for long
        idxs = torch.tensor(done_idxs + self.push_idx.argwhere().tolist(),
                            device=self.device).int()
        if len(idxs) > 0:
            indices = gymtorch.unwrap_tensor(idxs)
            self.env.gym.set_actor_root_state_tensor_indexed(
                self.env.sim,  gymtorch.unwrap_tensor(self.env.root_states), indices, len(idxs))

        self.env.refresh_buffers()
        self.post_physics_step()

        # compute new observations and rewards
        new_obs = self.add_noise(
            self.make_observation(), self.obs_noise_mean, self.obs_noise_var)

        logs = self.make_logs()

        # update tracking info (episodes done, terminated environments)
        self.ep_lens += 1
        self.term_idx = self.get_termination_list(reward)

        self.push_idx = torch.rand_like(self.push_idx.float()) < self.push_prob

        push_count = self.push_idx.sum()

        if self.should_push and push_count > 0:
            pushes = self.generate_pushes(push_count.item())

            self.env.root_lin_vel[self.push_idx] += pushes

        # TODO: add specific reward information
        infos = [{}] * self.num_envs
        # reward_keys.append('total_reward')
        # reward_vals.append(sum(reward_vals))
        infos[0] = {'reward_names': reward_keys,
                    'reward_terms': reward_vals, 'logs': logs}
        return new_obs, reward, dones, infos

    def render(self) -> torch.Tensor:
        """Gets a screenshot from simulation environment as torch tensor

        Returns:
            torch.Tensor: RGBA screenshot from sim, shape `(height, width, 4)`
        """
        return self.env.render()

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Added because it is required as per the OpenAI gym specification"""
        return [False]

    def close(self):
        """Added because it is required as per the OpenAI gym specification"""
        raise NotImplementedError

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Added because it is required as per the OpenAI gym specification"""
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        """Added because it is required as per the OpenAI gym specification"""
        raise NotImplementedError

    def seed(self, seed=None):
        """Added because it is required as per the OpenAI gym specification"""
        raise NotImplementedError

    def set_attr(self, attr_name, value, indices=None):
        """Added because it is required as per the OpenAI gym specification"""
        raise NotImplementedError

    def step_async(self, actions):
        """Added because it is required as per the OpenAI gym specification"""
        raise NotImplementedError

    def step_wait(self):
        """Added because it is required as per the OpenAI gym specification"""
        raise NotImplementedError
