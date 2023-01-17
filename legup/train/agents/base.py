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

    def __init__(self, robot: Robot, num_environments: int, asset_path: str, asset_name: str):
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
            reward[term_idx] = -10  # TODO: add a config for this
            reset_idx.add(term_idx.item())

        return list(reset_idx)

    def reset_partial(self) -> List[int]:
        """Resets a subset of all environments, if they need to be reset

        Returns:
            List[int]: idxs of environments that were reset
        """

        done_idxs = np.array(self.term_idx, dtype=np.int32)

        if len(done_idxs) > 0:
            self.reset_envs(self.term_idx)
            self.env.reset(self.term_idx)

        dones = np.zeros(self.num_envs, dtype=np.bool)

        dones[done_idxs] = True

        self.term_idx.clear()
        return dones

    def reset(self):
        """Resets all environments, should only be called once at the beginning of training. Undefined behavior if not called only once at the beginning of training

        Returns:
            torch.Tensor: new observations after reset, shape `(num_envs, observation_space)`
        """
        self.env.step(None)
        self.reset_partial()
        self.env.refresh_buffers()

        return self.make_observation()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Does a single step of the simulation environment based on a given command, and computes new state information and rewards

        Args:
            actions (torch.Tensor): Commanded joint position, shape `(num_envs, num_degrees_of_freedom)`

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]: new observation, corresponding reward, truthy tensor of which 
            environments have terminated/truncated, and additional metadata (currently empty). Observation tensor has shape `(num_envs, observation_space)`
            and all other tensors have shape `(num_envs)`
        """
        # send actions through the network
        actions = self.make_actions(actions)
        self.env.step(actions)

        # reset any terminated environments and update buffers
        dones = self.reset_partial()
        self.env.refresh_buffers()
        self.post_physics_step()

        # compute new observations and rewards
        new_obs = self.make_observation()
        reward = self.make_reward(actions)

        # update tracking info (episodes done, terminated environments)
        self.ep_lens += 1
        self.term_idx = self.get_termination_list(reward)

        # TODO: add specific reward information
        infos = [{} for _ in range(self.num_envs)]
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
