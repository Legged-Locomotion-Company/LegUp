import torch
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Any
import warnings

from omegaconf import DictConfig

from legup.common.abstract_dynamics import AbstractDynamics
from legup.common.abstract_terrain import AbstractTerrain
from legup.common.robot import Robot


class AbstractAgent(ABC):
    """An abstract class for agents to inherit."""

    def __init__(self,
                 config: DictConfig,
                 robot: Robot,
                 dynamics: AbstractDynamics,
                 num_agents: int,
                 device: Optional[torch.device] = None):

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            warnings.warn(
                f"No device specified, agent decided to use: {device}")

        self.robot = robot
        self.config = config
        self.device = device

        self.num_agents = num_agents
        self.ep_lens = torch.zeros(
            self.num_agents, device=self.device, dtype=torch.int16)

        self.dynamics = dynamics

    @abstractmethod
    def make_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Constructs actions from raw model output

        Args:
            actions (torch.Tensor): Raw model output, shape `(num_agents, action_space)`

        Returns:
            torch.Tensor: DOF positions, shape `(num_agents, num_dof)`
        """

        pass

    @ abstractmethod
    def reset_agents(self, terminated_agents: torch.Tensor) -> None:
        """Called right after reseting any terminated agents, use this to reset any local buffers and resample new commands

        Args:
            terminated_agents (torch.Tensor): Truthy (bool) tensor of shape `(num_agents)`. 1 means the agent at that index was just reset
        """
        pass

    @ abstractmethod
    def post_physics_step(self) -> None:
        """Called right after stepping in the environment, use this to update any local buffers"""
        pass

    @ abstractmethod
    def make_observation(self, dynamics: AbstractDynamics) -> torch.Tensor:
        """Called to create a new observation from the environment

        Args:
            dynamics (AbstractDynamics): provides getters for agent kinematics

        Returns:
            torch.Tensor: Observation tensor of shape `(num_agents, obs_space)`
        """
        pass

    @ abstractmethod
    def make_reward(self, dynamics: AbstractDynamics) -> Tuple[torch.Tensor, dict]:
        """Called to calculate rewards in the new environment

        Args:
            dynamics (AbstractDynamics): provides getters for agent kinematics

        Returns:
            torch.Tensor: Reward tensor of shape `(num_agents)`
        """
        pass

    @ abstractmethod
    def find_terminated(self, dynamics: AbstractDynamics) -> torch.Tensor:
        """Called when the environment needs to check which agents have terminated

        Args:
            dynamics (AbstractDynamics): provides getters for agent kinematics

        Returns:
            torch.Tensor: returns which agents have terminated in the environment by either:
            - A) the agent indices that terminated
            - B) a truthy boolean tensor where 1 means an agent has terminated
        """
        pass

    @ abstractmethod
    def sample_new_position(self, num_positions: int, pos_lower: Tuple[int, int, int], pos_upper: Tuple[int, int, int]) -> torch.Tensor:
        """Called when the environment needs to sample new positions for robot root

        Args:
            num_positions (int): number of positions to sample
            pos_lower (Tuple[int, int, int]): lower bound of space that can be sampled
            pos_upper (Tuple[int, int, int]): upper bound of space that can be sampled

        Returns:
            torch.Tensor: returns sampled positions of shape `(num_agents, 3)`
        """
        pass

    @ abstractmethod
    def sample_new_quaternion(self, num_quats: int) -> torch.Tensor:
        """Called when the environment needs to sample new quaternions for robot root

        Args:
            num_quats (int): number of quaternions to sample

        Returns:
            torch.Tensor: returns sampled quaternions of shape `(num_agents, 4)`
        """
        pass

    @ abstractmethod
    def sample_new_joint_pos(self, num_pos: int) -> torch.Tensor:
        """Called when the environment needs to sample new robot joint positions

        Args:
            num_pos (int): number of positions to sample

        Returns:
            torch.Tensor: new joint positions of shape `(num_pos, num_dof)`
        """

    def generate_new_terrain(self, num_terrains: int) -> Optional[List[AbstractTerrain]]:
        """Called after every rollout to check if the environment should change its terrain

        Args:
            num_terrains (int): Number of terrains to create

        Returns:
            Optional[List[Any]]: None if no change, new terrain configurations otherwise
        """
        return None
