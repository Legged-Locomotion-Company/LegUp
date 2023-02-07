from abc import ABC, abstractmethod

import gym
import torch
import numpy as np
from typing import Optional, List

class AbstractEnv:
    @abstractmethod
    def step(self, actions: Optional[torch.Tensor] = None):
        """Moves robots using `actions`, steps the simulation forward, updates graphics, and refreshes state tensors
        Args:
            actions (torch.Tensor, optional): target joint positions to command each robot, shape `(num_environments, num_degrees_of_freedom)`. 
                If none, robots are commanded to the default joint position provided earlier Defaults to None.
        """
        pass

    @abstractmethod
    def render(self) -> torch.Tensor:
        """Gets an image of the environment from the camera and returns it
        Returns:
            np.ndarray: RGB image, shape `(camera_height, camera_width, 4)`
        """
        pass

    @abstractmethod
    def reset(self, env_index: Optional[List[int]] = None):
        """Resets the specified robot. Specifically, it will move it to a random position, give it zero velocity, and drop it from a height of 0.28 meters.
        Args:
            env_index (list, torch.Tensor, optional): Indices of environments to reset. If none, all environments are reset. Defaults to None.
        """
        pass