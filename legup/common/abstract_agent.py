import torch

from typing import Dict, Any

from abc import ABC, abstractmethod

class StepResult:
    """A class to store ther result of a step in the environment.
    This is a very thin class which only stores the action, dones and infos for type checking reasons."""
    def __init__(self, action: torch.Tensor, dones: torch.Tensor, infos: Dict[str, Any]):
        """Initailize the StepResult."""
        self.action = action
        self.dones = dones
        self.infos = infos

class AbstractAgent(ABC):
    """An abstract class for agents to inherit."""
    @abstractmethod
    def step(self, observation: torch.Tensor) -> StepResult:
        """Take a step in the environment."""
        pass