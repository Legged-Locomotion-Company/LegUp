import torch

from typing import Optional

from legup.common.abstract_agent import AbstractAgent, StepResult


class WildAnymalAgent(AbstractAgent):
    def __init__(self, config, robot, device: Optional[torch.device] = None):
        # TODO: add type hints here once the types are implemented
        # TODO: figure out how to get the device
        if device is None:
            device = robot.device
        elif device != robot.device:
            raise ValueError(f"Robot device {robot.device} does not match agent device {device}!")

        self.robot = robot
        self.config = config
        self.device = device

    def step(self, obs: torch.Tensor):
        action = torch.zeros(self.config.num_envs, self.robot.num_dofs, dtype=torch.float32, device=self.device)
        dones = torch.zeros(self.config.num_envs)
        return StepResult(action=action, dones=dones, infos={})
