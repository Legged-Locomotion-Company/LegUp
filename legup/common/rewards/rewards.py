from typing import NamedTuple, Dict, Callable

from legup.common.abstract_dynamics import AbstractDynamics
from legup.common.legged_robot import LeggedRobot

import torch


class RewardArgs(NamedTuple):
    dynamics: AbstractDynamics
    robot: LeggedRobot


rewards_dict: Dict[str, Callable[[RewardArgs], torch.Tensor]] = {}


def calculate_reward(reward_args: RewardArgs, reward_scale: Dict[str, float]):
    """This function calculates reward functions from the reward scale

    Args:
        reward_args (RewardArgs): _description_
        reward_scale (Dict[str, float]): _description_

    Returns:
        _type_: _description_
    """

    return {name: reward_func(reward_args)
            for name, reward_func in rewards_dict.items()}


def reward(reward_func: Callable[[RewardArgs], torch.Tensor]) -> Callable[[RewardArgs], torch.Tensor]:
    """Decorarator for reward functions. Will register with the name of the function"""

    rewards_dict[reward_func.__name__] = reward_func
    return reward_func
