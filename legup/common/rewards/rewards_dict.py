from typing import NamedTuple, Dict, Callable, Optional, Union, TypeVar

from legup.common.abstract_dynamics import AbstractDynamics
from legup.common.legged_robot import LeggedRobot

import torch


class RewardArgs(NamedTuple):
    dynamics: AbstractDynamics
    robot: LeggedRobot
    device: torch.device


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


RewardArgsType = TypeVar('RewardArgsType', bound=RewardArgs)

reward_func_type = Union[
    Callable[[RewardArgs], torch.Tensor],
    Callable[[RewardArgsType], torch.Tensor]]


def reward(reward_func: Optional[reward_func_type] = None, reward_name: Optional[str] = None) -> Callable:
    """
    A decorator to register custom reward functions in a global rewards dictionary.

    This decorator adds the reward function to a rewards dictionary, which can be later
    referenced when calculating rewards. To register a custom reward function, simply
    use the @reward decorator above its definition.

    To calculate rewards using the registered functions, provide the function names
    and corresponding scaling factors in the reward_scale dictionary argument of the
    calculate_reward function.

    Example:
    --------
    1. Define and register a custom reward function:

        @reward
        def my_reward(reward_args: RewardArgs) -> torch.Tensor:
            # Your custom reward calculation
            return reward_tensor

    2. Calculate rewards using the registered function:

        reward_scales = {"my_reward": scale_value, ...}
        rewards = calculate_reward(reward_args, reward_scales)

    Parameters:
    -----------
    reward_func : Callable[[RewardArgs], torch.Tensor]
        The custom reward function to be registered.

    Returns:
    --------
    Callable[[RewardArgs], torch.Tensor]
        The same reward function passed as input, for further use in the code.
    """

    if reward_name is not None:
        return lambda reward_func: _reward(reward_func, reward_name)
    elif reward_func is None:
        raise ValueError("The reward function cannot be None.")

    reward_name = reward_func.__name__

    return _reward(reward_func, reward_func.__name__)


def _reward(reward_func: reward_func_type, reward_name: str) -> Callable[[RewardArgs], torch.Tensor]:

    if reward_name in rewards_dict:
        raise ValueError(
            f"A reward called {reward_name} has been registered twice.")

    rewards_dict[reward_name] = reward_func
    return reward_func
