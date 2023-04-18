from typing import NamedTuple

from legup.common.abstract_dynamics import AbstractDynamics
from legup.common.legged_robot import LeggedRobot


class RewardArgs(NamedTuple):
    dynamics: AbstractDynamics
    robot: LeggedRobot
