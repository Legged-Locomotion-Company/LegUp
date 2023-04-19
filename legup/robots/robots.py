from typing import Dict

from legup.common.legged_robot import LeggedRobot

robots_dict: Dict[str, LeggedRobot] = {}


def register_robot(robot: LeggedRobot) -> None:
    """Register a robot to the robot registry"""
    robots_dict[robot.name] = robot


def get_robot(name: str) -> LeggedRobot:
    """Get a robot from the robot registry"""
    return robots_dict[name]
