from legup.common.robot import Robot, RobotLeg, RobotJoint, RevoluteJoint
from legup.common.spatial import Position, Direction

from typing import List

import torch

front_right_leg = RobotLeg(
    joints=[
        # This is the hip joint.
        RevoluteJoint(
            origin=Position(torch.tensor([0.14775, 0.049, 0])),
            axis=Direction(torch.tensor([0, 0, 1]))),
        # This is the shoulder joint.
        RevoluteJoint(
            origin=Position(torch.tensor([0.055, 0.019, 0])),
            axis=Direction(torch.tensor([0, 1, 0]))),
        # This is the knee joint.
        RevoluteJoint(
            origin=Position(torch.tensor([0, 0.049, -0.2085])),
            axis=Direction(torch.tensor([0, 0, 1]))),
    ],
    name='front_right_leg',
)

# mini_cheetah = Robot(
