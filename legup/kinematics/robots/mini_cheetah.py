from legup.abstract.kinematics.link_joint import Link, Joint
from legup.robot.legged_robot import LeggedRobot
from legup.abstract.spatial.spatial import Position, Direction

from typing import List

import torch


def create_mini_cheetah_leg(left: bool, front: bool,
                            device: torch.device = LeggedRobot._default_device()
                            ) -> Joint:

    direction_tensor = torch.tensor(
        [1. if front else -1., 1. if left else -1., 1.],
        device=device)

    positions_raw = torch.tensor([[0.14775, 0.049, 0],  # Hip
                                  [0.055, -0.019, 0],  # Shoulder
                                  [0, 0.049, -0.2085],  # Knee
                                  [0, 0, -0.194]],     # Foot
                                 device=device) * direction_tensor

    joint_transforms = Position(positions_raw).make_transform()

    foot_suffix = f"_{'F' if front else 'B'}{'L' if left else 'R'}"

    toe_link = Link(name='foot_link' + foot_suffix,
                    device=device)

    foot_joint = Joint.make_fixed(name='ankle' + foot_suffix,
                                  origin=joint_transforms[-1],
                                  child_link=toe_link)

    foot_link = Link(name="shank" + foot_suffix,
                     child_joints=[foot_joint])

    knee_joint = Joint.make_revolute(name="knee" + foot_suffix,
                                     origin=joint_transforms[-2],
                                     axis=Direction.from_list([0, 0, 1]),
                                     child_link=foot_link)

    thigh_link = Link(name="thigh" + foot_suffix,
                      child_joints=[knee_joint])

    shoulder_joint = Joint.make_revolute(name="shoulder" + foot_suffix,
                                         origin=joint_transforms[-3],
                                         axis=Direction.from_list([0, 1, 0]),
                                         child_link=thigh_link)

    abad_link = Link(name="abad" + foot_suffix,
                     child_joints=[shoulder_joint])

    hip_joint = Joint.make_revolute(name="hip" + foot_suffix,
                                    origin=joint_transforms[-3],
                                    axis=Direction.from_list([1, 0, 0]),
                                    child_link=abad_link)

    return hip_joint


mini_cheetah_base = Link(name="base",
                         child_joints=[create_mini_cheetah_leg(True, True),
                                       create_mini_cheetah_leg(True, False),
                                       create_mini_cheetah_leg(False, True),
                                       create_mini_cheetah_leg(False, False)])

mini_cheetah = LeggedRobot(name="mini_cheetah",
                           base_link=mini_cheetah_base,
                           primary_contacts=["foot_link_FL",
                                             "foot_link_FR",
                                             "foot_link_BL",
                                             "foot_link_BR"],
                           secondary_contacts=["thigh_FL",
                                               "thigh_FR",
                                               "thigh_BL",
                                               "thigh_BR"],
                           joint_limits={"knee_FL": 0.4,
                                         "knee_FR": 0.4,
                                         "knee_BL": 0.4,
                                         "knee_BR": 0.4},
                           # pain
                           model_idx_dict={"base": 0,
                                           "abad_FR": 1,
                                           "thigh_FR": 2,
                                           "shank_FR": 3,
                                           "foot_link_FR": 3,
                                           "abad_FL": 4,
                                           "thigh_FL": 5,
                                           "shank_FL": 6,
                                           "foot_link_FL": 6,
                                           "abad_BR": 7,
                                           "thigh_BR": 8,
                                           "shank_BR": 9,
                                           "foot_link_BR": 9,
                                           "abad_BL": 10,
                                           "thigh_BL": 11,
                                           "shank_BL": 12,
                                           "foot_link_BL": 12,
                                           "hip_FR": 0,
                                           "shoulder_FR": 1,
                                           "knee_FR": 2,
                                           "ankle_FR": 2,
                                           "hip_FL": 3,
                                           "shoulder_FL": 4,
                                           "knee_FL": 5,
                                           "ankle_FL": 5,
                                           "hip_BR": 6,
                                           "shoulder_BR": 7,
                                           "knee_BR": 8,
                                           "ankle_BR": 8,
                                           "hip_BL": 9,
                                           "shoulder_BL": 10,
                                           "knee_BL": 11,
                                           "ankle_BL": 11})
