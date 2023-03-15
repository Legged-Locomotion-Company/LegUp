import torch

from legup.common.robot import Robot, KinematicsObject
from legup.common.link_joint import Link, Joint
from legup.common.spatial.spatial import Transform, Position, Direction


def test_planar_2r_robot_kinematics():
    """Test that the robot kinematics work for a 2r planar robot"""

    ee_link = Link("ee_link")

    fixed_wrist = Joint.make_fixed(
        name="wrist",
        origin=Position.from_list([1, 0, 0]).make_transform(),
        child_link=ee_link,
    )

    forearm_link = Link("forearm",
                        child_joints=[fixed_wrist])

    elbow_joint = Joint.make_revolute(
        name="elbow",
        origin=Position.from_list([1, 0, 0]).make_transform(),
        axis=Direction.from_list([0, 0, 1]),
        child_link=forearm_link)

    upper_arm_link = Link("upper_arm",
                          child_joints=[elbow_joint])

    shoulder_joint = Joint.make_revolute("shoulder",
                                         origin=Transform.zero(),
                                         axis=Direction.from_list([0, 0, 1]),
                                         child_link=upper_arm_link)

    base_link = Link(name="base",
                     child_joints=[shoulder_joint])

    robot = Robot("planar_2r",
                  base_link=base_link)

    kinematics = robot.make_kinematics(query_link_names=["ee_link"])

    result = kinematics(torch.tensor([0, torch.pi/2], device=robot.device))

    pass
