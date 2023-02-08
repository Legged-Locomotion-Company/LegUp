import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Type, overload
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET

import torch


class TensorWrapper:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index]

    def __setitem__(self, index, value):
        self.tensor[index] = value


class AbstractJointAngles(ABC, TensorWrapper):
    """A collection of joint angles for a robot."""

    def __init__(self, joint_angles: torch.Tensor):
        self.num_dofs = joint_angles.shape[-1]
        self.pre_shape = joint_angles.shape[:-1]
        self.joint_angles = joint_angles


def JointAngles(robot: "Robot") -> Type[AbstractJointAngles]:
    """Factory that creates joint angles class for a robot.

    Args:
        robot: The robot.

    Returns:
        A joint angles class.
    """

    class_name = robot.name + "JointAngles"

    # Create a joint angles class for the robot.
    JointAnglesClass = type(class_name, (AbstractJointAngles,), {})

    return JointAnglesClass


class Robot:
    """Abstract class for a robot. A robot is a collection of legs that does forward kinematics."""

    def __init__(self, legs: List["RobotLeg"], name: str, home_position: Optional[AbstractJointAngles] = None):
        self.legs = legs
        self.name = name
        self.home_position = home_position
        self.joint_angles_class = JointAngles(self)

    @overload
    def forward_kinematics(self, joint_angles: AbstractJointAngles, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes the forward kinematics of the robot."""
        if not isinstance(joint_angles, self.joint_angles_class):
            raise ValueError("Joint angles must be of type " +
                             str(self.joint_angles_class))

    @overload
    def forward_kinematics(self, joint_angles_tensor: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes the forward kinematics of the robot."""
        joint_angles = self.joint_angles_class(joint_angles_tensor)

        for leg in self.legs:
            leg.forward_kinematics(joint_angles, out=out)


class RobotLeg:
    def __init__(self, joints: List["RobotJoint"], name: str):
        self.joints = joints
        self.name = name


class Screw(TensorWrapper):
    def __init__(self, screw_vec: torch.Tensor):
        if screw_vec.shape[-1] != (6,):
            raise ValueError("Screw vector must be of shape (6,).")
        self.pre_shape = screw_vec.shape[:-1]
        super().__init__(screw_vec)


def empty_screw(*shape, device):
    screw_tensor = torch.empty((*shape, 6), device=device)
    return Screw(screw_tensor)


class RobotJoint(ABC):
    def __init__(self, screw: Screw):
        self.screw = screw


class RevoluteJoint(RobotJoint):
    def __init__(self, origin: "Transform", axis: "Direction"):
        """Creates a revolute joint.

        Args:
            origin: The origin of the joint.
            axis: The axis of the joint.

        """

        torch.broadcast_shapes(origin.pre_shape, axis.pre_shape)

        screw_axis = empty_screw((*origin.pre_shape, 6), device=origin.device)

        screw_axis[..., :3] = axis.tensor
        screw_axis[...,
                   3:] = torch.cross(-axis.tensor, origin.tensor[..., :3, 3])

        super().__init__(screw_axis)


class Direction(TensorWrapper):
    def __init__(self, direction_vec: torch.Tensor):
        if direction_vec.shape[-1] != (3,):
            raise ValueError("Direction vector must be of shape (3,).")
        self.pre_shape = direction_vec.shape[:-1]
        super().__init__(direction_vec)


class Transform(TensorWrapper):
    def __init__(self, transform: torch.Tensor):
        if transform.shape[-2:] != (4, 4):
            raise ValueError("Transform matrix must be of shape (4, 4).")
        self.pre_shape = transform.shape[:-2]
        self.device = transform.device
        self.tensor = transform

    def compose(self, other: "Transform", out: Optional["Transform"]) -> "Transform":
        """Composes two transforms.
        """

        if out is None:
            out_shape = torch.broadcast_shapes(
                self.tensor.shape, other.tensor.shape)
            out = Transform(torch.empty(out_shape, device=self.device))

        torch.matmul(self.tensor, other.tensor, out=out.tensor)

        return out


def create_robot_from_urdf(filepath: str) -> Robot:
    tree = ET.parse(filepath)
    root = tree.getroot()

    if root.tag != "robot":
        raise ValueError("URDF root element must be robot.")
    elif "name" not in root.attrib:
        raise ValueError("URDF robot element must have a name attribute.")

    robot_name = root.attrib["name"]

    # Get all joints and links.
    joints = {joint_name: joint for joint_name, joint in root.iter("joint")}
    links = {link_name: link for link_name, link in root.iter("link")}

    # Create a dictionary of parent link names to child joint names.
    parent_link_to_child_joint = {}
    for joint_name, joint in joints.items():
        parent_link_name = joint.find("parent").attrib["parent"]

        parent_link_name.
        parent_link_to_child_joint[parent_link_name] = joint_name

    # Create a dictionary of parent joint names to child link names.
    parent_joint_to_child_link = {}

    # Find the root link
    root_links = [(link_name, link) for link_name,
                 link in links.items() if link.find("parent") is None]

    # Make sure there is only one root link
    if len(root_links) != 1:
        raise ValueError("URDF must have exactly one root link.")

    # Grab the root link
    root_link = root_links[0]

    def link_name_is_leg_root(link_name: str):
        return link_name not in parent_link_to_child_joint

    # find leg names
    leg_names = [link_name for link_name, link in links.items() if link.find(
        "parent") is not None and link.find("parent").attrib["link"] == root_link[0]]
