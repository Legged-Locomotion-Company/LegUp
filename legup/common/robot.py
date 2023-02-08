from tensor_wrapper import TensorWrapper
from spatial import Transform, Screw, Position, Direction

from typing import List, Optional, overload, Union, Dict
from abc import ABC, abstractmethod

import xml.etree.ElementTree as ET

import torch


class RobotJointAngles(TensorWrapper):
    def __init__(self, robot: "Robot", joint_angles_tensor: torch.Tensor):
        """Factory that creates joint angles class for a robot.

        Args:
            robot: The robot.

        Returns:
            A joint angles class.
        """

        self.robot_name = robot.name
        if joint_angles_tensor.shape[-1] != robot.num_dofs:
            raise ValueError(
                f"Joint angles must be of shape (..., {robot.num_dofs}).")

        super().__init__(joint_angles_tensor, end_dims=1)

    def for_robot(self, robot: "Robot"):
        return robot.name == self.robot_name

    def assert_for_robot(self, robot: "Robot"):
        """This function throws an error when the joint angles are not for the given robot."""
        if not self.for_robot(robot):
            raise ValueError(
                f"Joint angles for {self.robot_name} used for {robot.name}.")


class Robot:
    """Abstract class for a robot. A robot is a collection of legs that does forward kinematics."""

    def __init__(self, legs: List["RobotLeg"], name: str, home_position: Optional[Union[RobotJointAngles, torch.Tensor]] = None, device: torch.device = torch.device("cpu")):
        self.legs = [leg.to(device) for leg in legs]
        self.name = name
        self.num_dofs = sum([leg.num_dofs for leg in legs])
        if home_position is None:
            home_position = RobotJointAngles(
                self, torch.zeros(self.num_dofs, device=device))
        else:
            home_position = home_position.to(device)
        self.home_position = home_position

    def create_joint_angles(self, joint_angles_tensor: torch.Tensor) -> RobotJointAngles:
        return RobotJointAngles(self, joint_angles_tensor)

    @overload
    def assign_home_position(self, home_position: RobotJointAngles):
        """Assigns a home position to the robot.

        Args:
            home_position (RobotJointAngles): The home position.
        """

        home_position.assert_for_robot(self)

        self.home_position = home_position

    @overload
    def assign_home_position(self, home_position: torch.Tensor):
        """Assigns a home position to the robot.

        Args:
            home_position: The home position.
        """

        home_position = self.create_joint_angles(home_position)

        self.home_position = home_position

    @overload
    def forward_kinematics(self, joint_angles: RobotJointAngles, out: Optional[Position] = None) -> Position:
        """Computes the forward kinematics of the robot."""
        joint_angles.assert_for_robot(self)

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


class RobotJoint(ABC):
    def __init__(self, screw: Screw):
        self.screw = screw


class RevoluteJoint(RobotJoint):
    def __init__(self, origin: Transform, axis: Direction):
        """Creates a revolute joint.

        Args:
            origin: The origin of the joint.
            axis: The axis of the joint.

        """

        torch.broadcast_shapes(origin.pre_shape, axis.pre_shape)

        screw_axis = Screw.empty_screw(
            (*origin.pre_shape, 6), device=origin.device)

        screw_axis[..., :3] = axis.tensor
        screw_axis[...,
                   3:] = torch.cross(-axis.tensor, origin.tensor[..., :3, 3])

        super().__init__(screw_axis)


def create_robot_from_urdf(filepath: str) -> Robot:
    tree = ET.parse(filepath)
    root = tree.getroot()

    if root.tag != "robot":
        raise ValueError("URDF root element must be robot.")
    elif "name" not in root.attrib:
        raise ValueError("URDF robot element must have a name attribute.")

    robot_name = root.attrib["name"]

    # Get all joints and links.
    joints = {joint.attrib["name"]: joint for joint in root.iter("joint")}
    links = {link.attrib["name"]: link for link in root.iter("link")}

    # Create a dictionaries to represent the parent-child relationships
    parent_link_to_child_joint: Dict[str, List[str]] = {}
    parent_joint_to_child_link: Dict[str, str] = {}

    for joint_name, joint in joints.items():
        parent_link_name_element = joint.find("parent")
        if parent_link_name_element is None:
            raise ValueError(
                f"Joint: {joint_name} does not have a parent link")
        parent_link_name = parent_link_name_element.attrib["link"]

        child_link_name_element = joint.find("child")
        if child_link_name_element is None:
            raise ValueError(
                f"Joint: {joint_name} does not have a child link")
        child_link_name = child_link_name_element.attrib["link"]

        if parent_link_name not in parent_link_to_child_joint:
            parent_link_to_child_joint[parent_link_name] = []

        parent_link_to_child_joint[parent_link_name].append(joint_name)
        parent_joint_to_child_link[joint_name] = child_link_name

    # Find the root link
    root_link_names = [link_name for link_name, link in links.items() if link.find(
        "parent") is None]

    if len(root_link_names) != 1:
        raise ValueError(
            f"URDF must have exactly one root link. Currently has {len(root_link_names)}")

    # Grab the root link
    root_link_name = root_link_names[0]

    # Leg root joint names
    leg_root_joint_names = [joint_name for joint_name, child_link_name in
                            parent_joint_to_child_link.items() if child_link_name == root_link_name]

    # Create the robot legs
    robot_legs: List[RobotLeg] = []

    for leg_root_joint_name in leg_root_joint_names:
        # Create the robot joints
        robot_joints: List[RobotJoint] = []

        current_joint_name = leg_root_joint_name

        while current_joint_name in parent_joint_to_child_link:
            current_joint = joints[current_joint_name]
            current_joint_type = current_joint.attrib["type"]

            if current_joint_type == "revolute":
                origin = Transform.from_urdf_element(
                    current_joint.find("origin"))
                axis = Direction.from_urdf_element(
                    current_joint.find("axis"))

                robot_joints.append(RevoluteJoint(origin, axis))
            else:
                raise ValueError(
                    f"Joint: {current_joint_name} has an unsupported type: {current_joint_type}")

            current_joint_name = parent_joint_to_child_link[current_joint_name]

        robot_legs.append(RobotLeg(robot_joints, leg_root_joint_name))


def get_leg_from_tree_and_root_name()
###### TENSOR OPERATIONS ######


@ torch.jit.script
def
