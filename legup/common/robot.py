from legup.common.tensor_wrapper import TensorWrapper
from legup.common.spatial import Transform, Screw, Position, Direction

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

        end_shape = list(joint_angles_tensor.shape[-1:])

        super().__init__(joint_angles_tensor,
                         end_shape=end_shape)

    def for_robot(self, robot: "Robot"):
        return robot.name == self.robot_name

    def assert_for_robot(self, robot: "Robot"):
        """This function throws an error when the joint angles are not for the given robot."""
        if not self.for_robot(robot):
            raise ValueError(
                f"Joint angles for {self.robot_name} used for {robot.name}.")


class Robot:
    """Abstract class for a robot. A robot is a collection of legs that does forward kinematics."""

    def __init__(self, legs: List["RobotLeg"],
                 name: str,
                 home_position: Optional[Union[RobotJointAngles,
                                               torch.Tensor]] = None,
                 device: Optional[torch.device] = None):

        if device is None:
            device = torch.device("cpu")

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

    def assign_home_position(self, home_position: RobotJointAngles):
        """Assigns a home position to the robot.

        Args:
            home_position (RobotJointAngles): The home position.
        """

        home_position.assert_for_robot(self)

        self.home_position = home_position

    # @overload
    def forward_kinematics(self, joint_angles: RobotJointAngles, out: Optional[Position] = None) -> Position:
        """Computes the forward kinematics of the robot."""
        joint_angles.assert_for_robot(self)

        raise NotImplementedError(
            "This method is not implemented for this class.")

    # @overload
    # def forward_kinematics(self, joint_angles_tensor: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    #     """Computes the forward kinematics of the robot."""
    #     joint_angles = self.joint_angles_class(joint_angles_tensor)

    #     for leg in self.legs:
    #         leg.forward_kinematics(joint_angles, out=out)


class RobotLeg:
    def __init__(self, joints: List["RobotJoint"], name: str):
        self.joints = joints
        self.name = name
        self.num_dofs = len(joints)

    def to(self, device: torch.device):
        return RobotLeg([joint.to(device) for joint in self.joints], self.name)


class RobotJoint(ABC):
    """This is an abstract class that represents a robot joint.
    This class can be overridden by subclasses to support various types of joints."""

    def __init__(self, screw: Screw):
        self.screw = screw

    def to(self, device: torch.device) -> "RobotJoint":
        if self.screw.device == device:
            return self

        moved_screw = self.screw.to(device)
        return self.__class__(moved_screw)


class RevoluteJoint(RobotJoint):
    def __init__(self, origin: Position, axis: Direction):
        """Creates a revolute joint.

        Args:
            origin: The origin of the joint.
            axis: The axis of the joint.

        """

        broadcasted_shape = torch.broadcast_shapes(
            origin.pre_shape, axis.pre_shape)

        # origin = origin.broadcast_to(broadcasted_shape)

        screw_axis = Screw.empty_screw(
            (*origin.pre_shape(), 6), device=origin.device)

        screw_axis[..., :3] = axis.tensor
        screw_axis[..., 3:] = torch.cross(-axis.tensor, origin.tensor, dim=-1)

        super().__init__(screw_axis)

###### TENSOR OPERATIONS ######


# @ torch.jit.script
# def
