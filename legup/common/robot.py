from tensor_wrapper import TensorWrapper
from spatial import Transform, Screw, Position, Direction

from typing import List, Optional, overload, Union, Dict
from abc import ABC, abstractmethod

import xml.etree.ElementTree as ET

import torch

@torch.jit.script  # type: ignore
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
    @torch.jit.script
    def __init__(self, origin: Position, axis: Direction):
        """Creates a revolute joint.

        Args:
            origin: The origin of the joint.
            axis: The axis of the joint.

        """

        broadcasted_shape = torch.broadcast_shapes(
            origin.pre_shape, axis.pre_shape)
        
        origin = origin.broadcast_to(broadcasted_shape)

        screw_axis = Screw.empty_screw(
            (*origin.pre_shape, 6), device=origin.device)

        screw_axis[..., :3] = axis.tensor
        screw_axis[..., 3:] = torch.cross(-axis.tensor, origin.tensor, dim=-1)

        super().__init__(screw_axis)

###### TENSOR OPERATIONS ######


@ torch.jit.script
def
