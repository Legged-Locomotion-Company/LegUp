from legup.common.tensor_wrapper import TensorWrapper
from legup.common.spatial import Transform, Screw, Position, Direction

from typing import List, Optional, Union, Dict, Iterable, TypeVar
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
        if joint_angles_tensor.shape[-1] != robot.get_num_dofs():
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
                 device: Optional[torch.device] = None,
                 knee_joint_indices: Optional[torch.Tensor] = None,
                 foot_link_indices: Optional[torch.Tensor] = None,
                 shank_link_indices: Optional[torch.Tensor] = None,
                 thigh_link_indices: Optional[torch.Tensor] = None,
                 knee_joint_limits: Optional[torch.Tensor] = None):
        """Initializes a robot.

        Args:
            legs (List[RobotLeg]): The legs of the robot.
            name (str): The name of the robot.
            home_position (Optional[Union[RobotJointAngles, torch.Tensor]], optional): The home position of the robot. Defaults to None.
            device (Optional[torch.device], optional): The device to use. Defaults to None.
            knee_joint_indices (Optional[torch.Tensor], optional): The indices of the knee joints. Defaults to None.
            foot_link_indices (Optional[torch.Tensor], optional): The indices of the foot links. Defaults to None.
            shank_link_indices (Optional[torch.Tensor], optional): The indices of the shank links. Defaults to None.
            thigh_link_indices (Optional[torch.Tensor], optional): The indices of the thigh links. Defaults to None.
        """

        if device is None:
            device = TensorWrapper._default_device()

        self.legs = [leg.to(device) for leg in legs]
        self.name = name
        self.num_dofs = sum([leg.num_dofs for leg in legs])
        if home_position is None:
            home_position = RobotJointAngles(
                self, torch.zeros(self.num_dofs, device=device))
        else:
            home_position = home_position.to(device)
        self.home_position = home_position

        self.knee_joint_indices = knee_joint_indices
        self.foot_link_indices = foot_link_indices
        self.shank_link_indices = shank_link_indices
        self.thigh_link_indices = thigh_link_indices
        self.knee_joint_limits = knee_joint_limits

    def get_knee_joint_indices(self) -> torch.Tensor:
        """This function either returns the knee joint indices or throws an error if they are not set.

        Returns:
            torch.Tensor: a tensor containing the indices of the knee joints in the joint angles tensor.
        """

        if self.knee_joint_indices is None:
            raise ValueError(
                f"Knee joint indices are not set for robot {self.name}.")

        return self.knee_joint_indices

    def get_foot_link_indices(self) -> torch.Tensor:
        """This function either returns the foot link indices or throws an error if they are not set.

        Returns:
            torch.Tensor: a tensor containing the indices of the foot links in the link transforms tensor.
        """

        if self.foot_link_indices is None:
            raise ValueError(
                f"Foot link indices are not set for robot {self.name}.")

        return self.foot_link_indices

    def get_shank_link_indices(self) -> torch.Tensor:
        """This function either returns the shank link indices or throws an error if they are not set.

        Returns:
            torch.Tensor: a tensor containing the indices of the shank links in the link transforms tensor.
        """

        if self.shank_link_indices is None:
            raise ValueError(
                f"Shank link indices are not set for robot {self.name}.")

        return self.shank_link_indices

    def get_thigh_link_indices(self) -> torch.Tensor:
        """This function either returns the thigh link indices or throws an error if they are not set.

        Returns:
            torch.Tensor: a tensor containing the indices of the thigh links in the link transforms tensor.
        """

        if self.thigh_link_indices is None:
            raise ValueError(
                f"Thigh link indices are not set for robot {self.name}.")

        return self.thigh_link_indices

    def get_knee_joint_limits(self) -> torch.Tensor:
        """This function either returns the knee joint limits or throws an error if they are not set.

        Returns:
            torch.Tensor: a tensor containing the limits of the knee joints in the joint angles tensor.
        """

        if self.knee_joint_limits is None:
            raise ValueError(
                f"Knee joint limits are not set for robot {self.name}.")

        return self.knee_joint_limits

    def get_num_dofs(self) -> int:
        """This function returns the number of degrees of freedom of the robot.

        Returns:
            torch.Tensor: a tensor containing the number of degrees of freedom of the robot.
        """

        return self.num_dofs

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


class RobotLink:
    def __init__(self, name: str, child_joints: Iterable["RobotJoint"] = [], device: torch.device = TensorWrapper._default_device()):
        self.name = name
        self.child_joints = child_joints
        self.device = device
        self.to(device)

    def to(self, device: torch.device):
        for child_joint in self.child_joints:
            child_joint.to(device)

        return self


RobotJointSubclass = TypeVar("RobotJointSubclass", bound="RobotJoint")


class RobotJoint:
    """This is an abstract class that represents a robot joint.
    This class can be overridden by subclasses to support various types of joints."""

    def __init__(self, name: str,
                 origin: Transform,
                 screws: Iterable[Screw],
                 child_link: RobotLink,
                 device: Optional[torch.device] = None):
        """Creates a robot joint.

        Args:
            name (str): The name of the joint.
            origin (Transform): The origin of the in its parent link.
            screws (Iterable[Screw]): The screws of the joint.
            child_link (RobotLink): The child link of the joint.
            device (torch.device, optional): The device of the joint. Defaults to None.
        """

        screw_devices = set([screw.device for screw in screws])

        if device is None and len(screw_devices) == 1:
            device = screw_devices.pop()
        elif device is None:
            raise ValueError(
                "Either all screws must be on the same device or a device must be specified.")

        self.name = name
        self.screws = list(screws)
        self.origin = origin
        self.child_link = child_link
        self.device = device
        self.num_dofs = len(self.screws)

    @staticmethod
    def make_revolute(name: str,
                      origin: Transform,
                      axis: Direction,
                      child_link: RobotLink,
                      device: Optional[torch.device] = None) -> "RobotJoint":
        """Creates a revolute joint.

        Args:
            name (str): The name of the joint.
            origin (Transform): The origin of the in its parent link.
            axis (Direction): The axis of the joint.
            child_link (RobotLink): The child link of the joint.
            device (torch.device, optional): The device of the joint. Defaults to None.

        Returns:
            RobotJoint: The revolute joint.
        """

        origin_pos = origin.get_position()
        screw = Screw.from_axis_and_origin(axis, origin_pos)

        return RobotJoint(name, origin, [screw], child_link, device)

    def to(self: RobotJointSubclass, device: torch.device) -> RobotJointSubclass:
        """Moves this joint to the given device. This modifies the joint so other references to it will be modified as well.

        Args:
            device (torch.device): The device to move the joint to.

            Returns:
                RobotJoint: itself.
        """

        if self.screw.device != device:

            self.screw = self.screw.to(device)
            self.child_link = self.child_link.to(device)

        return self


# class RevoluteJoint(RobotJoint):
#     def __init__(self, name: str,
#                  origin: Position,
#                  axis: Direction,
#                  child_link: RobotLink,
#                  device: Optional[torch.device] = None):
#         """Creates a revolute joint.

#         Args:
#             origin: The origin of the joint.
#             axis: The axis of the joint.
#         """

#         if device is None and origin.device != axis.device:
#             raise ValueError(
#                 f"The origin and axis must be on the same device, or the device must be specified. Origin is on {origin.device} and axis is on {axis.device}.")
#         elif device is None:
#             device = origin.device
#         elif origin.pre_shape != axis.pre_shape:
#             raise ValueError(
#                 f"Cannot create a revolute joint with origin with pre shape {origin.pre_shape} and axis with pre shape {axis.pre_shape}. They must be the same")

#         screw = Screw.from_axis_and_origin(
#             axis.to(device=device), origin.to(device=device))

#         super().__init__(name=name, screws=[screw],
#                          child_link=child_link, device=device)

###### TENSOR OPERATIONS ######
