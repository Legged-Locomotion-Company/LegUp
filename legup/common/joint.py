
import torch

from typing import Iterable, Optional, List, Union, Dict

from abc import abstractmethod

from legup.common.spatial import Transform, Screw, Direction
from legup.common.link import Link


class Joint:
    """This is an abstract class that represents a robot joint.
    This class can be overridden by subclasses to support various types of joints."""

    def __init__(self, name: str,
                 origin: Transform,
                 screws: Iterable[Screw],
                 child_link: Link,
                 device: Optional[torch.device] = None):
        """Creates a robot joint.

        Args:
            name (str): The name of the joint.
            origin (Transform): The origin of the in its parent link.
            screws (Iterable[Screw]): The screws of the joint.
            child_link (RobotLink): The child link of the joint.
            device (torch.device, optional): The device of the joint. Defaults to None.
        """

        if device is None:
            devices = set([screw.device for screw in screws])
            devices.add(origin.device)
            devices.add(child_link.device)

            if len(devices) == 1:
                device = devices.pop()
            else:
                raise ValueError(
                    "Either all screws, origin, and child link must be on the same device or a device must be specified.")

        self.name = name

        # stack screws
        self.screws = Screw.stack(screws)

        self.dof_idx_dict = {f"{self.name}_{dof_idx}": dof_idx
                             for dof_idx in range(len(list(screws)))}

        self.origin = origin
        self.child_link = child_link
        self.device = device
        self.num_dofs = len(list(screws))

        self.to(device)

    def get_dofs(self) -> Dict[str, Screw]:
        return {name: self.screws[..., idx]
                for name, idx in self.dof_idx_dict}

    def apply(self, angles: torch.Tensor) -> Transform:
        """Applies the joint to the given joint angles.

        Args:
            joint_angles (torch.Tensor): An (...) shaped tensor of joint angles

        Returns:
            Transform: A (...) shaped transform.
        """

        if not self.num_dofs == 1 and angles.shape[-1] != self.num_dofs:
            raise ValueError(
                f"Expected {self.num_dofs} joint angles but got {angles.shape[-1]}.")

        return self.origin * self.screws.apply(angles)

    @ staticmethod
    def make_revolute(name: str,
                      origin: Transform,
                      axis: Direction,
                      child_link: Link,
                      device: Optional[torch.device] = None) -> "Joint":
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

        return Joint(name, origin, [screw], child_link, device)

    @ staticmethod
    def make_fixed(name: str, origin: Transform, child_link: Link, device: Optional[torch.device] = None) -> "Joint":
        """Creates a fixed joint.

        Args:
            name (str): The name of the joint.
            origin (Transform): The origin of the in its parent link.
            child_link (RobotLink): The child link of the joint.
            device (torch.device, optional): The device of the joint. Defaults to None.

        Returns:
            RobotJoint: The fixed joint.
        """

        return Joint(name, origin, [], child_link, device)

    def to(self, device: torch.device):
        """Either moves this RobotJoint to a new device,
        or if it is already on device does nothing

        Args:
            device (torch.device): The device to move the joint to.
        """

        for screw in self.screws:
            screw.to(device)

        self.child_link.to(device)

    def get_dof_names(self) -> List[str]:
        dof_names_and_idxs = [(name, idx) for name, idx in
                              self.dof_idx_dict.items()]

        return [name for name, _ in
                sorted(dof_names_and_idxs, key=lambda x: x[1])]

    @abstractmethod
    def make_screw(self, absolute_origin: Transform) -> Screw:
        pass


# class RevoluteJoint(Joint):
#     """A revolute joint."""

#     def __init__(self, name: str,
#                  origin: Transform,
#                  axis: Direction,
#                  child_link: Link,
#                  device: Optional[torch.device] = None):
#         """Creates a revolute joint.

#         Args:
#             name (str): The name of the joint.
#             origin (Transform): The origin of the in its parent link.
#             axis (Direction): The axis of the joint.
#             child_link (RobotLink): The child link of the joint.
#             device (torch.device, optional): The device of the joint. Defaults to None.
#         """

#         self.name = name
#         self.origin = origin
#         self.axis = axis
#         self.child_link = child_link
#         self.device = device

#     def make_screw(self, absolute_origin: Transform) -> Screw:
#         rotation = absolute_origin.get_rotation()
