from legup.abstract.spatial.spatial import Transform, Screw, Direction
from typing import Iterable, Optional, List, Union, Dict
import torch

from typing import Iterable, List, Optional, Dict, Tuple

from legup.robot.kinematics.tensor_types import TensorWrapper, TensorIndexer, WrappedScalar
from legup.abstract.spatial.spatial import Transform, Screw

from .forward_kinematics import FKResult, FKFunction, QueryLinkInfo


class Link:
    def __init__(self, name: str,
                 child_joints: Iterable["Joint"] = [],
                 device: torch.device = TensorWrapper._default_device()):
        self.name = name
        self.child_joints_dict = {child_joint.name: child_joint
                                  for child_joint in child_joints}
        self.to(device)

    def to(self, device: torch.device):
        """This either does not do anything if this RobotLink is already on device,
        or moves this RobotLink to the device

        Args:
            device (torch.device): Device to move to
        """

        for child_joint in self.child_joints_dict.values():
            child_joint.to(device)

        self.device = device

    def get_child_links(self, result: Optional[Dict[str, "Link"]] = None) -> Dict[str, "Link"]:
        """Returns a dictionary mapping link names to links.

        Returns:
            Dict[str, Link]: A dictionary mapping link names to links.
        """

        if result is None:
            result = {}

        if self.name in result:
            raise ValueError(
                f"Found two links with the same name: {self.name}")

        for child_joint in self.child_joints_dict.values():
            child_joint.child_link.get_child_links(result)

        return result

    def get_links(self) -> Dict[str, "Link"]:
        """Returns a dictionary mapping link names to links including itself.

        Returns:
            Dict[str, Link]: A dictionary mapping link names to links.
        """

        result = self.get_child_links()

        if self.name in result:
            raise ValueError(
                f"Found two links with the same name: {self.name}")

        result[self.name] = self

        return result

    def get_joints(self) -> Dict[str, "Joint"]:
        """Returns a dictionary mapping joint names to joints.

        Returns:
            Dict[str, Joint]: A dictionary mapping joint names to joints.
        """

        links = self.get_links()

        result: Dict[str, "Joint"] = {}

        for link in links.values():
            for joint in link.child_joints_dict.values():
                if joint.name in result:
                    raise ValueError(
                        f"Found two joints with the same name: {joint.name}")

                result[joint.name] = joint

        return result

    def get_dofs(self) -> Dict[str, Screw]:
        """Returns a dictionary mapping joint names to joints.

        Returns:
            Dict[str, Screw]: A dictionary mapping dof names to dof screws.
        """

        joints = self.get_joints()

        result: Dict[str, Screw] = {}

        for joint in joints.values():
            for dof_name, dof_screw in joint.dof_screws_dict.items():
                if dof_name in result:
                    raise ValueError(
                        f"Found two dofs with the same name: {dof_name}")

                result[dof_name] = dof_screw

        return result

    def num_dofs(self) -> int:
        """This function walks along the whole robot and counts the number of dofs.

        Returns:
            int: Number of DOFs
        """

        return len(self.get_dofs())

    def add_child_joint(self, child_joint: "Joint"):
        if child_joint.name in self.get_joints():
            raise ValueError(
                f"Child joint with name {child_joint.name} already exists under link {self.name}.")

        self.child_joints_dict[child_joint.name] = child_joint

    def get_joint_parent_links_dict(self) -> Dict[str, str]:
        """Returns a dictionary mapping joint names to parent link names.
        """

        return {**{child_joint.name: self.name
                   for child_joint in self.child_joints_dict.values()},
                **{child_joint_name: parent_link
                   for child_joint in self.child_joints_dict.values()
                   for child_joint_name, parent_link in child_joint.child_link.get_joint_parent_links_dict().items()}}

    def find_link_dof_chain(self, name: str) -> List[str]:

        joint_chain = self.find_link_joint_chain(name)

        return [dof_name for joint in joint_chain for dof_name in joint.get_dof_names()]

    def find_link_joint_chain(self, name: str) -> List["Joint"]:

        found_child_chains = self.find_link_joint_chain_rec(name)

        if found_child_chains is None:
            raise ValueError(
                f"Could not find link with name: {name} under link: {self.name}")

        return found_child_chains

    def find_link_joint_chain_rec(self, name: str) -> Optional[List["Joint"]]:

        if self.name == name:
            return []

        found_child_chains = {child_joint_name: child_joint.child_link.find_link_joint_chain_rec(name)
                              for child_joint_name, child_joint in self.child_joints_dict.items()}

        found_child_chains = [[self.child_joints_dict[child_joint_name]] + found_child_chain
                              for child_joint_name, found_child_chain in found_child_chains.items()
                              if found_child_chain is not None]

        if len(found_child_chains) is None:
            return None

        if len(found_child_chains) > 1:
            raise ValueError(
                f"Found multiple links with name: {name} under link: {self.name}")

        return found_child_chains.pop()

    def get_zero_transforms(self, query_links: Optional[Iterable[str]] = None) -> Dict[str, Transform]:
        """Returns a dictionary mapping link names to zero transforms.
        """

        query_links = set(query_links) if query_links is not None else None

        return {**({self.name: Transform.zero(device=self.device)}
                   if query_links is None or self.name in query_links
                   else {}),
                **{link_name: child_joint.origin * link_transform
                   for child_joint in self.child_joints_dict.values()
                   for link_name, link_transform in
                   child_joint.child_link.get_zero_transforms(query_links).items()}}

    def make_kinematics(self, query_link_names: Iterable[str]) -> FKFunction:

        # here we create a list of the joint chains that move each query link
        query_link_joint_chains = {link_name: self.find_link_joint_chain(link_name)
                                   for link_name in query_link_names}

        # Here we create a dict that maps each relevant joint name to the joint object
        relevant_joints = {joint.name: joint
                           for query_link_joint_chain in query_link_joint_chains.values()
                           for joint in query_link_joint_chain}

        # Here we create a dict that maps each relevant joint name to its parent link name
        joint_parent_link_name_dict = self.get_joint_parent_links_dict()

        # Here we create a set of all of the relevant links to the query links, including the query links themselves
        relevant_link_names = \
            {*joint_parent_link_name_dict.values(), *query_link_names}

        # Now we compute the zero transforms of all of the relevant links
        link_zero_transforms = self.get_zero_transforms(
            query_links=relevant_link_names)

        # Here we map each joint name to its parent link transform
        joint_zero_transforms = {joint_name: link_zero_transforms[joint_parent_link_name_dict[joint_name]]
                                 for joint_name in relevant_joints.keys()}

        # Here we map each dof name to its parent joint transforms
        dof_zero_transform_dict = {dof_name: joint_zero_transforms[joint.name]
                                   for joint in relevant_joints.values()
                                   for dof_name in joint.get_dof_names()}

        # Here we create a dict that maps each name of a dof that moves the query links to their screws
        relevant_dof_body_screws = {dof_name: dof_screw
                                    for joint in relevant_joints.values()
                                    for dof_name, dof_screw in joint.dof_screws_dict.items()}

        # Here we transform each body screw to the space frame using the adjoint of the zero transform
        dof_space_screw_dict = {dof_name: dof_zero_transform_dict[dof_name].adjoint() * dof_body_screw
                                for dof_name, dof_body_screw in relevant_dof_body_screws.items()}

        query_link_dof_chain_names = {query_link_name: [dof_name for joint in query_link_joint_chain
                                                        for dof_name in joint.get_dof_names()]
                                      for query_link_name, query_link_joint_chain in query_link_joint_chains.items()}

        # Here we create a list of the zero transforms of the query links and the joint chains that move them
        query_link_info_list = \
            [QueryLinkInfo(link_zero_transforms[query_link_name],
                           query_link_dof_chain_names[query_link_name])
             for query_link_name in query_link_names]

        return FKFunction(dof_space_screws=dof_space_screw_dict,
                          query_link_infos=query_link_info_list)


class Joint:
    """This is a class that represents any robot joint that can be mathematically represented by a screw vector"""

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
            device = child_link.device

        self.name = name

        self.dof_screws_dict = {f"{self.name}_{dof_idx}": dof_screw
                                for dof_idx, dof_screw in enumerate(list(screws))}

        self.origin = origin
        self.child_link = child_link
        self.device = device
        self.num_dofs = len(list(screws))

        self.to(device)

    @staticmethod
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

    @staticmethod
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

        for screw in self.dof_screws_dict.values():
            screw.to(device)

        self.child_link.to(device)

    def get_dof_names(self) -> List[str]:
        return list(self.dof_screws_dict.keys())
