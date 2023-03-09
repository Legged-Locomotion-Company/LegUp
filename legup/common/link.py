import torch

from typing import Iterable, List, Optional, Dict, Tuple

from legup.common.tensor_types import TensorWrapper, TensorIndexer, WrappedScalar
from legup.common.joint import Joint
from legup.common.spatial import Transform, Screw


class Link:
    def __init__(self, name: str,
                 child_joints: Iterable["Joint"] = [],
                 device: torch.device = TensorWrapper._default_device()):
        self.name = name
        self.child_joints_dict = {child_joint.name: child_joint
                                  for child_joint in child_joints}
        self.device = device
        self.to(device)

    def to(self, device: torch.device):
        """This either does not do anything if this RobotLink is already on device,
        or moves this RobotLink to the device

        Args:
            device (torch.device): Device to move to
        """

        for child_joint in self.child_joints_dict.values():
            child_joint.to(device)

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

    def get_joints(self) -> Dict[str, Joint]:
        """Returns a dictionary mapping joint names to joints.

        Returns:
            Dict[str, Joint]: A dictionary mapping joint names to joints.
        """

        links = self.get_links()

        result: Dict[str, Joint] = {}

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
            for dof_name, dof_screw in joint.get_dofs().items():
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

    def add_child_joint(self, child_joint: Joint):
        if child_joint.name in self.get_joints():
            raise ValueError(
                f"Child joint with name {child_joint.name} already exists under link {self.name}.")

        self.child_joints_dict[child_joint.name] = child_joint

    def forward_kinematics(self,
                           query_links: Optional[Iterable[str]] = None,
                           dof_angles: Optional[TensorIndexer[WrappedScalar]] = None
                           ) -> Dict[str, Transform]:
        """Computes the forward kinematics of the robot.

        Args:
            joint_angle_dict (Dict[str, torch.Tensor]): A dictionary mapping joint names to joint angles.

        Returns:
            Dict[str, Transform]: A dictionary mapping joint names to transforms.
        """

        if query_links is None:
            query_links = self.get_links().keys()

        if dof_angles is None:
            dof_angles = TensorIndexer(WrappedScalar(torch.zeros(self.num_dofs(), device=self.device)),
                                       {dof_name: i for i, dof_name in enumerate(self.get_dofs().keys())})

        link_names = list(query_links)

        dof_transforms = self.create_dof_transforms(dof_angles)

        return {link_name: self.get_link_transform(link_name, dof_transforms) for link_name in link_names}

    def get_link_transform(self, link_name: str, dof_transforms: TensorIndexer[Transform]) -> Transform:
        """Computes the transform of a given link.

        Args:
            link_name (str): Name of the link to compute the transform for.
            dof_transforms (Dict[str, Transform]): A dictionary mapping joint names to transforms.

        Returns:
            Transform: The transform of the link.
        """

        joint_chain = self.find_link_joint_chain(link_name)

        relative_transforms: List[Transform] = []

        for joint in joint_chain:

            joint_dof_names = joint.get_dof_names()

            relative_transforms.append(joint.origin)
            relative_transforms.extend([dof_transforms[dof_name]
                                        for dof_name in joint_dof_names])

        return Transform.compose(*relative_transforms)

    def find_link_dof_chain(self, name: str) -> List[str]:

        joint_chain = self.find_link_joint_chain(name)

        return [dof_name for joint in joint_chain for dof_name in joint.get_dof_names()]

    def find_link_joint_chain(self, name: str) -> List[Joint]:

        found_child_chains = self._find_link_joint_chain(name)

        if found_child_chains is None:
            raise ValueError(
                f"Could not find link with name: {name} under link: {self.name}")

        return found_child_chains

    def _find_link_joint_chain(self, name: str) -> Optional[List[Joint]]:

        if self.name == name:
            return []

        found_child_chains = {child_joint_name: child_joint.child_link._find_link_joint_chain(name)
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

    def create_dof_transforms(self, dof_angles: TensorIndexer[WrappedScalar]) -> TensorIndexer[Transform]:

        dofs = self.get_dofs()

        ordered_dofs: List[Screw] = []

        dof_names = dof_angles.get_idx_names()

        for dof_name in dof_names:
            if dof_name not in dofs:
                raise ValueError(
                    f"Dof with name {dof_name} from DOFAngles is not under link {self.name}")

            ordered_dofs.append(dofs[dof_name])

        screw_stack = Screw.stack(ordered_dofs)

        dof_transforms = screw_stack.apply(dof_angles.tensor_wrapper.tensor)

        dof_names_to_idx = {dof_name: idx for idx, dof_name
                            in enumerate(dof_names)}

        return TensorIndexer(dof_transforms, dof_names_to_idx)
