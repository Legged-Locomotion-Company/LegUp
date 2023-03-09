from legup.common.tensor_types import TensorWrapper, TensorIndexer, WrappedScalar
from legup.common.spatial import Transform, Screw, Position, Direction, ScrewSkew, _raw_twist_skew_exp_map
from legup.common.link import Link, Joint

from typing import List, Optional, Union, Dict, Iterable, TypeVar, Callable
from copy import copy
from abc import ABC, abstractmethod

import xml.etree.ElementTree as ET

import torch


class DOFAngle(TensorWrapper):
    def __init__(self, joint_angles: torch.Tensor):

        self.num_dofs = joint_angles.shape[-1]

        super().initialize_base(joint_angles, [self.num_dofs])


class KinematicsObject:
    def __init__(self, ):

    def make_function(self):
        def kinematics(joint_angles: torch.Tensor):

            if joint_angles.shape[-1] != self.num_dofs:
                raise ValueError(
                    f"Joint angles must have {self.num_dofs} DOFs, but got {joint_angles.shape[-1]}.")


class Robot:
    """Abstract class for a robot. A robot is a collection of legs that does forward kinematics."""

    def __init__(self, base_link: Link,
                 name: str,
                 home_position: Optional[Dict[str, List[float]]],
                 device: Optional[torch.device] = None,
                 knee_joint_names: Optional[List[str]] = None,
                 foot_link_names: Optional[List[str]] = None,
                 shank_link_names: Optional[List[torch.Tensor]] = None,
                 thigh_link_names: Optional[List[torch.Tensor]] = None,
                 knee_joint_limits: Optional[List[torch.Tensor]] = None):
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

        # Assign the straightforward values
        # We assume that the robot referenced will not change

        if device is None:
            device = base_link.device

        self.base_link = base_link

        self.name = name

        self.knee_joint_names = knee_joint_names
        self.foot_link_names = foot_link_names
        self.shank_link_names = shank_link_names
        self.thigh_link_names = thigh_link_names
        self.knee_joint_limits = knee_joint_limits
        self.foot_link_names = foot_link_names

        self.device = device

        dof_screws_dict = self.base_link.get_dofs()

        links_dict = self.base_link.get_links()

        self.dof_screws = TensorIndexer.from_dict(dof_screws_dict)

        self.dof_screw_skews = TensorIndexer.from_dict(
            {dof_name: dof_screw.skew()
             for dof_name, dof_screw in dof_screws_dict.items()})

        robot_joints = self.base_link.get_joints()

        if home_position is not None:
            home_dof_theta_dict: Dict[str, float] = {}

            for joint_name, joint_thetas in home_position.items():

                if len(joint_thetas) != robot_joints[joint_name].num_dofs:
                    raise ValueError(
                        f"Home position for joint {joint_name} has {len(joint_thetas)} values, but joint has {robot_joints[joint_name].num_dofs} DOFs.")

                home_dof_theta_dict.update(
                    zip(robot_joints[joint_name].get_dof_names(), joint_thetas))

            self.home_position = torch.tensor([home_dof_theta_dict[dof_name]
                                               for dof_name in self.dof_screws.get_idx_names()])

        else:
            self.home_position = torch.zeros(
                self.dof_screws.num_idxs(), device=self.device)

        self.zero_transforms = self.base_link.forward_kinematics()

        self.link_body_screw_skews: Dict[str, TensorIndexer[ScrewSkew]] = {}

        for link_name in self.base_link.get_links().keys():

            # Get dofs that move this link
            relevant_dofs: List[str] = []

            link_joint_chain = base_link.find_link_joint_chain(link_name)
            for joint in link_joint_chain:
                relevant_dofs.extend(joint.get_dof_names())

            # In Modern Robotics this value is called M
            zero_transform = self.zero_transforms[link_name]

            # In Modern Robotics this value is called [Ad_M^{-1}]
            zero_transform_inv_adj = zero_transform.invert().adjoint()

            body_screws_dict = {dof_name: zero_transform_inv_adj * self.dof_screws[dof_name]
                                for dof_name in relevant_dofs}

            body_screw_skews_dict = {dof_name: dof_screw.skew()
                                     for dof_name, dof_screw in body_screws_dict.items()}

            self.link_body_screw_skews[link_name] = \
                TensorIndexer.from_dict(body_screw_skews_dict)

    def add_joint_and_link(self, joint: Joint, link: Link):
        """Adds a joint and link to the robot.

        Args:
            joint (Joint): The joint to add.
            link (Link): The link to add.
        """

        self.base_link.add_joint_and_link(joint, link)

    def get_dof_idx(self, dof_name: str) -> int:
        """Gets the index of a degree of freedom.

        Args:
            dof_name (str): The name of the degree of freedom.

        Returns:
            int: The index of the degree of freedom.
        """

        return self.dof_screws.get_idx(dof_name)

    def forward_kinematics(self, query_links: Iterable[str], dof_thetas: TensorIndexer[WrappedScalar]) -> TensorIndexer[Transform]:

        # First we create a tensor with the dof angles in the correct order
        theta_tensor = \
            dof_thetas.to_tensor(self.dof_screw_skews.get_idx_names())

        # Now we create a TensorIndexer to relate dof name strings to their relative transforms
        dof_transforms = \
            self.dof_screw_skews.apply(ScrewSkew.apply, theta_tensor)

        # Now we need the joint chains so we can create a transform list for each joint
        query_link_joint_chains = {link_name: self.base_link.find_link_joint_chain(link_name)
                                   for link_name in query_links}

        # Now we use a set comprehension to pull out all the unique joint names
        unique_joints = \
            {joint
             for joint_chain in query_link_joint_chains.values()
             for joint in joint_chain}

        # Now we create a list of transforms for each joint
        joint_transform_lists = \
            {joint.name: [joint.origin] + [dof_transforms[dof_name]
                                           for dof_name in joint.get_dof_names()]
             for joint in unique_joints}

        # Now we create a list of transforms for each query link
        query_link_transform_lists = \
            {query_link_name: [transform
                               for joint in joint_chain
                               for transform in joint_transform_lists[joint.name]]
                for query_link_name, joint_chain in query_link_joint_chains.items()}

        # Now we compose the transforms for each query link
        query_link_transforms = \
            {query_link_name: Transform.compose(*transform_list)
             for query_link_name, transform_list
             in query_link_transform_lists.items()}

        return TensorIndexer.from_dict(query_link_transforms)

    def get_knee_joint_indices(self) -> torch.Tensor:

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

    @staticmethod
    def make_kinematics(dofs: TensorIndexer[ScrewSkew], joint_chains: List[List[Joint]]) -> torch.ScriptFunction:

        joint_dict = {joint.name: joint for joint_chain in joint_chains
                      for joint in joint_chain}

        joint_names = list(joint_dict.keys())
        joint_origins = {joint_name: joint_dict[joint_name].origin
                         for joint_name in joint_names}
        joint_dof_name_lists = [joint_dict[joint_name].get_dof_names()
                                for joint_name in joint_names]
        joint_chain_joint_names = [[joint.name for joint in joint_chain]
                                   for joint_chain in joint_chains]

        raw_dof_screw_skews = dofs.tensor_wrapper.tensor.clone()

        dof_names = dofs.get_idx_names()
        num_dofs = len(dof_names)

        zero_transforms = torch.zeros(
            [len(joint_names), 4, 4], dtype=torch.float32)

        def kinematics(dof_thetas: torch.Tensor) -> torch.Tensor:

            if dof_thetas.shape[-1] != num_dofs:
                raise ValueError(
                    f"Expected {num_dofs} dofs but got {dof_thetas.shape[0]}.")

            # First we broadcast the raw_dof_screw_skews and dof_thetas together
            broadcast_shape = torch.broadcast_shapes(
                raw_dof_screw_skews.shape[:-2], dof_thetas.shape)
            broadcast_dof_screw_skews = raw_dof_screw_skews.broadcast_to(
                [*broadcast_shape, 4, 4])
            broadcast_dof_thetas = dof_thetas.broadcast_to(broadcast_shape)

            # Now we scale the screw skews by the dof angles
            scaled_dof_screw_skews = \
                broadcast_dof_screw_skews * \
                broadcast_dof_thetas[..., None, None]

            # Now we take the exp map of the scaled screw skews
            dof_relative_transforms_tensor = \
                _raw_twist_skew_exp_map(scaled_dof_screw_skews)

            # Now we turn the dof relative transforms into a list
            dof_relative_transforms_list = \
                [dof_relative_transforms_tensor[..., dof_idx, :, :]
                 for dof_idx in range(len(dof_names))]

            # Here we create a dict which holds the relative transforms for each joint
            # To do this we combine the origin transform for the joint and its dof transforms
            joint_transforms: List[torch.Tensor] = \
                [torch.chain_matmul([joint_origin] + [dof_relative_transforms_list[dof_names.index(dof_name)]
                                                      for dof_name in joint_dof_name_list])
                 for joint_origin, joint_dof_name_list in zip(joint_origins, joint_dof_name_lists)]

            # Now we combine the joint transforms for each joint chain
            joint_chain_transforms = \
                [torch.chain_matmul([joint_transforms[joint_names.index(joint_name)]
                                     for joint_name in joint_chain_joint_names])
                 for joint_chain_joint_names in joint_chain_joint_names]

        return torch.jit.script(kinematics)  # type: ignore
