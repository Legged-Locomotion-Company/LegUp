from legup.common.tensor_types import TensorWrapper, TensorIndexer, WrappedScalar
from legup.common.spatial import Transform, Screw, Position, Direction, ScrewSkew, ScrewJacobian, raw_spatial_methods
from legup.common.link_joint import Link

from typing import List, Optional, Union, Dict, Iterable, TypeVar, Callable, Tuple
from copy import copy
from abc import ABC, abstractmethod

import xml.etree.ElementTree as ET

import torch


class DOFAngle(TensorWrapper):
    def __init__(self, joint_angles: torch.Tensor):

        self.num_dofs = joint_angles.shape[-1]

        super().initialize_base(joint_angles, [self.num_dofs])


class KinematicsResult:
    def __init__(self,
                 transform: Transform,
                 jacobian: ScrewJacobian):
        self.transform = transform
        self.jacobian = jacobian

    @staticmethod
    def stack(kinematics_results: List["KinematicsResult"]):
        return KinematicsResult(
            transform=Transform.stack(
                [result.transform for result in kinematics_results], dim=-1),
            jacobian=ScrewJacobian.stack(
                [result.jacobian for result in kinematics_results], dim=-1))


class KinematicsObject:
    def __init__(self,
                 dof_space_screws: Union[Dict[str, Screw], TensorIndexer[Screw]],
                 query_link_zero_transforms: List[Transform],
                 query_link_dof_chains: List[List[str]],
                 dof_order: Optional[Iterable[str]] = None,):
        """Initializes a kinematics object.

        Args:
            dof_space_screws(Union[Dict[str, Screw], TensorIndexer[Screw]]): This is mapping from dof names to their screws in space frame
            query_link_zero_transforms (List[Transform]): This is a list of transforms from the base frame to the zero configuration of each query link
            query_link_dof_chains (List[List[str]]): This is a list of lists of dof names for each query link
            dof_order (Optional[Iterable[str]], optional): The order of dofs for calls to this object. Passing this as None will use the order of the keys in dof_body_screw_skews.
        """

        if len(query_link_zero_transforms) != len(query_link_dof_chains):
            raise ValueError(
                "The number of query link zero transforms must be equal to the number of query link dof chains.")

        # Check to make sure every dof name in every link dof chain is in the dof_space_screws
        for link_dof_chain in query_link_dof_chains:
            for dof_name in link_dof_chain:
                if dof_name not in dof_space_screws:
                    raise ValueError(
                        f"The dof {dof_name} is not in the dof_space_screws.")

        # If the dof_space_screws is a dict, we convert it to a TensorIndexer
        if not isinstance(dof_space_screws, TensorIndexer):
            dof_space_screws = TensorIndexer.from_dict(dof_space_screws)

        # If the dof_order is None, we use the order of the keys in the dof_space_screws
        if dof_order is None:
            dof_order = dof_space_screws.get_idx_names()
        self.dof_order = list(dof_order)

        self.query_link_zero_transforms = query_link_zero_transforms
        self.query_link_dof_chains = query_link_dof_chains

        self.dof_space_screws = dof_space_screws.reordered(self.dof_order)

    def get_dof_idxs(self, dof_names: Iterable[str]) -> List[int]:
        """This function returns the indices of the dofs in the dof_names list.

        Args:
            dof_names (Iterable[str]): This is a list of dof names

        Returns:
            torch.Tensor: This is a list of indices of the dofs in the dof_names list
        """

        return [self.dof_order.index(dof_name) for dof_name in dof_names]

    def __call__(self, joint_angles: torch.Tensor) -> KinematicsResult:

        dof_exps = self.dof_space_screws.apply(Screw.apply, joint_angles)

        # Now for each query link, we compute the kinematics
        link_kinematics = [KinematicsObject.single_kinematics(
            dof_space_screws=self.dof_space_screws,
            dof_exps=dof_exps,
            dof_chain_names=link_dof_chain_names,
            zero_transform=link_zero_transform)

            for (link_zero_transform, link_dof_chain_names)
            in zip(self.query_link_zero_transforms, self.query_link_dof_chains)]

        return KinematicsResult.stack(link_kinematics)

    @staticmethod
    def single_kinematics(dof_space_screws: TensorIndexer[Screw],
                          dof_exps: TensorIndexer[Transform],
                          dof_chain_names: List[str],
                          zero_transform: Transform) -> KinematicsResult:
        """This function computes the forward kinematics and jacobian of a single chain of dofs.

        Args:
            dof_space_screws (TensorIndexer[Screw]): This is a TensorIndexer mapping from dof names to their screw skews in space frame
            dof_exps (TensorIndexer[Transform]): This is a TensorIndexer mapping from dof names to their scaled screw exps in space frame
            dof_chain_names (List[str]): This is a list of dof names that represent the chain of dofs that affect the end effector
            zero_transform (Transform): This is the transform of the end effector when all dofs are at 0

        Returns:
            KinematicsResult: This is a KinematicsResult object that contains the transforms and jacobian of the end effector
        """

        raw_transform, raw_jacobian = KinematicsObject.raw_single_kinematics(
            dof_space_screws=dof_space_screws.to_raw_dict(),
            dof_exps=dof_exps.to_raw_dict(),
            dof_chain_names=dof_chain_names,
            zero_transform=zero_transform.tensor)

        return KinematicsResult(Transform(raw_transform), ScrewJacobian(raw_jacobian))

    @staticmethod
    @torch.jit.script  # type: ignore
    def raw_single_kinematics(dof_space_screws: Dict[str, torch.Tensor],
                              dof_exps: Dict[str, torch.Tensor],
                              dof_chain_names: List[str],
                              zero_transform: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This function computes the forward kinematics and jacobian of a single chain of dofs.

        Args:
            dof_space_screws (Dict[str, torch.Tensor]): This is a dict mapping from dof names to their screw skews in space frame
            dof_exps (Dict[str, torch.Tensor]): This is a dict mapping from dof names to their scaled screw exps in space frame
            dof_chain_names (List[str]): This is a list of dof names that represent the chain of dofs that affect the end effector
            zero_transform (torch.Tensor): This is the transform of the end effector when all dofs are at 0

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: this is a tuple containing (position, jacobian) of the end effector relative to the dofs in dof_chain_names
        """

        # Here we're gonna grab some element from the dof_exps dict
        dof_exps_element = list(dof_exps.values())[0]

        # Now we extract the shape and device from it
        # The final 2 shape elements should be 4, 4 because the dof exps are transforms
        # The shape other than the final 2 elements should be the shape of robots we are computing this for
        device, shape = dof_exps_element.device, dof_exps_element.shape[:-2]

        # Now we create a list products for every dof_exp in the chain for every dof_exp leading up to it
        prev_dof_chain_prods: List[torch.Tensor] = \
            [torch.eye(4, device=device, dtype=dof_exps_element.dtype)
             .expand(shape + (4, 4))]

        # Now we populate that list iteratively
        for dof_chain_name in dof_chain_names:
            prev_dof_chain_prods.append(
                prev_dof_chain_prods[-1] @ dof_exps[dof_chain_name])

        # Here we calculate the jacobians according to equation 5.11 in Modern Robotics
        space_jacobian_cols = [(raw_spatial_methods.transform_adjoint(prev_prod) @ dof_space_screws[dof_name])
                               for prev_prod, dof_name in zip(prev_dof_chain_prods, dof_chain_names)]

        # Now we calculate the end effector position according to the product of exponentials formula equation 5.9 and 4.14 in Modern Robotics
        transform_space_body = prev_dof_chain_prods[-1] @ zero_transform

        # Now we stack the columns of the jacobian to get the full space jacobian matrix
        space_jacobian = torch.stack(space_jacobian_cols, dim=-1)

        # Now I get the ee jacobian in space frame by multiplying the space jacobian by the adjoint of the transform which
        # translates the body frame to the space frame, without rotating it
        translate_body_space = \
            raw_spatial_methods.eyes_like(transform_space_body)
        translate_body_space[..., 0:3, 3] = -transform_space_body[..., 0:3, 3]

        # Calculate the ee jacobian in space frame
        translate_body_space_adjoint = \
            raw_spatial_methods.transform_adjoint(translate_body_space)
        ee_jacobian_space_frame = \
            translate_body_space_adjoint @ space_jacobian

        return transform_space_body, ee_jacobian_space_frame


class Robot:
    """Abstract class for a robot."""

    def __init__(self,
                 name: str,
                 base_link: Link,
                 home_position: Optional[Dict[str, List[float]]] = None,
                 device: Optional[torch.device] = None):

        if device is None:
            device = base_link.device

        self.base_link = base_link

        self.name = name

        self.device = device

    def make_kinematics(self, query_link_names: Iterable[str]) -> KinematicsObject:

        # here we create a list of the joint chains that move each query link
        query_link_joint_chains = {link_name: self.base_link.find_link_joint_chain(link_name)
                                   for link_name in query_link_names}

        # Here we create a dict that maps each relevant joint name to the joint object
        relevant_joints = {joint.name: joint
                           for query_link_joint_chain in query_link_joint_chains.values()
                           for joint in query_link_joint_chain}

        # Here we create a dict that maps each relevant joint name to its parent link name
        joint_parent_link_name_dict = self.base_link.get_joint_parent_links_dict()

        # Here we create a set of all of the relevant links to the query links, including the query links themselves
        relevant_link_names = \
            {*joint_parent_link_name_dict.values(), *query_link_names}

        # Now we compute the zero transforms of all of the relevant links
        link_zero_transforms = self.base_link.get_zero_transforms(
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
        query_link_zero_transform_and_joint_chain_tuple_list = \
            [(link_zero_transforms[query_link_name],
              query_link_dof_chain_names[query_link_name])
             for query_link_name in query_link_names]

        # Here we unpack the list of tuples into two lists
        query_link_zero_transform_list = [zero_transform for zero_transform, joint_chain in
                                          query_link_zero_transform_and_joint_chain_tuple_list]
        query_link_joint_chain_list = [joint_chain for zero_transform, joint_chain in
                                       query_link_zero_transform_and_joint_chain_tuple_list]

        return KinematicsObject(dof_space_screws=dof_space_screw_dict,
                                query_link_zero_transforms=query_link_zero_transform_list,
                                query_link_dof_chains=query_link_joint_chain_list,)
