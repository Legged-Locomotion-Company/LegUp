import torch

from typing import Dict, List, Optional, Union, Iterable, Tuple, NamedTuple

from legup.tensor_types import TensorWrapper, TensorIndexer, WrappedScalar
from legup.spatial import Transform, Screw, ScrewJacobian, raw_spatial_methods


class FKResult:
    def __init__(self,
                 transform: Transform,
                 jacobian: ScrewJacobian):
        self.transform = transform
        self.jacobian = jacobian

    @staticmethod
    def stack(kinematics_results: List["FKResult"]):
        return FKResult(
            transform=Transform.stack(
                [result.transform for result in kinematics_results], dim=-1),
            jacobian=ScrewJacobian.stack(
                [result.jacobian for result in kinematics_results], dim=-1))


class QueryLinkInfo(NamedTuple):
    zero_transform: Transform
    dof_chain_names: List[str]


class FKFunction:
    def __init__(self,
                 dof_space_screws: Union[Dict[str, Screw], TensorIndexer[Screw]],
                 query_link_infos: List[QueryLinkInfo],
                 dof_order: Optional[Iterable[str]] = None,):
        """Initializes a kinematics object.

        Args:
            dof_space_screws(Union[Dict[str, Screw], TensorIndexer[Screw]]): This is mapping from dof names to their screws in space frame
            query_link_zero_transforms (List[QueryLinkInfo]): This is a list of zero transforms and dof chains for links to compute in the kinematics function
            dof_order (Optional[Iterable[str]], optional): The order of dofs for calls to this object. Passing this as None will use the order of the keys in dof_body_screw_skews.
        """

        # Check to make sure every dof name in every link dof chain is in the dof_space_screws
        for _, link_dof_chain in query_link_infos:
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

        self.query_link_infos = query_link_infos

        self.dof_space_screws = dof_space_screws.reordered(self.dof_order)

    def get_dof_idxs(self, dof_names: Iterable[str]) -> List[int]:
        """This function returns the indices of the dofs in the dof_names list.

        Args:
            dof_names (Iterable[str]): This is a list of dof names

        Returns:
            torch.Tensor: This is a list of indices of the dofs in the dof_names list
        """

        return [self.dof_order.index(dof_name) for dof_name in dof_names]

    def __call__(self, joint_angles: torch.Tensor) -> FKResult:

        dof_exps = self.dof_space_screws.apply(Screw.apply, joint_angles)

        # Now for each query link, we compute the kinematics
        link_kinematics = \
            [FKFunction.single_kinematics(
                dof_space_screws=self.dof_space_screws,
                dof_exps=dof_exps,
                query_link_info=query_link_info)
             for query_link_info in self.query_link_infos]

        return FKResult.stack(link_kinematics)

    @staticmethod
    def single_kinematics(dof_space_screws: TensorIndexer[Screw],
                          dof_exps: TensorIndexer[Transform],
                          query_link_info: QueryLinkInfo) -> FKResult:
        """This function computes the forward kinematics and jacobian of a single chain of dofs.

        Args:
            dof_space_screws (TensorIndexer[Screw]): This is a TensorIndexer mapping from dof names to their screw skews in space frame
            dof_exps (TensorIndexer[Transform]): This is a TensorIndexer mapping from dof names to their scaled screw exps in space frame
            query_link_info (QueryLinkInfo): This is a tuple containing the zero transform and dof chain for this link

        Returns:
            KinematicsResult: This is a KinematicsResult object that contains the transforms and jacobian of the end effector
        """

        raw_transform, raw_jacobian = FKFunction.raw_single_kinematics(
            dof_space_screws=dof_space_screws.to_raw_dict(),
            dof_exps=dof_exps.to_raw_dict(),
            dof_idxs=dof_exps.idx_dict,
            dof_chain_names=query_link_info.dof_chain_names,
            zero_transform=query_link_info.zero_transform.tensor)

        return FKResult(Transform(raw_transform), ScrewJacobian(raw_jacobian))

    @staticmethod
    @torch.jit.script  # type: ignore
    def raw_single_kinematics(dof_space_screws: Dict[str, torch.Tensor],
                              dof_exps: Dict[str, torch.Tensor],
                              dof_idxs: Dict[str, int],
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
        device = dof_exps_element.device
        shape = dof_exps_element.shape[:-2]
        dtype = dof_exps_element.dtype

        # Now we create a list products for every dof_exp in the chain for every dof_exp leading up to it
        prev_dof_chain_prods: List[torch.Tensor] = \
            [torch.eye(4, device=device, dtype=dof_exps_element.dtype)
             .expand(shape + (4, 4))]

        # Now we populate that list iteratively
        for chain_dof_name in dof_chain_names:
            prev_dof_chain_prods.append(
                prev_dof_chain_prods[-1] @ dof_exps[chain_dof_name])

        # Now we create a tensor to hold the full jacobian
        space_jacobian = \
            torch.zeros(shape + (6, len(dof_idxs)), device=device, dtype=dtype)

        # space_jacobian_cols = [(raw_spatial_methods.transform_adjoint(prev_prod) @ dof_space_screws[dof_name])
        #                        for prev_prod, dof_name in zip(prev_dof_chain_prods, dof_chain_names)]
        # Here we calculate and assign the jacobian cols according to equation 5.11 in Modern Robotics
        for prev_prod, dof_name in zip(prev_dof_chain_prods, dof_chain_names):

            transform_adjoint = \
                raw_spatial_methods.transform_adjoint(prev_prod)
            space_jacobian[..., :, dof_idxs[dof_name]] = \
                transform_adjoint @ dof_space_screws[dof_name]

        # Now we calculate the end effector position according to the product of exponentials formula equation 5.9 and 4.14 in Modern Robotics
        transform_space_body = prev_dof_chain_prods[-1] @ zero_transform

        # Now I get the ee jacobian in space frame by multiplying the space jacobian by the adjoint of the transform which
        # translates the body frame to the space frame, without rotating it
        translate_body_space = raw_spatial_methods.eyes_like(
            transform_space_body)
        translate_body_space[..., 0:3, 3] = -transform_space_body[..., 0:3, 3]

        # Calculate the ee jacobian in space frame
        translate_body_space_adjoint = raw_spatial_methods.transform_adjoint(
            translate_body_space)
        ee_jacobian_space_frame = translate_body_space_adjoint @ space_jacobian

        return transform_space_body, ee_jacobian_space_frame
