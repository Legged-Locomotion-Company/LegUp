from legup.common.kinematics import Joint, Link
from legup.common.spatial import Position
from legup.common.kinematics.inverse_kinematics import IKFunction, dls_ik

from typing import Iterable, Optional, List, Dict, Tuple, Union, TypeVar
from tensor_types import TensorWrapper

import torch


class LeggedRobot:
    """_summary_
    """

    def __init__(self,
                 name: str,
                 base_link: Link,
                 primary_contacts: Iterable[str],
                 secondary_contacts: Iterable[str],
                 model_idx_dict: Dict[str, int],
                 dof_limits: Union[None,
                                   Dict[str, Tuple[Optional[float], Optional[float]]],
                                   #    Iterable[Tuple[Optional[float],  TODO: Add this back in when I'm less lazy
                                   #                   Optional[float]]],
                                   torch.Tensor
                                   ] = None,
                 home_position: Union[None,
                                      Iterable[float],
                                      torch.Tensor
                                      ] = None):
        """_summary_

        Args:
            name (str): Name of this robot
            base_link (Link): The kinematics root of this robot
            primary_contacts (Iterable[str]): This is a list of the names of the links that should be in contact with the ground during a normal walking gait.
            secondary_contacts (Iterable[str]): This is a list of the names of the links that could be in conctact with the ground in a non-catastrophic failure
            limited_joint_names (Iterable[str]): This is a list of the names of the joints that are limited by an asthetic constraint such as knees
            home_position (Optional[Dict[str, List[float]]], optional): This is a way to set the default positions for the joints,
                kinematics will be calculated as a delta to these. Defaults to None.
            device (torch.device, optional): The device for this robot's data. Defaults to TensorWrapper._default_device().
        """

        self.name = name

        self.base_link = base_link

        self.primary_contacts = primary_contacts
        self.secondary_contacts = secondary_contacts

        self.model_idx_dict = model_idx_dict

        self.primary_contact_kinematics = \
            base_link.make_kinematics(self.primary_contacts)

        self.set_home_position(home_position)

        if dof_limits is not None:
            self.set_dof_limits(dof_limits)

    def relative_pos_ik(self,
                        relative_targets: Position,
                        current_angles: torch.Tensor,
                        ik_func: IKFunction = dls_ik
                        ) -> torch.Tensor:

        absolute_targets = \
            self.home_position_transforms.extract_translation() + relative_targets
        new_angles = \
            dls_ik.apply(absolute_targets, current_angles,
                         self.primary_contact_kinematics)

        return new_angles

    def set_dof_limits(self, dof_limits: Union[Dict[str, Tuple[Optional[float], Optional[float]]], torch.Tensor]) -> None:

        ndof = len(self.primary_contact_kinematics.dof_order)

        if isinstance(dof_limits, dict):
            self.dof_limits = torch.empty(
                ndof, 2, device=self.base_link.device)
            self.dof_limits[:, 0] = -torch.inf
            self.dof_limits[:, 1] = torch.inf

            for dof_name, (dof_low, dof_high) in dof_limits.items():
                dof_idx = self.dof_order.index(dof_name)
                if dof_low is not None:
                    self.dof_limits[dof_idx, 0] = dof_low
                if dof_high is not None:
                    self.dof_limits[dof_idx, 1] = dof_high

        elif isinstance(dof_limits, torch.Tensor):
            if dof_limits.shape[0] != ndof or dof_limits.shape[1] != 2 or len(dof_limits.shape) != 2:
                raise ValueError(
                    f"Recieved dof_limits with shape {dof_limits.shape} but expected ({ndof}, 2)")
            else:
                dof_limits = dof_limits.to(self.base_link.device)

    def set_home_position(self, home_position: Union[Iterable[float], torch.Tensor, None]) -> None:

        if isinstance(home_position, Iterable):
            home_position = torch.tensor(list(home_position),
                                         device=self.base_link.device)
        elif home_position is None:
            home_position = torch.zeros(
                len(self.primary_contact_kinematics.dof_order), device=self.base_link.device)

        home_position = home_position.to(self.base_link.device)

        ndof = len(self.primary_contact_kinematics.dof_order)

        if home_position.shape[0] != ndof or len(home_position.shape) != 1:
            raise ValueError(
                f"Recieved home position with shape {home_position.shape} but expected {ndof}")

        self.home_position = home_position

        self.home_position_transforms = \
            self.primary_contact_kinematics(self.home_position).transform

    @ staticmethod
    def _default_device() -> torch.device:
        return TensorWrapper._default_device()

    @ property
    def dof_order(self) -> List[str]:

        return self.primary_contact_kinematics.dof_order

    @ property
    def primary_contact_link_model_idxs(self) -> List[int]:

        return [self.model_idx_dict[primary_contact_name]
                for primary_contact_name in self.primary_contacts]

    @ property
    def secondary_contact_link_model_idxs(self) -> List[int]:

        return [self.model_idx_dict[secondary_contact_name]
                for secondary_contact_name in self.secondary_contacts]

    @ property
    def num_dofs(self) -> int:
        return len(self.dof_order)
