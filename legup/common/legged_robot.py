from legup.common.kinematics import Joint, Link
from .spatial import Position
from .kinematics.inverse_kinematics import IKFunction, dls_ik, IKInput

from typing import Iterable, Optional, List, Dict
from tensor_types import TensorWrapper

import torch


class LeggedRobot:
    def __init__(self,
                 name: str,
                 base_link: Link,
                 primary_contacts: Iterable[str],
                 secondary_contacts: Iterable[str],
                 limited_joint_names: Iterable[str],
                 model_idx_dict: Dict[str, int],
                 home_position: Optional[Dict[str, List[float]]] = None):
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
        self.limited_joint_names = limited_joint_names

        self.model_idx_dict = model_idx_dict

        self.primary_contact_kinematics = \
            base_link.make_kinematics(self.primary_contacts)

        if home_position is not None:
            self.home_position = torch.zeros(
                len(self.primary_contact_kinematics.dof_order),
                device=self.base_link.device)

        self.home_position_transforms = \
            self.primary_contact_kinematics(self.home_position).transform

    def relative_pos_ik(self,
                        target_pos: Position,
                        current_angles: torch.Tensor,
                        ik_func: IKFunction = dls_ik):

        kin_result = self.primary_contact_kinematics(current_angles)

        pos_error = target_pos - kin_result.transform.extract_translation()

        ik_func.apply()



        angle_delta = ik_func(pos_error, kin_result.jacobian)

    @property
    def dof_order(self) -> List[str]:

        return self.primary_contact_kinematics.dof_order

    @property
    def primary_contact_link_model_idxs(self) -> List[int]:

        return [self.model_idx_dict[primary_contact_name]
                for primary_contact_name in self.primary_contacts]

    @property
    def secondary_contact_link_model_idxs(self) -> List[int]:

        return [self.model_idx_dict[secondary_contact_name]
                for secondary_contact_name in self.secondary_contacts]

    @property
    def limited_joint_model_idxs(self) -> List[int]:

        return [self.model_idx_dict[limited_joint_name]
                for limited_joint_name in self.limited_joint_names]
