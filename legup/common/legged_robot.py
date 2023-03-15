from robot import Robot

from typing import Iterable, Optional, List, Dict
from tensor_types import TensorWrapper
from legup.common.link_joint import Link

import torch


class LeggedRobot(Robot):
    def __init__(self,
                 name: str,
                 base_link: Link,
                 foot_link_names: Iterable[str],
                 shank_link_names: Iterable[str],
                 knee_joint_names: Iterable[str],
                 home_position: Optional[Dict[str, List[float]]] = None,
                 device: torch.device = TensorWrapper._default_device()):

        super().__init__(base_link=base_link,
                         name=name,
                         home_position=home_position,
                         device=device)

        self.foot_link_names = foot_link_names
        self.shank_link_names = shank_link_names
        self.knee_joint_names = knee_joint_names

        self.foot_kinematics = self.make_kinematics(self.foot_link_names)

    @property
    def dof_order(self) -> List[str]:

        return self.foot_kinematics.dof_order

    @property
    def foot_link_indices(self) -> List[int]:

        return self.foot_kinematics.get_dof_idxs(self.foot_link_names)

    @property
    def shank_link_indices(self) -> List[int]:

        return self.foot_kinematics.get_dof_idxs(self.shank_link_names)

    @property
    def knee_joint_indices(self) -> List[int]:

        return self.foot_kinematics.get_dof_idxs(self.knee_joint_names)
