import torch

from rlloco.robots.robot import robot
from rlloco.robots.mini_cheetah.kinematics.mini_cheetah_kin_torch import build_jacobian_and_fk

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class mini_cheetah(robot):
    """This class respresents the mini_cheetah robot and stores information relevant to it
    Attributes:
        home_position (torch.Tensor):
            Contains the dof angles for the home position of the represented robot
        foot_indeces (torch.Tensor)
            Contains the indeces of the 4 feet links
    """

    home_position = torch.tensor([0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6]).to(device)

    foot_indeces = torch.tensor([3, 6, 9, 12])

    def foot_positions(q: torch.Tensor):
        """Calculates the foot positions given joint angles

        Args:
            q (torch.Tensor): _description_
        """


