import torch

from rlloco.robots.robot import robot
from rlloco.robots.mini_cheetah.kinematics.mini_cheetah_kin_torch import build_jacobian_and_fk_all_feet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MiniCheetah(robot):
    """This class respresents the mini_cheetah robot and stores information relevant to it
    Attributes:
        home_position (torch.Tensor):
            Contains the dof angles for the home position of the represented robot
        foot_indices (torch.Tensor)
            Contains the indices of the 4 feet links
    """

    home_position = torch.tensor(
        [0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6], device=device)

    foot_indices = [3, 6, 9, 12]

    knee_indices = [2, 5, 8, 11]

    thigh_indices = [2, 5, 8, 11]

    def foot_positions(q: torch.Tensor):
        """Calculates the foot positions given joint angles

        Args:
            q (torch.Tensor): _description_
        """

        jacobian, foot_pos = build_jacobian_and_fk_all_feet(q)

        return foot_pos

    def foot_twist(q: torch.Tensor):
        """Calculates the 6 dimensional twist vector for the feet

        Args:
            q (torch.Tensor): A (NUM_ENVS x 12) vector containing the joint positions for the robot

        Returns:
            (torch.Tensor): A (NUM_ENVS x 4 x 6) vector containing the twist vectors for each of the robots feet
        """
        raise NotImplementedError("The foot_twist function has not been implemented for the mini_cheetah robot")