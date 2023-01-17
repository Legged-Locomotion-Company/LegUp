from typing import List

import torch


class Robot:
    """An abstract class to be inherited by specific robot classes

    Attributes:
        home_position (torch.Tensor):
            To be implemented by a subclass
            Contains the dof angles for the home position of the represented robot

        foot_indices (List[int]):
            To be set in a subclass.
            Contains the indices for the links which are feet
        
        knee_indices (List[int]):
            To be set in a subclass.
            Contains the indices for the joints which are knees
        
        thigh_indices (List[int]):
            To be set in a subclass.
            Contains the indices for the links which immediately precede the knee links
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # This should be a (NDOF) dimension tensor with joint positions for the robot at its home position
    home_position: torch.Tensor

    # This one should be a list which contains the indices of all of the bodies which are feet and should be allowed to contact the ground
    foot_indices: List[int]

    # This one should be a list which contains the indices of all of the joints which are knees
    knee_indices: List[int]

    # This one should be a list which contains the indices of all of links which immediately precede the knee links
    thigh_indices: List[int]

    # This should be a list which contains the indices of all of the links which immediately follow knees. In quads sometimes this is the same as the feet
    shank_indeces: List[int]

    num_joints: int
    

    def foot_positions(q: torch.Tensor) -> torch.Tensor:
        """Calculates the foot position for the robot

        Args:
            q (torch.Tensor): This is a (NUM_ENVS x NUM_DOFS) vector containing the positions of each of the robots joints in each environment

        Raises:
            NotImplementedError: This error is thrown if the function is called on the base class, or if the function is not implemented in a subclass

        Returns:
            (torch.Tensor): This is a (NUM_ENVS x NUM_FEET x 3) vector containing the (x,y,z) coordinates of each of the robots feet in the robots local coordinates
        """

        raise NotImplementedError(
            "Cannot call foot_positions function of abstract robot base class")

    def foot_twist(q: torch.Tensor) -> torch.Tensor:
        """ Calculates the 6 dimensional twist vector for the foot positions

        Args:
            q (torch.Tensor): This is a (NUM_ENVS x NUM_DOFS) vector containing the positions of each of the robots joints in each environment

        Raises:
            NotImplementedError: This error is thrown if the function is called on the base class, or if the function is not implemented in a subclass

        Returns:
            (torch.Tensor): This is a (NUM_ENVS x NUM_FEET x 6) vector containing the twist vectors for each of the robots feet in the robots local coordinates
        """

        raise NotImplementedError(
            "Cannot call foot_twist function of abstract robot base class")

    def foot_jacobians(q: torch.Tensor) -> torch.Tensor:
        """ Calculates the jacobian for the feet relative to their joints

        Args:
            q (torch.Tensor): This is a (NUM_ENVS x NUM_DOFS) vector containing the positions of each of the robots joints in each environment

        Raises:
            NotImplementedError: This error is thrown if the function is called on the base class, or if the function is not implemented in a subclass

        Returns:
            (torch.Tensor): This is a (NUM_ENVS x NUM_FEET x 6 x NUM_DOFS) vector containing the twist vectors for each of the robots feet in the robots local coordinates
        """

        raise NotImplementedError(
            "Cannot call foot_twist function of abstract robot base class")
