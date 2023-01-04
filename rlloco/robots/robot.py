import torch

class robot:
    """An abstract class to be inherited by specific robot classes

    Attributes:
        home_position (torch.Tensor):
            To be implemented by a subclass
            Contains the dof angles for the home position of the represented robot
        
        foot_indeces (torch.Tensor):
            To be implemented by a subclass.
            Contains the indeces for the links which are feet
    """

    # This should be a (NDOF) dimension tensor with joint positions for the robot at its home position
    home_position: torch.Tensor

    # This one should be a tensor which contains the indeces of all of the links which are feet
    foot_indeces: torch.Tensor

    def foot_positions(q: torch.Tensor) -> torch.Tensor:
        """Calculates the foot position for the robot

        Args:
            q (torch.Tensor): This is a (NUM_ENVS x NUM_DOFS) vector containing the positions of each of the robots joints in each environment

        Raises:
            NotImplementedError: This error is thrown if the function is called on the base class, or if the function is not implemented in a subclass

        Returns:
            (torch.Tensor): This is a (NUM_ENVS x NUM_FEET x 3) vector containing the (x,y,z) coordinates of each of the robots feet in the robots local coordinates
        """
        
        raise NotImplementedError("Cannot call foot_positions function of abstract robot base class")