import torch

from typing import Dict, Any

from abc import ABC, abstractmethod


class AbstractDynamics(ABC):
    @abstractmethod
    def get_position(self) -> torch.Tensor:
        """Gets the root position of each robot
        Returns:
            torch.Tensor: shape `(num_environments, 3)`
        """
        pass

    @abstractmethod
    def get_rotation(self) -> torch.Tensor:
        """Gets the root rotation (as quaternion) of each robot
        Returns:
            torch.Tensor: shape `(num_environments, 4)`
        """
        pass

    @abstractmethod
    def get_linear_velocity(self) -> torch.Tensor:
        """Gets the root linear velocity of each robot
        Returns:
            torch.Tensor: shape `(num_environments, 3)`
        """
        pass

    @abstractmethod
    def get_angular_velocity(self) -> torch.Tensor:
        """Gets the root angular velocity of each robot
        Returns:
            torch.Tensor: shape `(num_environments, 3)`
        """
        pass

    @abstractmethod
    def get_joint_position(self) -> torch.Tensor:
        """Gets the joint positions of each robot
        Returns:
            torch.Tensor: shape `(num_environments, num_degrees_of_freedom)`
        """
        pass

    @abstractmethod
    def get_joint_position_hist(self) -> torch.Tensor:
        """Gets the joint positions of each robot over the last few environment steps TODO decide oh a history length

        Returns:
            torch.Tensor: shape `(hist_length, num_environments, num_degrees_of_freedom)`. Index 0 is most recent, -1 is the oldest
        """
        pass

    @abstractmethod
    def get_joint_velocity(self) -> torch.Tensor:
        """Gets the joint velocities of each robot
        Returns:
            torch.Tensor: shape `(num_environments, num_degrees_of_freedom)`
        """
        pass

    @abstractmethod
    def get_joint_velocity_hist(self) -> torch.Tensor:
        """Gets the joint velocities of each robot over the last few environment steps TODO: decide on a hist length

        Returns:
            torch.Tensor: shape `(hist_length, num_environments, num_degrees_of_freedom)`. Index 0 is most recent, -1 is the oldest
        """

    @abstractmethod
    def get_joint_torque(self) -> torch.Tensor:
        """Gets the joint torques of each robot
        Returns:
            torch.Tensor: shape `(num_environments, num_degrees_of_freedom)`
        """
        pass

    @abstractmethod
    def get_rb_position(self) -> torch.Tensor:
        """Gets the rigid body positions of each robot
        Returns:
            torch.Tensor: shape `(num_environments, num_rigid_bodies, 3)`
        """
        pass

    @abstractmethod
    def get_rb_rotation(self) -> torch.Tensor:
        """Gets the rigid body rotations (as quaternion) of each robot
        Returns:
            torch.Tensor: shape `(num_environments, num_rigid_bodies, 4)`
        """
        pass

    @abstractmethod
    def get_rb_linear_velocity(self) -> torch.Tensor:
        """Gets the rigid body linear velocities of each robot
        Returns:
            torch.Tensor: shape `(num_environments, num_rigid_bodies, 3)`
        """
        pass

    @abstractmethod
    def get_rb_angular_velocity(self) -> torch.Tensor:
        """Gets the rigid body angular velocities of each robot
        Returns:
            torch.Tensor: shape `(num_environments, num_rigid_bodies, 3)`
        """
        pass

    @abstractmethod
    def get_contact_states(self, collision_thresh: float = 1) -> torch.Tensor:
        """Gets whether or not each rigid body has collided with anything
        Args:
            collision_thresh (int, optional): Collision force threshold. Defaults to 1.
        Returns:
            torch.Tensor: truthy tensor, shape `(num_environments, num_rigid_bodies)`
        """
        pass

    @abstractmethod
    def get_contact_forces(self) -> torch.Tensor:
        """Gets the contact forces action on each rigid body
        Returns:
            torch.Tensor: shape `(num_environments, num_rigid_bodies, 3)`
        """
        pass

    @abstractmethod
    def get_num_agents(self) -> int:
        """Gets number of agents in the entire simulation

        Returns:
            int: number of agents running
        """
        pass

    @abstractmethod
    def get_dt(self) -> float:
        """Gets the current timestep

        Returns:
            float: timestep
        """
        pass
