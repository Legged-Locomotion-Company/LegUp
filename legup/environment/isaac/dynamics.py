from isaacgym import gymtorch

import torch
from typing import Callable

from legup.common.abstract_dynamics import AbstractDynamics


class TensorTracker:
    """Tracks when a state tensor is modified and makes sure it is only copied to simulation if it has been modified"""

    def __init__(self, state_tensor: torch.Tensor, tensor_setter: Callable[..., None]):
        """
        Args:
            state_tensor (torch.Tensor): state tensor acquired from simulation
            tensor_setter (Callable[..., None]): function that copies the tensor to simulator
        """
        self.state_tensor = state_tensor
        self.tensor_setter = tensor_setter
        self.last_version = self.state_tensor._version

    def update(self, sim):
        """Sends the tensor data to the simulator only if its been modified

        Args:
            sim (Sim): IsaacGym handle to simulator
        """
        if self.state_tensor._version != self.last_version:
            gymtensor = gymtorch.unwrap_tensor(self.state_tensor)
            self.tensor_setter(sim, gymtensor)
            self.last_version = self.state_tensor._version


class IsaacGymDynamics(AbstractDynamics):
    """IsaacGym implementation of environment dynamics, allows you to access and set the kinematic properties of the simulation"""

    def __init__(self, sim, gym, num_agents: int):
        """
        Args:
            sim (Sim): IsaacGym handle to simulation
            gym (Gym): IsaacGym handle to gym
            num_agents (int): number of parallel agents we have in the simulation
        """
        self.sim = sim
        self.gym = gym
        self.num_agents = num_agents
        self.state_tensors = []

        self._acquire_state_tensors()

    def _acquire_state_tensors(self):
        """Initializes all the state tensors"""
        _root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(
            _root_states).view(self.num_agents, 13)
        self.root_position = self.root_states[:, :3]
        self.root_rotation = self.root_states[:, 3:7]
        self.root_lin_vel = self.root_states[:, 7:10]
        self.root_ang_vel = self.root_states[:, 10:]
        self.state_tensors.append(TensorTracker(
            self.root_states, self.gym.set_actor_root_state_tensor))

        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(
            _dof_states).view(self.num_agents, -1, 2)
        self.dof_pos = self.dof_states[:, :, 0]
        self.dof_vel = self.dof_states[:, :, 1]
        self.num_dof = self.dof_states.shape[1]
        self.state_tensors.append(TensorTracker(
            self.dof_states, self.gym.set_dof_state_tensor))

        _dof_forces = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_forces = gymtorch.wrap_tensor(
            _dof_forces).view(self.num_agents, self.num_dof)

        _net_contact_forces = self.gym.acquire_net_contact_force_tensor(
            self.sim)
        self.net_contact_forces = gymtorch.wrap_tensor(
            _net_contact_forces).view(self.num_agents, -1, 3)

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(
            _rb_states).view(self.num_agents, -1, 13)
        self.rb_pos = self.rb_states[:, :, :3]
        self.rb_rot = self.rb_states[:, :, 3:7]
        self.rb_lin_vel = self.rb_states[:, :, 7:10]
        self.rb_ang_vel = self.rb_states[:, :, 10:]
        self.state_tensors.append(TensorTracker(
            self.rb_states, self.gym.set_rigid_body_state_tensor))

    def refresh_buffers(self):
        """Updates the data in the state tensors, must be called after stepping the simulation"""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def apply_tensor_changes(self):
        """Copies all local changes done to any tensors into the simulation environment. **Must only be called once between steps**"""
        for tracker in self.state_tensors:
            tracker.update(self.sim)

    def get_position(self) -> torch.Tensor:
        """Gets the root position of each robot
        Returns:
            torch.Tensor: shape `(num_agents, 3)`
        """
        return self.root_position

    def get_rotation(self) -> torch.Tensor:
        """Gets the root rotation (as quaternion) of each robot
        Returns:
            torch.Tensor: shape `(num_agents, 4)`
        """
        return self.root_rotation

    def get_linear_velocity(self) -> torch.Tensor:
        """Gets the root linear velocity of each robot
        Returns:
            torch.Tensor: shape `(num_agents, 3)`
        """
        return self.root_lin_vel

    def get_angular_velocity(self) -> torch.Tensor:
        """Gets the root angular velocity of each robot
        Returns:
            torch.Tensor: shape `(num_agents, 3)`
        """
        return self.root_ang_vel

    def get_joint_position(self) -> torch.Tensor:
        """Gets the joint positions of each robot
        Returns:
            torch.Tensor: shape `(num_agents, num_degrees_of_freedom)`
        """
        return self.dof_pos

    def get_joint_velocity(self) -> torch.Tensor:
        """Gets the joint velocities of each robot
        Returns:
            torch.Tensor: shape `(num_agents, num_degrees_of_freedom)`
        """
        return self.dof_vel

    def get_joint_torque(self) -> torch.Tensor:
        """Gets the joint torques of each robot
        Returns:
            torch.Tensor: shape `(num_agents, num_degrees_of_freedom)`
        """
        return self.dof_forces

    def get_rb_position(self) -> torch.Tensor:
        """Gets the rigid body positions of each robot
        Returns:
            torch.Tensor: shape `(num_agents, num_rigid_bodies, 3)`
        """
        return self.rb_pos

    def get_rb_rotation(self) -> torch.Tensor:
        """Gets the rigid body rotations (as quaternion) of each robot
        Returns:
            torch.Tensor: shape `(num_agents, num_rigid_bodies, 4)`
        """
        return self.rb_rot

    def get_rb_linear_velocity(self) -> torch.Tensor:
        """Gets the rigid body linear velocities of each robot
        Returns:
            torch.Tensor: shape `(num_agents, num_rigid_bodies, 3)`
        """
        return self.rb_lin_vel

    def get_rb_angular_velocity(self) -> torch.Tensor:
        """Gets the rigid body angular velocities of each robot
        Returns:
            torch.Tensor: shape `(num_agents, num_rigid_bodies, 3)`
        """
        return self.rb_ang_vel

    def get_contact_states(self, collision_thresh: float = 1) -> torch.Tensor:
        """Gets whether or not each rigid body has collided with anything
        Args:
            collision_thresh (int, optional): Collision force threshold. Defaults to 1.
        Returns:
            torch.Tensor: truthy tensor, shape `(num_agents, num_rigid_bodies)`
        """
        contact_forces = torch.norm(self.net_contact_forces, dim=2)
        collisions = contact_forces > collision_thresh
        return collisions

    def get_contact_forces(self) -> torch.Tensor:
        """Gets the contact forces action on each rigid body
        Returns:
            torch.Tensor: shape `(num_agents, num_rigid_bodies, 3)`
        """
        return self.net_contact_forces
