from isaacgym import gymapi, gymtorch

from legup.isaac.simulator import SimulatorContext

from typing import List
import numpy as np

import torch
import pytorch3d.transforms.rotation_conversions as R

class IsaacGymEnvironment:
    """Interfaces with IsaacGym to handle all of the simulation, and provides an API to get simulation properties and move the robot.
    This implementation creates a number of parallel environments, and adds one agent/robot to each environment.
    """

    def __init__(self, num_environments: int, use_cuda: bool, asset_root: str, asset_path: str, default_dof_pos: torch.Tensor):
        """
        Args:
            num_environments (int): number of parallel environments to create in simulator
            use_cuda (bool): whether or not to use the GPU pipeline
            asset_root (str): root folder where the URDF to load is
            asset_path (str): path to URDF file from `asset_root`
            default_dof_pos (torch.Tensor): Joint positions to set for all robots when they are initialized or reset
        """

        self.ctx = SimulatorContext(
            num_environments, use_cuda, asset_root, asset_path)
        self.num_environments = num_environments
        self.gym = self.ctx.gym
        self.sim = self.ctx.sim
        self.env_actor_handles = self.ctx.env_actor_handles
        self.camera_env = 0
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.all_env_index = torch.Tensor(
            range(num_environments)).to(torch.int).to(self.device)

        # TODO: parameterize this
        self.default_dof_pos = default_dof_pos
        self.command_dof_pos = self.default_dof_pos.repeat(num_environments, 1)

        self._acquire_state_tensors()

        # self.reset(self.all_env_index) # not sure if I can actually do this
        self.refresh_buffers()

        self._init_camera(640, 480)

    def _init_camera(self, width: int, height):
        """Creates a camera object in the simulator so that it can be visualized in headless mode

        Args:
            width (int): Width (in pixels) of camera
            height (int): Height (in pixels) of camera
        """
        camera_props = gymapi.CameraProperties()
        camera_props.width = width
        camera_props.height = height
        # camera_props.use_collision_geometry = True
        self.camera_handle = self.gym.create_camera_sensor(
            self.env_actor_handles[self.camera_env][0], camera_props)

        camera_offset = gymapi.Vec3(-0.5, -0.5, 1)
        camera_rotation = gymapi.Quat().from_euler_zyx(
            np.radians(0), np.radians(45), np.radians(45))
        body_handle = self.gym.get_actor_rigid_body_handle(
            self.env_actor_handles[self.camera_env][0], self.env_actor_handles[self.camera_env][1], 0)
        self.gym.attach_camera_to_body(self.camera_handle, self.env_actor_handles[self.camera_env][0], body_handle, gymapi.Transform(
            camera_offset, camera_rotation), gymapi.FOLLOW_POSITION)
        self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(
            0.8, 0.8, 0.8), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0, 0, 0))

        self.cam_width = width
        self.cam_height = height

    def _acquire_state_tensors(self):
        """Acquires the tensors that contain all of the robot's physical properties (dynamics), and creates
        several views to make reading/writing easier
        """
        _root_states = self.gym.acquire_actor_root_state_tensor(
            self.sim)  # (num_actors, (pos + rot + linvel + angvel) = 13)
        self.root_states = gymtorch.wrap_tensor(
            _root_states).view(self.num_environments, 13)
        self.root_position = self.root_states[:, :3]
        self.root_rotation = self.root_states[:, 3:7]
        self.root_lin_vel = self.root_states[:, 7:10]
        self.root_ang_vel = self.root_states[:, 10:]

        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(
            _dof_states).view(self.num_environments, -1, 2)
        self.dof_pos = self.dof_states[:, :, 0]
        self.dof_vel = self.dof_states[:, :, 1]
        self.num_dof = self.dof_states.shape[1]

        _dof_forces = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_forces = gymtorch.wrap_tensor(_dof_forces).view(
            self.num_environments, self.num_dof)

        _net_contact_forces = self.gym.acquire_net_contact_force_tensor(
            self.sim)  # (num_rb, 3)
        self.net_contact_forces = gymtorch.wrap_tensor(
            _net_contact_forces).view(self.num_environments, -1, 3)

        _rb_states = self.gym.acquire_rigid_body_state_tensor(
            self.sim)  # (num_rb, (pos + rot + linvel + angvel) = 13)
        self.rb_states = gymtorch.wrap_tensor(
            _rb_states).view(self.num_environments, -1, 13)
        self.rb_pos = self.rb_states[:, :, :3]
        self.rb_rot = self.rb_states[:, :, 3:7]
        self.rb_lin_vel = self.rb_states[:, :, 7:10]
        self.rb_ang_vel = self.rb_states[:, :, 10:]
        self.num_rb = self.rb_states.shape[1]

    def print_nan(self, tensor, label):
        if (torch.any(torch.isnan(tensor))):
            idxs = torch.unique(torch.argwhere(torch.isnan(tensor))[:, 0])
            print(f'FOUND NAN IN {label} AT INDEX {idxs}')

    def refresh_buffers(self):
        """Updates the data in the state tensors, must be called after stepping the simulation"""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.print_nan(self.root_states, 'ROOT_STATES')
        self.print_nan(self.dof_states, 'DOF_STATES')
        self.print_nan(self.dof_forces, 'DOF_FORCES')
        self.print_nan(self.net_contact_forces, 'NET_CONTACT')
        self.print_nan(self.rb_states, 'RB_STATES')


    def get_position(self) -> torch.Tensor:
        """Gets the root position of each robot

        Returns:
            torch.Tensor: shape `(num_environments, 3)`
        """
        return self.root_position

    def get_rotation(self) -> torch.Tensor:
        """Gets the root rotation (as quaternion) of each robot

        Returns:
            torch.Tensor: shape `(num_environments, 4)`
        """
        return self.root_rotation

    def get_linear_velocity(self) -> torch.Tensor:
        """Gets the root linear velocity of each robot

        Returns:
            torch.Tensor: shape `(num_environments, 3)`
        """
        return self.root_lin_vel

    def get_angular_velocity(self) -> torch.Tensor:
        """Gets the root angular velocity of each robot

        Returns:
            torch.Tensor: shape `(num_environments, 3)`
        """
        return self.root_ang_vel

    def get_joint_position(self) -> torch.Tensor:
        """Gets the joint positions of each robot

        Returns:
            torch.Tensor: shape `(num_environments, num_degrees_of_freedom)`
        """
        return self.dof_pos

    def get_joint_velocity(self) -> torch.Tensor:
        """Gets the joint velocities of each robot

        Returns:
            torch.Tensor: shape `(num_environments, num_degrees_of_freedom)`
        """
        return self.dof_vel

    def get_joint_torque(self) -> torch.Tensor:
        """Gets the joint torques of each robot

        Returns:
            torch.Tensor: shape `(num_environments, num_degrees_of_freedom)`
        """
        return self.dof_forces

    def get_rb_position(self) -> torch.Tensor:
        """Gets the rigid body positions of each robot

        Returns:
            torch.Tensor: shape `(num_environments, num_rigid_bodies, 3)`
        """
        return self.rb_pos

    def get_rb_rotation(self) -> torch.Tensor:
        """Gets the rigid body rotations (as quaternion) of each robot

        Returns:
            torch.Tensor: shape `(num_environments, num_rigid_bodies, 4)`
        """
        return self.rb_rot

    def get_rb_linear_velocity(self) -> torch.Tensor:
        """Gets the rigid body linear velocities of each robot

        Returns:
            torch.Tensor: shape `(num_environments, num_rigid_bodies, 3)`
        """
        return self.rb_lin_vel

    def get_rb_angular_velocity(self) -> torch.Tensor:
        """Gets the rigid body angular velocities of each robot

        Returns:
            torch.Tensor: shape `(num_environments, num_rigid_bodies, 3)`
        """
        return self.rb_ang_vel

    def get_contact_states(self, collision_thresh: float = 1) -> torch.Tensor:
        """Gets whether or not each rigid body has collided with anything

        Args:
            collision_thresh (int, optional): Collision force threshold. Defaults to 1.

        Returns:
            torch.Tensor: truthy tensor, shape `(num_environments, num_rigid_bodies)`
        """
        contact_forces = torch.norm(self.net_contact_forces, dim=2)
        collisions = contact_forces > collision_thresh
        return collisions

    def get_contact_forces(self) -> torch.Tensor:
        """Gets the contact forces action on each rigid body

        Returns:
            torch.Tensor: shape `(num_environments, num_rigid_bodies, 3)`
        """
        return self.net_contact_forces

    def step(self, actions: torch.Tensor = None):
        """Moves robots using `actions`, steps the simulation forward, updates graphics, and refreshes state tensors

        Args:
            actions (torch.Tensor, optional): target joint positions to command each robot, shape `(num_environments, num_degrees_of_freedom)`. 
                If none, robots are commanded to the default joint position provided earlier Defaults to None.
        """

        if actions is not None:
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
            if isinstance(actions, torch.Tensor):
                actions = gymtorch.unwrap_tensor(actions)
            
        else:
            actions = gymtorch.unwrap_tensor(self.command_dof_pos)

        self.gym.set_dof_position_target_tensor(self.sim, actions)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

    def render(self) -> torch.Tensor:
        """Gets an image of the environment from the camera and returns it

        Returns:
            np.ndarray: RGB image, shape `(camera_height, camera_width, 4)`
        """
        return self.gym.get_camera_image(self.sim, self.env_actor_handles[self.camera_env][0], self.camera_handle, gymapi.IMAGE_COLOR).reshape(self.cam_height, self.cam_width, 4)

    def reset(self, env_index: List[int] = None):
        """Resets the specified robot. Specifically, it will move it to a random position, give it zero velocity, and drop it from a height of 0.28 meters.

        Args:
            env_index (list, torch.Tensor, optional): Indices of environments to reset. If none, all environments are reset. Defaults to None.
        """
        if env_index is None:
            env_index = self.all_env_index
        else:
            env_index = self.all_env_index[env_index]

        random_pos = torch.rand(len(env_index), 3) * 2
        random_pos[:, 2] = 0.40

        # TODO: make faster for cuda?
        random_rot = torch.zeros(len(env_index), 3)
        random_rot[:, 0] = torch.deg2rad(torch.rand(len(env_index)) * 360)
        random_rot[:, 1] = 0
        random_rot[:, 2] = np.radians(180)
        random_rot = R.matrix_to_quaternion(
            R.euler_angles_to_matrix(random_rot, convention='XYZ'))

        idx_tensor = env_index.long()  # why can't I index with int32 tensors :(
        self.root_position[idx_tensor, :] = random_pos.to(self.device)
        self.root_lin_vel[idx_tensor, :] = 0
        self.root_ang_vel[idx_tensor, :] = 0
        self.root_rotation[idx_tensor, :] = random_rot.to(self.device)
        self.dof_pos[idx_tensor, :] = self.default_dof_pos
        self.dof_vel[idx_tensor, :] = 0

        indices = gymtorch.unwrap_tensor(env_index)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,  gymtorch.unwrap_tensor(self.dof_states), indices, len(env_index))
