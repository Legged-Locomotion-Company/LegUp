from isaacgym import gymapi, gymtorch

import torch
import numpy as np
import pytorch3d.transforms.rotation_conversions as R

from rlloco.isaacgym.simulator import SimulatorContext

class IsaacGymEnvironment:
    def __init__(self, num_environments, use_cuda, asset_root, asset_path, default_dof_pos):
        self.ctx = SimulatorContext(num_environments, use_cuda, asset_root, asset_path)
        self.num_environments = num_environments
        self.gym = self.ctx.gym
        self.sim = self.ctx.sim
        self.env_actor_handles = self.ctx.env_actor_handles
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.all_env_index = torch.Tensor(range(num_environments)).to(torch.int).to(self.device)
        

        # TODO: parameterize this
        self.default_dof_pos = default_dof_pos
        self.command_dof_pos = self.default_dof_pos.repeat(num_environments, 1)

        self._acquire_state_tensors()
        self._refresh()
        self._init_camera(640, 480)
    
    def _init_camera(self, width, height):
        camera_props = gymapi.CameraProperties()
        camera_props.width = width
        camera_props.height = height
        self.camera_handle = self.gym.create_camera_sensor(self.env_actor_handles[0][0], camera_props)

        camera_offset = gymapi.Vec3(-0.5, -0.5, 1)
        camera_rotation = gymapi.Quat().from_euler_zyx(np.radians(0), np.radians(45), np.radians(45))
        body_handle = self.gym.get_actor_rigid_body_handle(self.env_actor_handles[0][0], self.env_actor_handles[0][1], 0)
        self.gym.attach_camera_to_body(self.camera_handle, self.env_actor_handles[0][0], body_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_POSITION)
        self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0, 0, 0))

        self.cam_width = width
        self.cam_height = height

    def _acquire_state_tensors(self):
        _root_states = self.gym.acquire_actor_root_state_tensor(self.sim)  # (num_actors, (pos + rot + linvel + angvel) = 13)
        self.root_states = gymtorch.wrap_tensor(_root_states).view(self.num_environments, 13)
        self.root_position = self.root_states[:, :3]
        self.root_rotation = self.root_states[:, 3:7]
        self.root_lin_vel = self.root_states[:, 7:10]
        self.root_ang_vel = self.root_states[:, 10:]

        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states).view(self.num_environments, -1, 2)
        self.dof_pos = self.dof_states[:, :, 0]
        self.dof_vel = self.dof_states[:, :, 1]
        self.num_dof = self.dof_states.shape[1]

        _dof_forces = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_forces = gymtorch.wrap_tensor(_dof_forces).view(self.num_environments, self.num_dof)

        _net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)  # (num_rb, 3)
        self.net_contact_forces = gymtorch.wrap_tensor(_net_contact_forces).view(self.num_environments, -1, 3)

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)  # (num_rb, (pos + rot + linvel + angvel) = 13)
        self.rb_states = gymtorch.wrap_tensor(_rb_states).view(self.num_environments, -1, 13)
        self.rb_pos = self.rb_states[:, :, :3]
        self.rb_rot = self.rb_states[:, :, 3:7]
        self.rb_lin_vel = self.rb_states[:, :, 7:10]
        self.rb_ang_vel = self.rb_states[:, :, 10:]
        self.num_rb = self.rb_states.shape[1]

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
    
    '''
    Getters for different robot properties
    '''
    def get_position(self):
        return self.root_position  # (num_environments, 3)

    def get_rotation(self):
        return self.root_rotation  # (num_environments, 4) - quaternion!

    def get_linear_velocity(self):
        return self.root_lin_vel  # (num_environments, 3)

    def get_angular_velocity(self):
        return self.root_ang_vel  # (num_environments, 3)

    def get_joint_position(self):
        return self.dof_pos  # (num_environments, num_dof)

    def get_joint_velocity(self):
        return self.dof_vel  # (num_environments, num_dof)

    def get_joint_torque(self):
        return self.dof_forces  # (num_environments, num_dof)

    def get_rb_position(self):
        return self.rb_pos  # (num_environments, num_rb, 3)

    def get_rb_rotation(self):
        return self.rb_rot  # (num_environments, num_rb, 4) - quaternion!

    def get_rb_linear_velocity(self):
        return self.rb_lin_vel  # (num_environments, num_rb, 3)

    def get_rb_angular_velocity(self):
        return self.rb_ang_vel  # (num_environments, num_rb, 3)

    def get_contact_states(self, collision_thresh=1):
        contact_forces = torch.norm(self.net_contact_forces, dim=2)
        collisions = contact_forces > collision_thresh
        return collisions # (num_environments, num_rb)

    def get_contact_forces(self):
        return self.net_contact_forces  # (num_environments, num_rb, 3)

    '''
    MDP support
    '''
    def step(self, actions = None):
        if actions is not None:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(actions))
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.command_dof_pos))


        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        self._refresh()
    
    def render(self):
        return self.gym.get_camera_image(self.sim, self.env_actor_handles[0][0], self.camera_handle, gymapi.IMAGE_COLOR).reshape(self.cam_height, self.cam_width, 4)
    
    def reset(self, env_index = None):
        if env_index is None:
            env_index = self.all_env_index
        else:
            env_index = self.all_env_index[env_index]

        random_pos = torch.rand(len(env_index), 3) * 2
        random_pos[:, 2] = 0.35

        # TODO: make faster for cuda?
        random_rot = torch.zeros(len(env_index), 3)
        random_rot[:, 0] = torch.deg2rad(torch.rand(len(env_index)) * 360)
        random_rot[:, 1] = 0 
        random_rot[:, 2] = np.radians(180)
        random_rot = R.matrix_to_quaternion(R.euler_angles_to_matrix(random_rot, convention = 'XYZ'))
         
        idx_tensor = env_index.long() # why can't I index with int32 tensors :(
        self.root_position[idx_tensor, :] = random_pos.to(self.device)
        self.root_lin_vel[idx_tensor, :] = 0
        self.root_ang_vel[idx_tensor, :] = 0
        self.root_rotation[idx_tensor, :] = random_rot.to(self.device)
        self.dof_pos[idx_tensor, :] = self.default_dof_pos
        self.dof_vel[idx_tensor, :] = 0
        
        indices = gymtorch.unwrap_tensor(env_index)
        self.gym.set_dof_state_tensor_indexed(self.sim,  gymtorch.unwrap_tensor(self.dof_states), indices, len(env_index))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,  gymtorch.unwrap_tensor(self.root_states), indices, len(env_index))
