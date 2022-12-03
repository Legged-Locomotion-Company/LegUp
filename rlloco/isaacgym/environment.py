from isaacgym import gymapi, gymtorch
from scipy.spatial.transform import Rotation as R


from rlloco.isaacgym.simulator import SimulatorContext
from rlloco.isaacgym.debugvis import DebugInfo
import torch

# from simulator import SimulatorContext


class IsaacGymEnvironment:
    def __init__(self, num_environments, dt, asset_root, asset_path):

        # TODO: clean up parameters into objects (sim info, asset info), terrain config (flat/rough)
        self.ctx = SimulatorContext(
            num_environments, dt, asset_root, asset_path)
        self.num_environments = num_environments
        self.gym = self.ctx.gym
        self.sim = self.ctx.sim
        self.env_handles = self.ctx.env_handles
        self.actor_handles = self.ctx.actor_handles

        self.rb_names, self.dof_names = self._get_asset_info(
            self.ctx.asset_handle)
        self.num_rb = len(self.rb_names)
        self.num_dof = len(self.dof_names)

        # TODO: parameterize this
        self.default_dof_pos = torch.Tensor([0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6]).cuda()

        self.viewer = self.gym.create_viewer(
            self.sim, gymapi.CameraProperties())
        self.should_render = False
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_V, "toggle_render")

        self.dbg = DebugInfo(self.ctx)
        self.dbg.print_all_info()

        self._acquire_state_tensors()
        self._refresh()

    def _acquire_state_tensors(self):
        _root_states = self.gym.acquire_actor_root_state_tensor(
            self.sim)  # (num_actors, (pos + rot + linvel + angvel) = 13)
        self.root_states = gymtorch.wrap_tensor(
            _root_states).view(self.num_environments, 13)
        self.root_position = self.root_states[:, :3]
        self.root_rotation = self.root_states[:, 3:7]
        self.root_lin_vel = self.root_states[:, 7:10]
        self.root_ang_vel = self.root_states[:, 10:]

        _dof_forces = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_forces = gymtorch.wrap_tensor(_dof_forces).view(
            self.num_environments, self.num_dof)

        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states).view(
            self.num_environments, self.num_dof, 2)
        self.dof_pos = self.dof_states[:, :, 0]
        self.dof_vel = self.dof_states[:, :, 1]
        self.num_dof = self.dof_states.shape[1]

        _net_contact_forces = self.gym.acquire_net_contact_force_tensor(
            self.sim)  # (num_rb, 3)
        self.net_contact_forces = gymtorch.wrap_tensor(
            _net_contact_forces).view(self.num_environments, self.num_rb, 3)

        _rb_states = self.gym.acquire_rigid_body_state_tensor(
            self.sim)  # (num_rb, (pos + rot + linvel + angvel) = 13)
        self.rb_states = gymtorch.wrap_tensor(_rb_states).view(
            self.num_environments, self.num_rb, 13)
        self.rb_pos = self.rb_states[:, :, :3]
        self.rb_rot = self.rb_states[:, :, 3:7]
        self.rb_lin_vel = self.rb_states[:, :, 7:10]
        self.rb_ang_vel = self.rb_states[:, :, 10:]
        self.num_rb = self.rb_states.shape[1]

    def _get_asset_info(self, asset_handle):
        num_rb = self.gym.get_asset_rigid_body_count(asset_handle)
        rb_names = [self.gym.get_asset_rigid_body_name(
            asset_handle, i) for i in range(num_rb)]

        num_dofs = self.gym.get_asset_dof_count(asset_handle)
        dof_names = [self.gym.get_asset_dof_name(
            asset_handle, i) for i in range(num_dofs)]

        return rb_names, dof_names

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
        return collisions  # (num_environments, num_rb)

    def get_contact_forces(self):
        return self.net_contact_forces  # (num_environments, num_rb, 3)

    '''
    MDP support
    '''
    def step(self, actions = None):
        if actions is not None:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(actions))

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self._refresh()

    def render(self):
        for event in self.gym.query_viewer_action_events(self.viewer):
            if event.action == 'toggle_render' and event.value > 0:
                self.should_render = not self.should_render

        if self.should_render:
            self.gym.step_graphics(self.sim)

            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.clear_lines(self.viewer)
        else:
            self.gym.poll_viewer_events(self.viewer)

    def reset(self, env_index=None):
        if env_index is None:
            env_index = list(range(self.num_environments))
            num_times = self.num_environments
        else:
            num_times = len(env_index)

        random_pos = self.ctx.sample_data(
            (0, 0, 0.35), (2, 2, 2), num_times).squeeze().float().cuda()
        random_rot = torch.from_numpy(R.from_euler('xyz', self.ctx.sample_data(
            (0, 0, -1), (360, 360, 360), num_times)).as_quat()).squeeze().float().cuda()

        self.root_position[env_index, :] = random_pos
        self.root_lin_vel[env_index, :] = 0
        self.root_ang_vel[env_index, :] = 0
        self.root_rotation[env_index, :] = random_rot
        self.dof_pos[env_index, :] = self.default_dof_pos
        self.dof_vel[env_index, :] = 0

        env_index = torch.Tensor(env_index).to(torch.int).cuda()
        
        indices = gymtorch.unwrap_tensor(env_index)

        self.gym.set_dof_state_tensor_indexed(
            self.sim,  gymtorch.unwrap_tensor(self.dof_states), indices, num_times)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,  gymtorch.unwrap_tensor(self.root_states), indices, num_times)

    def release(self):
        self.gym.destroy_sim(self.sim)
        self.gym.destroy_viewer(self.viewer)