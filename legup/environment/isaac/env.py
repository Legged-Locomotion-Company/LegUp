from isaacgym import gymapi, gymtorch

import torch
from typing import List, Optional

from legup.common.abstract_env import AbstractEnv
from legup.environment.isaac.factory import IsaacGymFactory
from legup.environment.isaac.dynamics import IsaacGymDynamics


class IsaacGymEnvironment(AbstractEnv):
    def __init__(self):
        self.gym = gymapi.acquire_gym()
        self.sim = IsaacGymFactory.create_sim()
        self.assets = IsaacGymFactory.create_assets()
        self.envs, self.actors = IsaacGymFactory.create_actors()

        self.all_env_index = torch.arange(num_environments).to(torch.long).to(self.device)

        self.gym.prepare_sim(self.sim)
        self.dyn = IsaacGymDynamics()

    def step(self, actions: Optional[torch.Tensor] = None):
        """Moves robots using `actions`, steps the simulation forward, updates graphics, and refreshes state tensors
        Args:
            actions (torch.Tensor, optional): target joint positions to command each robot, shape `(num_environments, num_degrees_of_freedom)`. 
                If none, robots are commanded to the default joint position provided earlier Defaults to None.
        """

        actions = gymtorch.unwrap_tensor(actions if actions is not None else self.command_dof_pos)    

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
        pass # TODO
        # return self.gym.get_camera_image(self.sim, self.env_actor_handles[self.camera_env][0], self.camera_handle, gymapi.IMAGE_COLOR).reshape(self.cam_height, self.cam_width, 4)

    def reset(self, env_index: Optional[List[int]] = None):
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
        random_rot[:] = agent.sample_new_quat(env_index) 

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