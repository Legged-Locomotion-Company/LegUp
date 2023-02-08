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
        self.camera_handle = IsaacGymFactory.create_camera()

        self.all_env_index = torch.arange(num_environments, dtype=torch.long, device=self.device) # TODO: this should be num_agents, not num_envs
        self.terminated_agents = torch.ones(num_environments, dtype=torch.bool, device=self.device) # TODO: this should be num_agents, not num_envs
        self.dones = torch.zeros(num_environments, dtype=torch.bool, device=self.device) # TODO: this should be num_agents, not num_envs

        self.gym.prepare_sim(self.sim)
        self.dyn = IsaacGymDynamics(self.sim, self.gym, num_agents)

    def step(self, actions: Optional[torch.Tensor] = None):
        """Moves robots using `actions`, steps the simulation forward, updates graphics, and refreshes state tensors
        Args:
            actions (torch.Tensor, optional): target joint positions to command each robot, shape `(num_environments, num_degrees_of_freedom)`. 
                If none, robots are commanded to the default joint position provided earlier Defaults to None.
        """
        
        actions = self.agent.make_actions(actions)
        actions = gymtorch.unwrap_tensor(actions if actions is not None else self.command_dof_pos)    

        self.gym.set_dof_position_target_tensor(self.sim, actions)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # TODO: reset any previously terminated environments
        self.reset_terminated_agents()
        self.agent.update_commands(self.dones)

        self.dyn.apply_tensor_changes()
        self.dyn.refresh_buffers()

        # TODO: post physics step
        self.agent.post_physics_step()

        observation = self.agent.make_observation()
        reward = self.agent.make_reward()

        term_idxs = self.agent.find_terminated()
        self.terminated_agents[term_idxs] = True

        infos = {}
        return observation, reward, self.dones, infos

    def reset_terminated_agents(self):
        update_idx = self.all_env_index[self.terminated_agents]
        self.dyn.get_position()[update_idx, :] = self.agent.sample_new_position(update_idx)
        self.dyn.get_linear_velocity()[update_idx, :] = 0
        self.dyn.get_angular_velocity()[update_idx, :] = 0
        self.dyn.get_rotation()[update_idx, :] = self.agent.sample_new_quaternion(update_idx)
        self.dyn.get_joint_position()[update_idx, :] = self.default_dof_pos
        self.dyn.get_joint_velocity()[update_idx, :] = 0

        self.dones[:] = self.terminated_agents
        self.terminated_agents[:] = False
        
    def render(self) -> torch.Tensor:
        """Gets an image of the environment from the camera and returns it
        Returns:
            np.ndarray: RGB image, shape `(camera_height, camera_width, 4)`
        """
        pass # TODO
        # return self.gym.get_camera_image(self.sim, self.env_actor_handles[self.camera_env][0], self.camera_handle, gymapi.IMAGE_COLOR).reshape(self.cam_height, self.cam_width, 4)