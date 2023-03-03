from isaacgym import gymapi, gymtorch

import torch
import numpy as np

from legup.common.abstract_env import AbstractEnv, StepResult
from legup.environment.isaac.factory import IsaacGymFactory
from legup.environment.isaac.dynamics import IsaacGymDynamics
from legup.common.legup_config import IsaacConfig, AgentConfig
from legup.agents.wild_anymal_agent.wild_anymal_agent import WildAnymalAgent


class IsaacGymEnvironment(AbstractEnv):
    def __init__(self, env_config: IsaacConfig, agent_config: AgentConfig, device: torch.device):
        num_agents = env_config.num_agents_per_env * \
            env_config.num_envs_per_terrain_type * env_config.num_terrain
        self.agent = WildAnymalAgent(
            agent_config, None, num_agents, env_config.sim_config.dt, device)
        self.config = env_config

        self.all_agent_index = torch.arange(
            num_agents, dtype=torch.long, device=device)
        self.terminated_agents = torch.ones(
            num_agents, dtype=torch.bool, device=device)
        self.dones = torch.zeros(num_agents, dtype=torch.bool, device=device)

        self.gym = gymapi.acquire_gym()  # type: ignore
        self.sim = IsaacGymFactory.create_sim(self.gym, env_config)
        self.heightfield = IsaacGymFactory.create_terrain(
            self.sim, self.gym, self.agent, env_config)
        self.envs, self.actors, self.asset = IsaacGymFactory.create_actors(
            self.sim, self.gym, self.agent, env_config)
        self.camera_handle = IsaacGymFactory.create_camera(
            self.sim, self.gym, env_config)
        self.gym.prepare_sim(self.sim)

        self.dyn = IsaacGymDynamics(self.sim, self.gym, num_agents)

    def step(self, actions: torch.Tensor) -> StepResult:
        """Moves robots using `actions`, steps the simulation forward, updates graphics, and refreshes state tensors

        Args:
            actions (torch.Tensor): raw network outputs to convert into DOF pos and send through environment

        Returns:
            StepResult: new observation, corresponding rewards, and which environments have terminated
        """

        # compute actions and send to environment
        actions = self.agent.make_actions(actions)
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(actions))

        # step in simulation environment
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # reset any agents that have terminated
        self.reset_terminated_agents()
        self.agent.reset_agents(self.dones)

        self.dyn.apply_tensor_changes()
        self.dyn.refresh_buffers()

        self.agent.post_physics_step()

        observation = self.agent.make_observation(self.dyn)
        rewards, reward_dict = self.agent.make_reward(self.dyn)

        term_idxs = self.agent.find_terminated(self.dyn)
        self.terminated_agents[term_idxs] = True

        return StepResult(observation, rewards, self.dones, reward_dict)

    def reset_terminated_agents(self):
        update_idx = self.all_agent_index[self.terminated_agents]
        num_updates = len(update_idx)

        self.dyn.get_position()[update_idx, :] = \
            self.agent.sample_new_position(num_updates)  # type: ignore
        self.dyn.get_linear_velocity()[update_idx, :] = 0
        self.dyn.get_angular_velocity()[update_idx, :] = 0
        self.dyn.get_rotation()[update_idx, :] = self.agent.sample_new_quaternion(
            num_updates)
        self.dyn.get_joint_position(
        )[update_idx, :] = self.agent.sample_new_joint_pos(num_updates)
        self.dyn.get_joint_velocity()[update_idx, :] = 0

        self.dones[:] = self.terminated_agents
        self.terminated_agents[:] = False

    def render(self) -> np.ndarray:
        """Gets an image of the environment from the camera and returns it
        Returns:
            np.ndarray: RGB image, shape `(camera_height, camera_width, 4)`
        """

        cam_config = self.config.camera_config
        render_target = self.envs[cam_config.render_target]
        captured_image = self.gym.get_camera_image(
            self.sim, render_target, self.camera_handle, gymapi.IMAGE_COLOR)
        captured_image = captured_image.reshape(
            cam_config.capture_height, cam_config.capture_width, 4)

        # TODO: draw dyn information onto image
        return captured_image
