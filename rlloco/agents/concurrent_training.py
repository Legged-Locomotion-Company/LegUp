from rlloco.isaac.environment import IsaacGymEnvironment

from typing import Union, List, Tuple

import gym
import torch
import numpy as np
import torchgeometry as tgm
from isaacgym.torch_utils import *
from stable_baselines3.common.vec_env import VecEnv

class HistoryBuffer:
    """Buffer to store most recently updated data that is updated every few seconds"""
    def __init__(self, num_envs: int, dt: float, update_freq: int, history_size: int, data_size: int, device: torch.device, fill: torch.Tensor = None):
        """
        Args:
            num_envs (int): number of environments to store data for
            dt (float): `dt` that the simulator is using, this essentially is the time (seconds) between each successive `step` call
            update_freq (int): how often (in seconds) to update the data in the buffer
            history_size (int): how much data to store in the buffer
            data_size (int): size of the data we are storing -- data must be 1-dimensional
            device (torch.device): device (cpu/cuda) that data should be on
            fill (torch.Tensor): values to fill the buffer with, defaults to None
        """
        self.device = device
        self.dt = dt
        self.update_freq = update_freq
        self.history_size = history_size
        self.num_envs = num_envs

        self.elapsed_time = torch.zeros(num_envs).to(self.device)
        self.data = torch.zeros(num_envs, data_size, history_size).to(self.device)

        if fill is not None:
            for i in range(history_size):
                self.data[:, :, i] = fill

    
    def step(self, new_data: torch.Tensor):
        """Updates the buffer if enough time has elapsed (specified by `dt`) from the previous call

        Args:
            new_data (torch.Tensor): Most recent data to be added if time has passed
        """
        self.elapsed_time += self.dt

        update_idx = self.elapsed_time >= self.update_freq
        self.elapsed_time[update_idx] = 0

        self.data[update_idx, :, 1:] = self.data[update_idx, :, :-1]
        self.data[update_idx, :, 0] = new_data[update_idx, :] # 0 is the newest
    
    def get(self, idx: Union[int, List[int]]) -> torch.Tensor:
        """Gets the data at the specified index

        Args:
            idx (Union[int, List[int]]): index of data, can be list or int

        Returns:
            torch.Tensor: data at that index, shape `(num_envs, data_size, :)`
        """
        return self.data[:, :, idx]
    
    def flatten(self) -> torch.Tensor:
        """Gets all the data in the buffer, flattened

        Returns:
            torch.Tensor: flattened data in buffer, shape `(num_envs, data_size * history_size)`
        """
        return self.data.view(self.num_envs, -1)

    def reset(self, update_idx: Union[int, List[int]]):
        """Zeros out all data in the buffer

        Args:
            update_idx (Union[int, List[int]]): indices of data to zero out
        """
        self.elapsed_time[update_idx] = 0
        self.data[update_idx, :, :] = 0

class G:
    """Random constants used in the paper"""
    action_scale = 0.5
    des_foot_height = 0.09

    k_v = 3.0
    k_w = 3.0
    k_a = 0.3
    k_slip = -0.08
    k_cl = -15.0
    k_ori = -3.0
    k_t = -6e-4
    k_q = -0.75
    k_q_d = -6e-4
    k_q_dd = -0.02
    k_s1 = -2.5
    k_s2 = -1.2
    k_base = -1.5

class ConcurrentTrainingEnv(VecEnv):
    """Implementation of an agent that learns a simple locomotion policy for the mini-cheetah. This entire implementation was based off of
    the following paper: https://arxiv.org/abs/2202.05481. It is hardcoded to run on GPU.
    """
    def __init__(self, num_environments: int, asset_path: str, asset_name: str):
        """
        Args:
            num_environments (int): number of parallel environments to create in simulator
            asset_path (str): path to URDF file from `asset_root`
            asset_root (str): root folder where the URDF to load is
        """
        self.device = torch.device("cuda")

        self.all_envs = list(range(num_environments))
        self.num_envs = num_environments
        self.default_dof_pos = torch.Tensor([0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6]).to(self.device)

        self.env = IsaacGymEnvironment(num_environments, True, asset_path, asset_name, self.default_dof_pos)
        self.feet_idx = [3, 6, 9, 12] # TODO: dynamically get this by rb name

        abduct = [1, 4, 7, 10]
        thigh = [2, 5, 8, 11]
        self.term_contacts = [0] + thigh # base + knee


        self.dt = 1. / 60. # TODO: get this dynamically
        self.max_ep_len = 1000 / self.dt
        self.cmd = torch.Tensor([1, 0, 0]).repeat(num_environments, 1).to(self.device) # desired [x_vel, y_vel, ang_vel]
        self.gravity_vec = torch.Tensor([0, 0, -1]).repeat(num_environments, 1).to(self.device)

        # tracking for agent
        self.takeoff_time = torch.zeros(num_environments, 4).to(self.device)
        self.touchdown_time = torch.zeros(num_environments, 4).to(self.device)
        self.ep_lens = torch.zeros(num_environments).to(self.device)
        self.prev_joint_pos = torch.zeros(num_environments, 12).to(self.device)
        self.des_joint_pos_hist = HistoryBuffer(num_environments, self.dt, 0.02, 2, 12, self.device, fill = self.default_dof_pos)
        self.joint_pos_err_hist = HistoryBuffer(num_environments, self.dt, 0.02, 3, 12, self.device)
        self.joint_vel_hist = HistoryBuffer(num_environments, self.dt, 0.02, 3, 12, self.device)

        self.prev_obs = HistoryBuffer(num_environments, self.dt, self.dt, 5, 153, self.device)

        # OpenAI Gym Environment required fields
        self.observation_space = gym.spaces.Box(low = np.ones(153) * -10000000, high = np.ones(153) * 10000000, dtype = np.float32)
        self.action_space = gym.spaces.Box(low = np.ones(12) * -10000000, high = np.ones(12) * 10000000, dtype = np.float32)
        self.metadata = {"render_modes": ['rgb_array']}
        self.reward_range = (-float("inf"), float("inf"))
        self.spec = None

        self.thresh = torch.zeros(12).to(self.device)
        self.thresh[[0, 3, 6, 9]] = torch.pi / 3 # abduct
        self.thresh[[1, 4, 7, 10]] = torch.pi / 3 # thigh
        self.thresh[[2, 5, 8, 11]] = 2 * torch.pi / 3 # knee

        # self.lower = (-self.thresh - self.default_dof_pos) / G.action_scale
        # self.upper = (self.thresh - self.default_dof_pos) / G.action_scale

    def sq_norm(self, tensor: torch.Tensor, dim: int = -1):
        """Helper function to compute norm(x)^2 without actually doing norm, thus avoiding the sqrt

        Args:
            tensor (torch.Tensor): input tensor to norm
            dim (int, optional): dimension to compute norm over. Defaults to -1.

        Returns:
            torch.Tensor: squared norm of given tensor
        """
        return torch.sum(torch.pow(tensor, 2), dim = dim)

    def make_reward(self, root_quat, root_ang_vel, joint_pos, joint_vel, des_jpos_t1, des_jpos_t2, command, root_lin_vel, foot_height, contact_probs, actions) -> torch.Tensor:
        """Computes the reward as per the paper https://arxiv.org/abs/2202.05481, refer to the paper for the formulas and descriptions of the rewards. 

        Args:
            root_quat (torch.Tensor): Robot rotation as a quaternion, shape `(num_envs, 4)`
            root_ang_vel (torch.Tensor): Robot angular velocity, shape `(num_envs, 3)`
            joint_pos (torch.Tensor): Robot joint position, shape `(num_envs, num_degrees_of_freedom)`
            joint_vel (torch.Tensor): Robot joint velocity, shape `(num_envs, num_degrees_of_freedom)`
            des_jpos_t1 (torch.Tensor): Desired joint position from timestep `t-1`, shape `(num_envs, num_degrees_of_freedom)`
            des_jpos_t2 (torch.Tensor): Desired joint position from timestep `t-2`, shape `(num_envs, num_degrees_of_freedom)`
            command (torch.Tensor): Robot target x/y position command and angular velocity command, shape `(num_envs, 3)`
            root_lin_vel (torch.Tensor): robot root linear velocity, shape `(num_envs, 3)`
            foot_height (torch.Tensor): robot foot height, shape `(num_envs, 4)`. Currently does not factor in uneven terrain and is relative to the robot base
            contact_probs (torch.Tensor): probability that a given foot is on the ground, shape `(num_envs, 4)`
            actions (torch.Tensor): action that was commanded previously, shape `(num_envs, num_degrees_of_freedom)`

        Returns:
            torch.Tensor: final computed reward for each environment, shape `(num_envs)`
        """
        r_v = G.k_v * torch.exp(-self.sq_norm(command[:, :2] - root_lin_vel[:, :2]))

        r_w = G.k_w * torch.exp(-1.5 * torch.square(command[:, 2] - root_ang_vel[:, 2]))

        tmax = torch.maximum(self.touchdown_time, self.takeoff_time)
        r_air = G.k_a * torch.clamp(tmax, max = 0.2) * (tmax < 0.25)

        feet_vel = self.env.get_rb_linear_velocity()[:, self.feet_idx, :][:, :, [0, 1]]
        r_slip = G.k_slip * contact_probs * self.sq_norm(feet_vel)

        sqrt_vel = torch.sqrt(torch.linalg.norm(feet_vel, dim = -1))
        r_cl = G.k_cl * torch.pow(foot_height - G.des_foot_height, 2) * sqrt_vel

        # just wrote this, im not really sure if it works
        # yaw = tgm.quaternion_to_angle_axis(root_quat)[:, 2]
        # yaw_diff = torch.where(yaw > torch.pi, 2 * torch.pi - yaw, yaw)
        # r_ori = G.k_ori * torch.pow(yaw_diff, 2)
        
        # taken from NVIDIA-Omniverse example
        # ori = quat_rotate_inverse(root_quat, self.gravity_vec)
        # r_ori = G.k_ori * self.sq_norm(ori[:, :2])

        # not really sure if this works either :(
        angle = tgm.quaternion_to_angle_axis(root_quat)
        rot_error = np.pi - angle[:, 2]
        r_ori = G.k_ori * torch.abs(rot_error)

        r_t = G.k_t * self.sq_norm(self.env.get_joint_torque())

        r_q = G.k_q * self.sq_norm(joint_pos - self.default_dof_pos)

        r_q_d = G.k_q_d * self.sq_norm(joint_vel)

        r_q_dd = G.k_q_dd * self.sq_norm(joint_vel - self.joint_vel_hist.get(0))

        r_s1 = G.k_s1 * self.sq_norm(actions - des_jpos_t1)

        r_s2 = G.k_s2 * self.sq_norm(actions - 2 * des_jpos_t1 + des_jpos_t2)

        r_base = G.k_base * (0.8 * torch.pow(root_lin_vel[:, 2], 2) + 0.2 * torch.abs(root_ang_vel[:, 0]) + 0.2 * torch.abs(root_ang_vel[:, 1]))

        # zeroing out some rewards I think aren't working properly
        # r_w[:] = 0
        # r_slip[:] = 0
        # r_s1[:] = 0
        # r_s2[:] = 0
        # r_q_dd[:] = 0
    
        r_pos = r_v + r_w + torch.sum(r_air, dim = -1) # in the paper, they only sum 3 feet instead of 4?
        r_neg = r_ori + r_t + r_q + r_q_d + r_q_dd + r_s1 + r_s2 + r_base + torch.sum(r_slip + r_cl, dim = -1) # in the paper, they only sum 3 feet instead of 4?
        r_tot = r_pos + torch.exp(0.2 * r_neg)

        terms = torch.mean(torch.stack([r_tot, r_pos, r_neg, r_v, r_w, torch.sum(r_air, dim = -1), torch.sum(r_slip, dim = -1), torch.sum(r_cl, dim = -1), r_t, r_q, r_q_d, r_q_dd, r_s1, r_s2, r_base, r_ori]), dim = 1)
        names = ['total', 'pos', 'neg', 'r_v', 'r_w', 'r_air', 'r_slip', 'r_cl', 'r_t', 'r_q', 'r_qdot', 'r_qddot', 'r_s1', 'r_s2', 'r_base', 'r_ori']

        return r_tot, terms, names

    def explain_obs(self, idx, obs):
        root_quat = obs[idx, 0:4]
        root_ang_vel = obs[idx, 4:7]
        joint_pos = obs[idx, 7:19]
        joint_vel = obs[idx, 19:31]
        des_jpos_t1 = obs[idx, 31:43]
        des_jpos_t2 = obs[idx, 43:55]
        Q_err_hist = obs[idx, 55:91]
        Q_vel_hist = obs[idx, 91:127]
        rel_feet_pos = obs[idx, 127:139]
        cmd = obs[idx, 139:142]
        root_lin_vel = obs[idx, 142:145]
        feet_height = obs[idx, 145:149]
        contact_probs = obs[idx, 149:153]

        print(f'NaN found at {idx}, timestep {self.ep_lens[idx]}')
        # print(f'root_quat: {root_quat}')
        # print(f'root_ang_vel: {root_ang_vel}')
        print(f'joint_pos: {joint_pos}')
        # print(f'joint_vel: {joint_vel}')
        print(f'des_jpos_t1: {des_jpos_t1}')
        print(f'des_jpos_t2: {des_jpos_t2}')
        # print(f'Q_err_hist: {Q_err_hist}')
        # print(f'Q_vel_hist: {Q_vel_hist}')
        print(f'rel_feet_pos: {rel_feet_pos}')
        print(f'root_lin_vel: {root_lin_vel}')
        print(f'real contact states: {self.env.get_contact_states()[idx]}')
        # print(f'cmd: {cmd}')
        # print(f'feet_height: {feet_height}')
        # print(f'contact_probs: {contact_probs}')
        print()



    def make_observation_and_reward(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Makes observation from simulation environment and computes the reward from the observations and action

        Args:
            actions (torch.Tensor): Previously commanded target joint position, shape `(num_envs, num_degrees_of_freedom)`

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Newest observation of shape `(num_envs, observation_space)` and respective reward of shape `(num_envs)`
        """        
        root_quat = self.env.get_rotation() # (num_envs, 4)
        root_ang_vel = self.env.get_angular_velocity() # (num_envs, 3)
        joint_pos = self.env.get_joint_position() # (num_envs, 12)
        joint_vel = self.env.get_joint_velocity() # (num_envs, 12)
        des_jpos_t1 = self.des_joint_pos_hist.get(0) # (num_envs, 12)
        des_jpos_t2 = self.des_joint_pos_hist.get(1) # (num_envs, 12)
        Q_err_hist = self.joint_pos_err_hist.flatten() # (num_envs, 36)
        Q_vel_hist = self.joint_vel_hist.flatten() # (num_envs, 36)
        
        feet_pos = self.env.get_rb_position()[:, self.feet_idx, :]
        body_pos = self.env.get_rb_position()[:, [0], :]
        relative_feet_pos = feet_pos - body_pos
        relative_feet_pos = relative_feet_pos.view(-1, 12) # (num_envs, 12)
        command = self.cmd # (num_envs, 3)

        # this stuff should be estimated but we're starting with perfect info
        root_lin_vel = self.env.get_linear_velocity() # (num_envs, 3)
        foot_height = feet_pos[:, :, 2] # (num_envs, 4), assuming flat ground
        contact_probs = self.env.get_contact_states()[:, self.feet_idx] # (num_envs, 4)

        reward, terms, names = None, None, None
        if actions is not None:
            reward, terms, names = self.make_reward(root_quat, root_ang_vel, joint_pos, joint_vel, des_jpos_t1, des_jpos_t2, command, root_lin_vel, foot_height, contact_probs, actions)

        obs_list = [root_quat, root_ang_vel, joint_pos, joint_vel, des_jpos_t1, des_jpos_t2, Q_err_hist, Q_vel_hist, relative_feet_pos, command, root_lin_vel, foot_height, contact_probs]
        obs = torch.cat(obs_list, dim = 1) # (num_envs, 153)

        if torch.any(torch.isnan(obs)):
            where_nan = torch.unique(torch.argwhere(torch.isnan(obs))[:, 0])

            for i in range(5):
                print(f'Prev obs {i}')
                self.explain_obs(where_nan, self.prev_obs.get(i))

            print(f'Current obs:')
            self.explain_obs(where_nan, obs)
    
        return obs, reward, terms, names

    def check_termination(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Checks if any environments have terminated (because of collisions) or truncated (ran out of time)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Both tensors are truthy tensors of shape `(num_envs)`, they represent which environments have
            terminated or truncated (respectively)
        """
        term = torch.any(self.env.get_contact_states()[:, self.term_contacts], dim = -1)
        trunc = self.ep_lens > self.max_ep_len

        return term, trunc

    def update_feet_states(self):
        """Called at every simulation step, updates the tracking buffers for feet states"""
        feet_contacts = self.env.get_contact_states()[:, self.feet_idx]
        self.takeoff_time[feet_contacts] += self.dt
        self.touchdown_time[feet_contacts] = 0

        self.takeoff_time[~feet_contacts] = 0
        self.touchdown_time[~feet_contacts] += self.dt

    def reset_partial(self, env_idx: Union[torch.Tensor, List[int], int]) -> torch.Tensor:
        """Resets a subset of all environments, specified by `env_idx`

        Args:
            env_idx (Union[torch.Tensor, List[int], int]): Indices of environments to reset

        Returns:
            torch.Tensor: New observations from the environments that were reset
        """
        self.takeoff_time[env_idx] = 0
        self.touchdown_time[env_idx] = 0
        self.ep_lens[env_idx] = 0
        self.prev_joint_pos[:] = self.default_dof_pos

        self.des_joint_pos_hist.reset(env_idx)
        self.joint_pos_err_hist.reset(env_idx)
        self.joint_vel_hist.reset(env_idx)
        self.prev_obs.reset(env_idx)
        
        self.env.reset(env_idx)

        obs, rew, terms, names = self.make_observation_and_reward(None)
        return obs[env_idx]

    def reset(self) -> torch.Tensor:
        """Resets all environments. From what I understand, this is really only called once (at the beginning)

        Returns:
            torch.Tensor: New observations from all environments
        """
        return self.reset_partial(self.all_envs)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Does a single step of the simulation environment based on a given command, and computes new state information and rewards

        Args:
            actions (torch.Tensor): Commanded joint position, shape `(num_envs, num_degrees_of_freedom)`

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]: new observation, corresponding reward, truthy tensor of which 
            environments have terminated/truncated, and additional metadata (currently empty). Observation tensor has shape `(num_envs, observation_space)`
            and all other tensors have shape `(num_envs)`
        """

        self.prev_joint_pos[:] = self.env.get_joint_position()

        # actions < (thresh - self.default_dof_pos) / G.action_scale
        # actions > (-thresh - self.default_dof_pos) / G.action_scale

        actions = actions * G.action_scale + self.default_dof_pos
        actions = torch.clamp(actions, min = -self.thresh, max = self.thresh)
        self.env.step(actions)

        self.env._refresh()
        self.update_feet_states()

        new_obs, reward, terms, names = self.make_observation_and_reward(actions)

        new_joint_pos = self.env.get_joint_position()
        joint_vel = (new_joint_pos - self.prev_joint_pos) / self.dt
        self.des_joint_pos_hist.step(actions)
        self.joint_pos_err_hist.step(actions - new_joint_pos)
        self.joint_vel_hist.step(joint_vel)
                
        self.ep_lens += 1
        term, trunc = self.check_termination()
        
        reset_idx = set()
        for term_idx in torch.where(trunc)[0]:
            reset_idx.add(term_idx.item())

        for term_idx in torch.where(term)[0]:
            reward[term_idx] = -10
            reset_idx.add(term_idx.item())

        reset_idx = list(reset_idx)

        if len(reset_idx) > 0:
            new_obs[reset_idx, :] = self.reset_partial(reset_idx)

        self.prev_obs.step(new_obs)

        infos = [{} for _ in range(self.num_envs)]
        infos[0]['terms'] = terms
        infos[0]['names'] = names

        return new_obs, reward, torch.logical_or(term, trunc), infos       

    def render(self) -> torch.Tensor:
        """Gets a screenshot from simulation environment as torch tensor

        Returns:
            torch.Tensor: RGBA screenshot from sim, shape `(height, width, 4)`
        """
        return self.env.render()

    def env_is_wrapped(self, wrapper_class, indices = None):
        """Added because it is required as per the OpenAI gym specification"""
        return [False]

    def close(self):
        """Added because it is required as per the OpenAI gym specification"""
        raise NotImplementedError

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Added because it is required as per the OpenAI gym specification"""
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        """Added because it is required as per the OpenAI gym specification"""
        raise NotImplementedError

    def seed(self, seed = None):
        """Added because it is required as per the OpenAI gym specification"""
        raise NotImplementedError

    def set_attr(self, attr_name, value, indices=None):
        """Added because it is required as per the OpenAI gym specification"""
        raise NotImplementedError

    def step_async(self, actions):
        """Added because it is required as per the OpenAI gym specification"""
        raise NotImplementedError

    def step_wait(self):
        """Added because it is required as per the OpenAI gym specification"""
        raise NotImplementedError