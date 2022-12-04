from rlloco.isaac.environment import IsaacGymEnvironment

import gym
import torch
import numpy as np
import torchgeometry as tgm
from stable_baselines3.common.vec_env import VecEnv


class HistoryBuffer:
    def __init__(self, num_envs, dt, update_freq, history_size, data_size, device):
        self.device = device
        self.dt = dt
        self.update_freq = update_freq
        self.history_size = history_size
        self.num_envs = num_envs

        self.elapsed_time = torch.zeros(num_envs).to(self.device)
        self.data = torch.zeros(num_envs, data_size, history_size).to(self.device)
    
    def step(self, new_data):
        self.elapsed_time += self.dt

        update_idx = self.elapsed_time >= self.update_freq
        self.elapsed_time[update_idx] = 0

        self.data[update_idx, :, :self.history_size - 1] = self.data[update_idx, :, 1:]
        self.data[update_idx, :, 0] = new_data[update_idx, :] # 0 is the newest
    
    def get(self, idx):
        return self.data[:, :, idx]
    
    def flatten(self):
        return self.data.view(self.num_envs, -1)

    def reset(self, update_idx):
        self.elapsed_time[update_idx] = 0
        self.data[update_idx, :, :] = 0

class G:
    action_scale = 0.1
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
    def __init__(self, num_environments, asset_path, asset_name):
        self.device = torch.device("cuda")

        self.all_envs = list(range(num_environments))
        self.num_envs = num_environments
        self.default_dof_pos = torch.Tensor([0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6]).to(self.device)

        self.env = IsaacGymEnvironment(num_environments, True, asset_path, asset_name, self.default_dof_pos)
        self.feet_idx = [3, 6, 9, 12] # TODO: dynamically get this by rb name
        self.term_contacts = [0, 1, 4, 7, 10] # body, shoulder
        self.dt = 1. / 60. # TODO: get this dynamically
        self.max_ep_len = 1000 / self.dt
        self.cmd = torch.Tensor([3.5, 0, 0]).repeat(num_environments, 1).to(self.device) # desired [x_vel, y_vel, ang_vel]

        # tracking for agent
        self.takeoff_time = torch.zeros(num_environments, 4).to(self.device)
        self.touchdown_time = torch.zeros(num_environments, 4).to(self.device)
        self.ep_lens = torch.zeros(num_environments).to(self.device)
        self.prev_joint_pos = torch.zeros(num_environments, 12).to(self.device)
        self.des_joint_pos_hist = HistoryBuffer(num_environments, self.dt, 0.02, 2, 12, self.device)
        self.joint_pos_err_hist = HistoryBuffer(num_environments, self.dt, 0.02, 3, 12, self.device)
        self.joint_vel_hist = HistoryBuffer(num_environments, self.dt, 0.02, 3, 12, self.device)

        # OpenAI Gym Environment required fields
        #self.observation_space = gym.spaces.Box(low = np.ones(153) * -np.Inf, high = np.ones(153) * np.Inf, dtype = np.float32)
        #self.action_space = gym.spaces.Box(low = np.ones(12) * -np.Inf, high = np.ones(12) * np.Inf, dtype = np.float32)
        self.observation_space = gym.spaces.Box(low = np.ones(153) * -10000000, high = np.ones(153) * 10000000, dtype = np.float32)
        self.action_space = gym.spaces.Box(low = np.ones(12) * -10000000, high = np.ones(12) * 10000000, dtype = np.float32)
        self.metadata = {"render_modes": ['rgb_array']}
        self.reward_range = (-float("inf"), float("inf"))
        self.spec = None

    
    def sq_norm(self, tensor, dim = -1):
        return torch.sum(torch.pow(tensor, 2), dim = dim)

    def make_observation_and_reward(self, actions):
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

        reward = None
        if actions is not None:
            reward = self.make_reward(root_quat, root_ang_vel, joint_pos, joint_vel, des_jpos_t1, des_jpos_t2, Q_err_hist, Q_vel_hist, relative_feet_pos, command, root_lin_vel, foot_height, contact_probs, actions)

        obs_list = [root_quat, root_ang_vel, joint_pos, joint_vel, des_jpos_t1, des_jpos_t2, Q_err_hist, Q_vel_hist, relative_feet_pos, command, root_lin_vel, foot_height, contact_probs]
        obs = torch.cat(obs_list, dim = 1) # (num_envs, 153)

        return obs, reward
        
    def make_reward(self, root_quat, root_ang_vel, joint_pos, joint_vel, des_jpos_t1, des_jpos_t2, Q_err_hist, Q_vel_hist, relative_feet_pos, command, root_lin_vel, foot_height, contact_probs, actions):
        r_v = G.k_v * torch.exp(-self.sq_norm(command[:, :2] - root_lin_vel[:, :2]))

        r_w = G.k_w * torch.exp(-1.5 * self.sq_norm(command[:, [2]] - root_lin_vel[:, [2]]))

        tmax = torch.maximum(self.touchdown_time, self.takeoff_time)
        r_air = G.k_a * torch.clamp(tmax, max = 0.2) * (tmax < 0.25)

        feet_vel = self.env.get_rb_linear_velocity()[:, self.feet_idx, :][:, :, [0, 1]]
        r_slip = G.k_slip * contact_probs * self.sq_norm(feet_vel)

        sqrt_vel = torch.sqrt(torch.linalg.norm(feet_vel, dim = -1))
        r_cl = G.k_cl * torch.pow(foot_height - G.des_foot_height, 2) * sqrt_vel

        yaw = tgm.quaternion_to_angle_axis(root_quat)[:, 2]
        yaw_diff = torch.where(yaw > torch.pi, 2 * torch.pi - yaw, yaw)
        r_ori = G.k_ori * torch.pow(yaw_diff, 2) # maybe wrong, not sure what it should be

        r_t = G.k_t * self.sq_norm(self.env.get_joint_torque())

        r_q = G.k_q * self.sq_norm(joint_pos - self.default_dof_pos)

        r_q_d = G.k_q_d * self.sq_norm(joint_vel)

        r_q_dd = G.k_q_dd * self.sq_norm(joint_vel - self.joint_vel_hist.get(0))

        r_s1 = G.k_s1 * self.sq_norm(actions - self.des_joint_pos_hist.get(0))

        r_s2 = G.k_s2 * self.sq_norm(actions - 2 * des_jpos_t1 + des_jpos_t2)

        r_base = G.k_base * (0.8 * torch.pow(root_lin_vel[:, 2], 2) + 0.2 * torch.abs(root_ang_vel[:, 0]) + 0.2 * torch.abs(root_ang_vel[:, 1]))

        r_pos = r_v + r_w + torch.sum(r_air, dim = -1) # in the paper, they only sum 3 feet instead of 4?
        r_neg = r_ori + r_t + r_q + r_q_d + r_q_dd + r_s1 + r_s2 + r_base + torch.sum(r_slip + r_cl, dim = -1) # in the paper, they only sum 3 feet instead of 4?
        r_tot = r_pos + torch.exp(0.2 * r_neg)

        return r_tot      

    def check_termination(self):
        term = torch.any(self.env.get_contact_states()[:, self.term_contacts], dim = -1)
        trunc = self.ep_lens > self.max_ep_len

        return term, trunc

    def update_feet_states(self):
        feet_contacts = self.env.get_contact_states()[:, self.feet_idx]
        self.takeoff_time[feet_contacts] += self.dt
        self.touchdown_time[feet_contacts] = 0

        self.takeoff_time[~feet_contacts] = 0
        self.touchdown_time[~feet_contacts] += self.dt

    def reset_partial(self, env_idx):
        self.takeoff_time[env_idx] = 0
        self.touchdown_time[env_idx] = 0
        self.ep_lens[env_idx] = 0
        self.prev_joint_pos[:] = self.default_dof_pos

        self.des_joint_pos_hist.reset(env_idx)
        self.joint_pos_err_hist.reset(env_idx)
        self.joint_vel_hist.reset(env_idx)
        
        self.env.reset(env_idx)

        obs, _ = self.make_observation_and_reward(None)
        return obs[env_idx]

    def reset(self):
        return self.reset_partial(self.all_envs)

    def step(self, actions):
        self.prev_joint_pos[:] = self.env.get_joint_position()

        actions = actions * G.action_scale + self.default_dof_pos
        self.env.step(actions)
        self.update_feet_states()

        new_obs, reward = self.make_observation_and_reward(actions)

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

        infos = [{} for _ in range(self.num_envs)]
        return new_obs, reward, torch.logical_or(term, trunc), infos       

    def render(self):
        return self.env.render()

    def env_is_wrapped(self, wrapper_class, indices = None):
        return [False]

    def close(self):
        raise NotImplementedError

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError

    def seed(self, seed = None):
        raise NotImplementedError

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def step_async(self, actions):
        raise NotImplementedError

    def step_wait(self):
        raise NotImplementedError