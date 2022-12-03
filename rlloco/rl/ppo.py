import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import cv2
import numpy as np

def orthogonal_weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
        m.bias.data.fill_(0.01)

'''
class Actor(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        self.action_space = action_space
        self.mlp = nn.Sequential(
                nn.Linear(observation_space, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, action_space * 2),
        )

        self.mlp.apply(orthogonal_weight_init)
    
    def forward(self, obs):
        probs = self.mlp(obs)
        mu, sigma = probs[..., :self.action_space], probs[..., self.action_space:]

        sigma = torch.exp(sigma)
        dist = distributions.Normal(mu, sigma)

        actions = dist.sample()
        return actions, dist.log_prob(actions)
    
    def evaluate(self, obs, actions):
        probs = self.mlp(obs)
        mu, sigma = probs[..., :self.action_space], probs[..., self.action_space:]

        sigma = torch.exp(sigma)
        dist = distributions.Normal(mu, sigma)

        return dist.log_prob(actions), dist.entropy()
'''

class Actor(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        self.action_space = action_space
        self.mlp = nn.Sequential(
                nn.Linear(observation_space, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_space),
        )

        self.sigma = torch.nn.Parameter(torch.zeros(1))

        self.mlp.apply(orthogonal_weight_init)
    
    def predict(self, obs):
        mu = self.mlp(obs)
        sigma = torch.exp(self.sigma.repeat(self.action_space))

        return distributions.Normal(mu, sigma)
    
    def forward(self, obs):
        dist = self.predict(obs)
        actions = dist.sample()
        return actions, dist.log_prob(actions)
    
    def evaluate(self, obs, actions):
        dist = self.predict(obs)
        return dist.log_prob(actions), dist.entropy()


class Critic(nn.Module):
    def __init__(self, observation_space):
        super().__init__()

        self.mlp = nn.Sequential(
                nn.Linear(observation_space, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
        )

        self.mlp.apply(orthogonal_weight_init)
    
    def forward(self, x):
        return self.mlp(x)

class PPO:
    def __init__(self, observation_space, action_space, num_environments):
        self.num_environments = num_environments

        '''
        # Pendulum-v1
        self.num_timesteps = 2048
        self.num_minibatches = 32
        self.num_epochs = 10
        self.gamma = 0.99
        self.lmbda = 0.95
        self.clip_ratio = 0.2
        self.learning_rate = 3e-4
        self.value_coefficient = 0.5
        self.entropy_coefficient = 0.0  
        self.minibatch_size = self.num_timesteps * self.num_environments // self.num_minibatches
        '''

        # HalfCheetah-v4
        self.num_timesteps = 8192
        self.num_epochs = 32
        self.gamma = 0.99
        self.lmbda = 0.95
        self.clip_ratio = 0.2
        self.learning_rate = 3e-4
        self.value_coefficient = 0.5
        self.entropy_coefficient = 0.0  

        # self.num_minibatches = 4096
        # self.minibatch_size = self.num_timesteps * self.num_environments // self.num_minibatches

        self.minibatch_size = 4096
        self.num_minibatches = self.num_timesteps * self.num_environments // self.minibatch_size

        self.buffer_len = num_environments * self.num_timesteps
        self.gae_discount = self.gamma * self.lmbda

        if self.buffer_len % self.num_minibatches != 0:
            raise RuntimeError(f"Buffer length({self.num_environments} * {self.num_timesteps} = {self.buffer_len}) must be a multiple of num_minibatches {self.num_minibatches}")

        self.actor = Actor(observation_space, action_space).cuda()
        self.critic = Critic(observation_space).cuda()

        self.traj_rewards = torch.zeros(self.num_environments, self.num_timesteps).cuda()
        self.traj_value = torch.zeros(self.num_environments, self.num_timesteps + 1).cuda()
        self.traj_observations = torch.zeros(self.num_environments, self.num_timesteps, observation_space).cuda()
        self.traj_dones = torch.zeros(self.num_environments, self.num_timesteps + 1).cuda()
        self.traj_actions = torch.zeros(self.num_environments, self.num_timesteps, action_space).cuda()
        self.traj_action_probs = torch.zeros(self.num_environments, self.num_timesteps, action_space).cuda()
        self.traj_advantage = torch.zeros(self.num_environments, self.num_timesteps).cuda()
        self.traj_returns = torch.zeros(self.num_environments, self.num_timesteps).cuda()

        self.running_episode_rewards = torch.zeros(self.num_environments).cuda()

        self.next_obs = torch.zeros(self.num_environments, observation_space).cuda()

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr = self.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr = self.learning_rate)

        self.tensorboard = SummaryWriter(log_dir="ppo_tests", flush_secs=10)

        self.should_render = False
        self.action_space = action_space
    
    def update_rollout_buffers(self, mdp):
        next_obs = self.next_obs
        next_dones = self.traj_dones[:, -1]

        episodic_reward_sum = 0
        reward_sum_count = 0

        for t in range(self.num_timesteps):
            obs = next_obs
            dones = next_dones

            action, action_probs = self.actor(next_obs)
            value = self.critic(obs)

            np_action = action.cpu().numpy().reshape(self.num_environments, self.action_space)
            _next_obs, rew, terminated, truncated, info = mdp.step(np_action)


            if self.should_render:
                img = cv2.cvtColor(np.array(mdp.envs[0].render()), cv2.COLOR_BGR2RGB)
                cv2.imshow("Pendulum-v1", img)
                cv2.waitKey(1)

            next_obs = torch.from_numpy(_next_obs).to(torch.float32).cuda()
            rew = torch.from_numpy(rew).cuda()
            next_dones = torch.from_numpy(np.logical_or(terminated, truncated)).cuda()

            self.running_episode_rewards += rew
            episodic_reward_sum += self.running_episode_rewards[next_dones == 1].sum()
            self.running_episode_rewards[next_dones == 1] = 0
            reward_sum_count += torch.sum(next_dones == 1)

            self.traj_observations[:, t] = obs.view(self.num_environments, -1)
            self.traj_rewards[:, t] = rew.view(self.num_environments)
            self.traj_value[:, t] = value.view(self.num_environments)
            self.traj_actions[:, t] = action.view(self.num_environments, -1)
            self.traj_action_probs[:, t] = action_probs.view(self.num_environments, -1)
            self.traj_dones[:, t] = dones.view(self.num_environments)
        
        self.traj_value[:, -1] = self.critic(next_obs).view(self.num_environments)
        self.traj_dones[:, -1] = next_dones.view(self.num_environments)
        self.next_obs = next_obs.view(self.num_environments, -1)

        return episodic_reward_sum / max(reward_sum_count, 1)

    def compute_advantage(self):
        still_good = 1 - self.traj_dones

        td_err = self.traj_rewards + self.gamma * self.traj_value[:, 1:] * still_good[:, 1:] - self.traj_value[:, :-1]

        self.traj_advantage[:, -1] = td_err[:, -1]
        for index in range(self.num_timesteps - 2, -1, -1):
            self.traj_advantage[:, index] = td_err[:, index] + self.gae_discount * self.traj_advantage[:, index + 1] * still_good[:, index + 1]
        
        # still confused on this one, need to look into it more. Doesn't this only hold when lambda=1?
        self.traj_returns = self.traj_advantage + self.traj_value[:, :-1]
    
    def update_policy(self, observations, old_actions, old_action_probs, advantage, returns):
        advantage_before_norm = advantage.mean().item()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        new_action_probs, new_action_entropy = self.actor.evaluate(observations, old_actions)
        new_values = self.critic(observations)
        
        # TODO: use multivariate normal distribution instead of normal, so network outputs may be correlated -> have network learn covariance.  For now we'll make independent and sum log probs
        new_action_probs, old_action_probs = new_action_probs.sum(dim = -1), old_action_probs.sum(dim = -1)

        ratio = torch.exp(new_action_probs - old_action_probs)

        objective = ratio * advantage
        clipped_obj = torch.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
        
        l_clip = torch.minimum(objective, clipped_obj).mean()

        l_vf = self.value_coefficient * torch.pow(returns - new_values, 2).mean() # TODO: maybe vf clipping, might not really help
        l_s = self.entropy_coefficient * new_action_entropy.squeeze().mean() # not sure why this returns shape [..., 1]

        actor_loss = -1 * (l_clip + l_s)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()
        
        critic_loss = l_vf
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()

        with torch.no_grad():
            r = new_action_probs - old_action_probs
            kl = torch.mean(torch.exp(r) - r - 1)
        
        return critic_loss.item(), ratio.mean().item(), advantage_before_norm, l_clip.item(), l_s.item(), kl.item()

    def train(self, mdp, num_iterations):
        first_obs, info = mdp.reset()
        self.next_obs = torch.from_numpy(first_obs).to(torch.float32).cuda()

        data_names = ["critic_loss", "pi_change", "advantage", "objective_loss", "entropy_loss", "kl_divergence"]
        logged_data = [[] for _ in data_names]

        
        for iter_num in range(num_iterations):    
            step_num = (iter_num+1) * self.num_environments * self.num_timesteps
            
            with torch.no_grad():
                mean_reward = self.update_rollout_buffers(mdp)
                self.compute_advantage()  

            for epoch in range(self.num_epochs):
                batches = torch.randperm(self.buffer_len).split(self.minibatch_size)
                for minibatch_index in batches:
                    obs = self.traj_observations.view(self.buffer_len, -1)[minibatch_index]
                    act = self.traj_actions.view(self.buffer_len, -1)[minibatch_index]
                    act_prob = self.traj_action_probs.view(self.buffer_len, -1)[minibatch_index]
                    adv = self.traj_advantage.view(self.buffer_len)[minibatch_index]
                    ret = self.traj_returns.view(self.buffer_len)[minibatch_index]


                    data = self.update_policy(obs, act, act_prob, adv, ret)

                    for idx, item in enumerate(data):
                        logged_data[idx].append(item)                    
                
            self.tensorboard.add_scalar("train/episode_reward", mean_reward, step_num)
            self.tensorboard.add_scalar("train/log_std", self.actor.sigma.item(), step_num)
            for name, data in zip(data_names, logged_data):
                self.tensorboard.add_scalar(f"train/{name}", np.mean(data), step_num)

            if mean_reward > 200:
                self.should_render = True

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {step_num} has an average episodic reward of {mean_reward}")
            