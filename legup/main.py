from legup.train.agents.concurrent_training import ConcurrentTrainingEnv

import cv2

from legup.train.agents.anymal import AnymalAgent

from legup.robots.mini_cheetah.mini_cheetah import MiniCheetah

from legup.train.models.anymal.teacher import CustomTeacherActorCriticPolicy

import torch
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from wandb.integration.sb3 import WandbCallback

import numpy as np
import argparse


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, env, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.env_ = env
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # print("BEFORE FIRST ROLLOUT TRAINING START)")
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        # print("BEFORE ROLLOUT (ROLLOUT START)")
        pass

    def _on_step(self) -> bool:
        cv2.imshow('training', self.env_.render())
        cv2.waitKey(1)
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

        infos = self.locals['infos'][0]
        for idx, name in enumerate(infos['names']):
            self.logger.record(f"rewards/{name}", infos['terms'][idx].item())

        self.model.save(os.path.join('saved_models', str(self.num_timesteps)))

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


class CustomWandbCallback(WandbCallback):
    def __init__(self, env, verbose=1):
        super().__init__(gradient_save_freq=100,
                         model_save_path='saved_models', verbose=verbose)
        self.env_ = env
        self.video_buffer = []

        Monitor(env)

    def _on_step(self) -> bool:
        super()._on_step()

        self.video_buffer.append(self.env_.render())

        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

        super()._on_rollout_end()

        # For some reason the video needs to be transposed to frames, channels, height, width

        numpy_video = np.array(self.video_buffer).transpose([0, 3, 1, 2])

        wandb.log(
            {"video": wandb.Video(numpy_video, fps=20, format="gif")})

        infos = self.locals['infos'][0]
        for idx, name in enumerate(infos['names']):
            self.logger.record(f"rewards/{name}", infos['terms'][idx].item())

        self.model.save(os.path.join('saved_models', str(self.num_timesteps)))

        self.video_buffer = []


# number of parallel environments to run
PARALLEL_ENVS = 128

# number of experiences to collect per parallel environment
N_STEPS = 256

# number of time we go through the entire rollout
N_EPOCHS = 5

# minibatch size
BATCH_SIZE = 32768

# total number of timesteps where each collection is one timestep
TOTAL_TIMESTEPS = PARALLEL_ENVS * N_STEPS * 10000

ENTROPY_COEF = 0.01

VALUE_COEF = 0.5

LEARNING_RATE = 3e-4

GAE_LAMBDA = 0.95

DISCOUNT = 0.99

CLIP_RANGE = 0.2

# Wrapper for ConcurrentTrainingEnv to convert returned torch tensors to numpy and input numpy arrays to torch tensors


# TODO: generalize it to not just the `ConcurrentTrainingEnv` environment
class GPUVecEnv(ConcurrentTrainingEnv):
    def step(self, actions):
        actions = torch.from_numpy(actions).cuda()

        new_obs, reward, dones, infos = super().step(actions)

        new_obs = new_obs.cpu().detach().numpy()
        reward = reward.cpu().detach().numpy()
        dones = dones.cpu().detach().numpy()

        return new_obs, reward, dones, infos

    def reset(self):
        obs = super().reset()
        return obs.cpu().detach().numpy()

# Trains the agent using PPO from stable_baselines3. Tensorboard logging to './concurrent_training_tb' and saves model to ConcurrentTrainingEnv


def train_ppo(headless=False, env_name="ConcurrentTrainingEnv", parallel_envs=4096, n_steps=256, n_epochs=5, batch_size=32768, entropy_coef=0.01, value_coef=0.5, learning_rate=3e-4, gae_lambda=0.95, discount=0.99, clip_range=0.2):
    total_timesteps = parallel_envs * n_steps * 10000

    # env = GPUVecEnv(
    #     parallel_envs, f"{os.getcwd()}/robots/mini_cheetah/physical_models", "mini-cheetah.urdf")

    
    reward_scales = {}
    reward_scales['velocity'] = 0.0
    reward_scales['body_motion'] = 0.0
    reward_scales['foot_clearance'] = 0.0
    reward_scales['shank_clearance'] = 0.0
    reward_scales['joint_velocity'] = 0.0
    reward_scales['joint_constraints'] = 0.0
    reward_scales['target_smoothness'] = 0.0
    reward_scales['torque'] = 0.0
    reward_scales['slip'] = 0.0

    train_cfg = {}
    train_cfg["max_tilt"] = 3*torch.pi/2
    train_cfg["max_torque"] = 90.0

    train_cfg['knee_threshold'] = [0.5, 0.5, 0.5, 0.5]

    train_cfg['reward_scales'] = reward_scales


    env = AnymalAgent(MiniCheetah, PARALLEL_ENVS, f"{os.getcwd()}/robots/mini_cheetah/physical_models", "mini-cheetah.urdf", train_cfg=train_cfg)


    cb = None

    if (not headless):
        cb = CustomCallback(env)
    else:
        config = {
            "env_name": env_name,
            "parallel_envs": parallel_envs,
            "n_steps": n_steps,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "total_timesteps": total_timesteps,
            "entropy_coef": entropy_coef,
            "value_coef": value_coef,
            "learning_rate": learning_rate,
            "gae_lambda": gae_lambda,
            "discount": discount,
            "clip_range": clip_range,
        }
        wandb.init(project="LegUp", config=config, entity="legged-locomotion-company",
                   sync_tensorboard=True, monitor_gym=True, save_code=True)

        cb = CustomWandbCallback(env)

    model = PPO(CustomTeacherActorCriticPolicy, env, tensorboard_log='./concurrent_training_tb', verbose=1, policy_kwargs={'net_arch': [512, 256, 64]},
                batch_size=batch_size, n_steps=n_steps, n_epochs=n_epochs, ent_coef=entropy_coef, learning_rate=learning_rate, clip_range=clip_range, gae_lambda=gae_lambda, gamma=discount, vf_coef=value_coef)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb)
    model.save(model)

# Runs the agent based on a saved model


def eval_ppo():
    env = GPUVecEnv(
        1, f"{os.getcwd()}/robots/mini_cheetah/physical_models", "mini-cheetah.urdf")
    model = PPO.load('saved_models/503316480.zip')

    obs = env.reset()
    for _ in range(100000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        cv2.imshow('training', env.render())
        cv2.waitKey(1)


if __name__ == '__main__':

    def print_usage():
        print('Valid usage is python3 main.py [train|eval] [headless|headful]')

    parser = argparse.ArgumentParser(
        prog='LegUp',
        description='LegUp is a legged locomotion simulator and reinforcement learning framework.',
        epilog='',)

    parser.add_argument('mode', type=str, choices=['train', 'eval'])

    parser.add_argument('env', type=str)
    parser.add_argument('-l', '--headless', action='store_true', default=False)
    parser.add_argument('-p', '--parallel_envs', type=int, default=4096)
    parser.add_argument('-n', '--n_steps', type=int, default=256)
    parser.add_argument('-e', '--n_epochs', type=int, default=5)
    parser.add_argument('-b', '--batch_size', type=int, default=32768)

    # add this functionality later
    # parser.add_argument('-c', '--checkpoint', type=str, default=None)

    args = parser.parse_args()

    if args.mode == 'train':
        train_ppo(
            headless=args.headless,
            env_name=args.env,
            parallel_envs=args.parallel_envs,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,)

    elif args.mode == 'eval':
        eval_ppo()
