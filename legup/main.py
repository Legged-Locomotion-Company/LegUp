from legup.train.agents.concurrent_training import ConcurrentTrainingEnv

import torch
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import cv2

import sys

import wandb

from wandb.integration.sb3 import WandbCallback

import numpy as np


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
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env_ = env
        self.video_buffer = []

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

        numpy_video = np.array(self.video_buffer).transpose([0,3,1,2])

        wandb.log(
            {"video": wandb.Video(numpy_video, fps=60, format="gif")})

        infos = self.locals['infos'][0]
        for idx, name in enumerate(infos['names']):
            self.logger.record(f"rewards/{name}", infos['terms'][idx].item())

        self.model.save(os.path.join('saved_models', str(self.num_timesteps)))

        self.video_buffer = []


# number of parallel environments to run
PARALLEL_ENVS = 4096

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


def train_ppo(headless=False):
    env = GPUVecEnv(
        PARALLEL_ENVS, f"{os.getcwd()}/robots/mini_cheetah/physical_models", "mini-cheetah.urdf")

    cb = CustomCallback(env)

    if (headless):
        config = {
            "learning_rate": LEARNING_RATE,
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
        }
        wandb.init(project="LegUp", config=config, entity="legged-locomotion-company")

        cb = CustomWandbCallback(env)

    model = PPO('MlpPolicy', env, tensorboard_log='./concurrent_training_tb', verbose=1, policy_kwargs={'net_arch': [512, 256, 64]},
                batch_size=BATCH_SIZE, n_steps=N_STEPS, n_epochs=N_EPOCHS, ent_coef=ENTROPY_COEF, learning_rate=LEARNING_RATE, clip_range=CLIP_RANGE, gae_lambda=GAE_LAMBDA, gamma=DISCOUNT, vf_coef=VALUE_COEF)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb)
    model.save('ConcurrentTrainingEnv')

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

    if len(sys.argv) >= 2:
        if sys.argv[1] == 'train':
            headless = False
            if len(sys.argv) >= 3:
                if sys.argv[2] == 'headless':
                    headless = True
                elif sys.argv[2] != 'headful':
                    print_usage()

            train_ppo(headless)
        elif sys.argv[1] == 'eval':
            eval_ppo()
        else:
            print_usage()

    train_ppo()
    # eval_ppo()
