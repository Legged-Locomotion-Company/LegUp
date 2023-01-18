import isaac
import isaacgym

from legup.robots.mini_cheetah.mini_cheetah import MiniCheetah
from legup.train.agents.anymal import AnymalAgent
from legup.train.agents.concurrent_training import ConcurrentTrainingEnv
from legup.train.models.anymal.teacher import CustomTeacherActorCriticPolicy

import uuid

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback

import torch


root_path = None

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
        super().__init__(model_save_path='saved_models', verbose=verbose)
        self.env_ = env
        self.video_buffer = []

        Monitor(env)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        TODO add a way to save the video every n steps

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        super()._on_step()

        self.video_buffer.append(self.env_.render())

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


# Wrapper for ConcurrentTrainingEnv to convert returned torch tensors to numpy and input numpy arrays to torch tensors


# TODO: generalize it to not just the `ConcurrentTrainingEnv` environment
class GPUVecEnv(AnymalAgent):
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


def train_ppo(cfg: DictConfig):

    total_timesteps = cfg.environment.parallel_envs * \
        cfg.environment.n_steps * 1e6

    # root_path = os.path.dirname(os.path.abspath(__file__))
    # root_path = '/home/mishmish/Documents/LegUp/legup'
    # root_path = '/opt/leggedloco/legup'

    # print("POOOOP    ", root_path)

    env = AnymalAgent(MiniCheetah, cfg.environment.parallel_envs,
                      f"{root_path}/robots/mini_cheetah/physical_models", "mini-cheetah.urdf", train_cfg=cfg.agent)

    cb = None

    if (not cfg.environment.headless):
        cb = CustomCallback(env)
    else:
        config = {
            "env_name": cfg.agent.env_name,
            "parallel_envs": cfg.environment.parallel_envs,
            "n_steps": cfg.environment.n_steps,
            "n_epochs": cfg.environment.n_epochs,
            "batch_size": cfg.environment.batch_size,
            "total_timesteps": total_timesteps,
            "entropy_coef": cfg.environment.entropy_coef,
            "value_coef": cfg.environment.value_coef,
            "learning_rate": cfg.environment.learning_rate,
            "gae_lambda": cfg.environment.gae_lambda,
            "discount": cfg.environment.discount,
            "clip_range": cfg.environment.clip_range,
        }
        wandb.init(project="LegUp", config=config, entity="legged-locomotion-company",
                   sync_tensorboard=True, monitor_gym=True, save_code=True)

        cb = CustomWandbCallback(env)

    model = PPO(CustomTeacherActorCriticPolicy, env, verbose=1,
                batch_size=cfg.environment.batch_size, n_steps=cfg.environment.n_steps, n_epochs=cfg.environment.n_epochs, ent_coef=cfg.environment.entropy_coef,
                learning_rate=cfg.environment.learning_rate, clip_range=cfg.environment.clip_range, gae_lambda=cfg.environment.gae_lambda, gamma=cfg.environment.discount, vf_coef=cfg.environment.value_coef)

    model.learn(total_timesteps=total_timesteps, callback=cb)
    model.save(f'saved_models/{uuid.uuid4().int}')

# Runs the agent based on a saved model


def eval_ppo(cfg: DictConfig):
    env = GPUVecEnv(
        1, f"{os.getcwd()}/robots/mini_cheetah/physical_models", "mini-cheetah.urdf")
    model = PPO.load('saved_models/503316480.zip')

    obs = env.reset()
    for _ in range(100000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        cv2.imshow('training', env.render())
        cv2.waitKey(1)


@hydra.main(config_path="config", config_name="config")
def run(cfg: DictConfig):
    if cfg.eval:
        eval_ppo(cfg)
    else:
        train_ppo(cfg)


if __name__ == '__main__':
    root_path = os.getcwd()
    run()
