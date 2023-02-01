import isaac
import isaacgym

from legup.utils.gpu_vec_env import GPUVecEnv
from legup.train.agents.anymal import AnymalAgent
from legup.utils.callback import CustomLocalCallback, CustomWandbCallback
from legup.utils.wandb_wrapper import WandBWrapper
from legup.robots.mini_cheetah.mini_cheetah import MiniCheetah
from legup.train.models.anymal.teacher import CustomTeacherActorCriticPolicy

import cv2
import hydra
from omegaconf import DictConfig
import os
# from stable_baselines3 import PPO
from gpu_gym.ppo import GPUPPO as PPO
import uuid
import time

root_path = None


def train_ppo(cfg: DictConfig, root_path: str):

    total_timesteps = cfg.environment.parallel_envs * \
        cfg.environment.n_steps * 1e7

    env = AnymalAgent(MiniCheetah, cfg.environment.parallel_envs, cfg.environment.curriculum_exponent,
                      f"{root_path}/robots/mini_cheetah/physical_models", "mini-cheetah.urdf", train_cfg=cfg.agent)

    cb = None

    wandb_wrapper = None

    if (not cfg.environment.headless):
        training_id = str(uuid.uuid4())
        cb = CustomLocalCallback(env, training_id, root_path)
    else:
        wandb_config = {
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

        wandb_wrapper = WandBWrapper(wandb_config)

        cb = CustomWandbCallback(
            env, training_id=wandb_wrapper.id, root_path=root_path)

    model = PPO(CustomTeacherActorCriticPolicy, env, tensorboard_log='./concurrent_training_tb', verbose=1,
                batch_size=cfg.environment.batch_size, n_steps=cfg.environment.n_steps, n_epochs=cfg.environment.n_epochs, ent_coef=cfg.environment.entropy_coef,
                learning_rate=cfg.environment.learning_rate, clip_range=cfg.environment.clip_range, gae_lambda=cfg.environment.gae_lambda, gamma=cfg.environment.discount, vf_coef=cfg.environment.value_coef)

    run_training(model, total_timesteps, cfg=cfg, callback=cb,
                 wandb_wrapper=wandb_wrapper, resume=False, log_dump_func=env.dump_log)

    model.save(f'saved_models/{uuid.uuid4().int}')

    if (cfg.environment.headless):
        wandb_wrapper.finish()


def run_training(model, total_timesteps, callback, cfg, id=None, wandb_wrapper=None, retry_count=0, resume=False, log_dump_func=None):

    if resume:
        if cfg.environment.headless and wandb_wrapper is not None:
            wandb_wrapper.recover()
        else:
            model.load(f'saved_models/{callback.training_id}')

    # Kill if there have been more than 5 exceptions each within 5 minutes of each other
    if retry_count > 5:
        print("Failed to resume training after 5 retries. Exiting")
        return

    os.listdir()

    # save the start time so that we know how long between exceptions
    start_time = time.time()

    model.learn(total_timesteps=total_timesteps, callback=callback)

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
        eval_ppo(cfg, root_path)
    else:
        train_ppo(cfg, root_path)


if __name__ == '__main__':
    root_path = os.getcwd()
    run()
