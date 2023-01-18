import isaac
import isaacgym

from legup.utils.gpu_vec_env import GPUVecEnv
from legup.utils.train_ppo import train_ppo

import cv2
import hydra
from omegaconf import DictConfig
import os
from stable_baselines3 import PPO


root_path = None

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
