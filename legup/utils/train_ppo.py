from legup.train.agents.anymal import AnymalAgent
from legup.robots.mini_cheetah.mini_cheetah import MiniCheetah
from legup.train.models.anymal.teacher import CustomTeacherActorCriticPolicy
from legup.utils.callback import CustomLocalCallback, CustomWandbCallback
from legup.utils.wandb_wrapper import WandBWrapper

from stable_baselines3 import PPO

from omegaconf import DictConfig
import uuid
import os
import time

# Trains the agent using PPO from stable_baselines3. Tensorboard logging to './concurrent_training_tb' and saves model to ConcurrentTrainingEnv


def train_ppo(cfg: DictConfig, root_path: str):

    total_timesteps = cfg.environment.parallel_envs * \
        cfg.environment.n_steps * 1e6

    env = AnymalAgent(MiniCheetah, cfg.environment.parallel_envs,
                      f"{root_path}/robots/mini_cheetah/physical_models", "mini-cheetah.urdf", train_cfg=cfg.agent)

    cb = None

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
                 wandb_wrapper=wandb_wrapper, resume=False)

    model.save(f'saved_models/{uuid.uuid4().int}')

    if (cfg.environment.headless):
        wandb_wrapper.finish()


def run_training(model, total_timesteps, callback, cfg, id=None, wandb_wrapper=None, retry_count=0, resume=False):

    if resume:
        if cfg.envirinment.headless and wandb_wrapper is not None:
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

    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except Exception as e:
        except_time = time.time()

        new_retry_count = retry_count + 1

        elapsed = except_time - start_time

        # if no exception has occurred for 5 minutes, reset the retry count
        if elapsed > 300:
            new_retry_count = 0

        print(
            f"Caught exception #{new_retry_count} during training after {elapsed} seconds.")
        print(f"Exception: {e}")
        print("Retrying training...")

        run_training(model, total_timesteps=total_timesteps, callback=callback, cfg=cfg,
                     id=id, wandb_wrapper=wandb_wrapper, retry_count=new_retry_count, resume=True)
