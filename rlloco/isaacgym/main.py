from rlloco.agents.concurrent_training import ConcurrentTrainingEnv

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

import cv2

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
        # cv2.imshow('training', self.env_.render())
        # cv2.waitKey(1)
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
        #print("POLICY UPDATE (ROLLOUT END)")
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

# number of parallel environments to run
PARALLEL_ENVS = 8192 #32

# number of experiences to collect per parallel environment
N_STEPS = 250 # 256

# number of time we go through the entire rollout
N_EPOCHS = 2 # 5

# minibatch size
BATCH_SIZE = 10240

# total number of timesteps where each collection is one timestep
TOTAL_TIMESTEPS = PARALLEL_ENVS * N_STEPS * 10000

ENTROPY_COEF = 0.005

LEARNING_RATE = 5e-4

GAE_LAMBDA = 0.95

DISCOUNT = 0.996

CLIP_RANGE = 0.2

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


def train_ppo():
    env = GPUVecEnv(PARALLEL_ENVS, "/home/rohan/Documents/rlloco/rlloco/assets", "mini-cheetah.urdf")
    cb = CustomCallback(env)

    model = PPO('MlpPolicy', env, tensorboard_log = './concurrent_training_tb', verbose = 2, policy_kwargs = {'net_arch': [512, 256, 64]}, 
                batch_size = BATCH_SIZE, n_steps = N_STEPS, n_epochs = N_EPOCHS, ent_coef = ENTROPY_COEF, learning_rate = LEARNING_RATE, clip_range = CLIP_RANGE, gae_lambda = GAE_LAMBDA, gamma = DISCOUNT)

    model.learn(total_timesteps = TOTAL_TIMESTEPS, callback = cb)
    model.save('ConcurrentTrainingEnv')

def eval_ppo():
    env = GPUVecEnv(PARALLEL_ENVS, "/home/rohan/Documents/rlloco/rlloco/assets", "mini-cheetah.urdf")
    model = PPO.load('ConcurrentTrainingEnv')

    obs = env.reset()
    for _ in range(100000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == '__main__':
    train_ppo()