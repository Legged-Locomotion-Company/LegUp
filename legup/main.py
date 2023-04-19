import isaacgym  # need to import isaacgym before torch :(

import hydra
import torch
from omegaconf import DictConfig

from legup.robots.mini_cheetah import mini_cheetah

'''
TODO:
- IsaacGym
  - support custom camera/viewer implementations, so we can switch between camera sensor and default viewer
  - draw dynamics information as text onto camera (maybe support better vis on viewer)
  - save simulation state before closing and restore if needed
  - multi GPU
- General
  - main execution loop where RL/agent/env interacts with eachother. This is where we query agent to reset envs, and recreate the main env if so
- Reinforcement Learning
  - SB3 PPO integration
  - Custom PPO
'''


def get_device(kwargs) -> torch.device:
    if 'device' in kwargs:
        dev_str = torch.device(kwargs['device'])
    elif torch.cuda.is_available():
        dev_str = 'cuda'
    else:
        dev_str = 'cpu'

    return torch.device(dev_str)


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    kwargs = dict(cfg['kwargs'])
    kwargs['device'] = get_device(kwargs)

    kwargs['robot'] = mini_cheetah
    kwargs['agent'] = hydra.utils.instantiate(cfg.agent, **kwargs)
    kwargs['env'] = hydra.utils.instantiate(cfg.environment, **kwargs)


if __name__ == '__main__':
    main()
