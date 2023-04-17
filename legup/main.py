import isaacgym

import hydra
import torch
from omegaconf import DictConfig


from legup.common.abstract_env import AbstractEnv
from legup.common.abstract_agent import AbstractAgent

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

    kwargs['agent'] = hydra.utils.instantiate(cfg.agent, **kwargs)
    kwargs['env'] = hydra.utils.instantiate(cfg.environment, **kwargs)

    

if __name__ == '__main__':
    main()
