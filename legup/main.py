import hydra
from omegaconf import OmegaConf

from legup.common.legup_config import LegupConfig


@hydra.main(config_path="config", config_name="config")
def main(cfg: LegupConfig):
    OmegaConf.merge(OmegaConf.structured(LegupConfig), cfg)
    device = 'cuda' if cfg.use_gpu else 'cpu'


if __name__ == '__main__':
    main()
