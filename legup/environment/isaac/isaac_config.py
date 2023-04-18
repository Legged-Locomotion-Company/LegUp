from legup.common.legup_config import QuiltSimConfig, AssetConfig, SimulationConfig, CameraConfig

from dataclasses import dataclass


@dataclass
class IsaacConfig(QuiltSimConfig):
    asset_config: AssetConfig
    sim_config: SimulationConfig
    camera_config: CameraConfig
