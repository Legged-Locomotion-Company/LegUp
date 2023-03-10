from legup.common.abstract_terrain import AbstractTerrain

from isaacgym import terrain_utils
from dataclasses import dataclass
import numpy as np

@dataclass
class RoughTerrain(AbstractTerrain):
    num_robots: int
    num_patches: int
    min_height: float
    max_height: float
    step: float
    downsampled_scale: float

    def create_heightfield(self, patch_width: int, horizontal_scale: float, vertical_scale: float) -> np.ndarray:
        subterrain = self.subterrain(
            patch_width, horizontal_scale, vertical_scale)

        return terrain_utils.random_uniform_terrain(subterrain, min_height=self.min_height, max_height=self.max_height, step=self.step, downsampled_scale=self.downsampled_scale).height_field_raw  # type: ignore
    
    def get_num_robots(self) -> int:
        return self.num_robots
    
    def get_num_patches(self) -> int:
        return self.num_patches
