from legup.common.abstract_terrain import AbstractTerrain

from isaacgym import terrain_utils 
from dataclasses import dataclass
import numpy as np
import torch

class StairTerrain(AbstractTerrain):
    num_robots: int
    num_patches: int
    stair_width: float
    stair_height: float

    def create_heightfield(self, patch_width: int,
                           horizontal_scale: float,
                           vertical_scale: float) -> np.ndarray:

        subterrain = self.subterrain(patch_width, horizontal_scale, vertical_scale)
        return terrain_utils.stairs_terrain(subterrain, step_width = self.stair_width, step_height=self.stair_height).height_field_raw
    
    def get_num_robots(self) -> int:
        return self.num_robots
    
    def get_num_patches(self) -> int:
        return self.num_patches