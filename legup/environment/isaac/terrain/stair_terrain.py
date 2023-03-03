from legup.common.abstract_terrain import AbstractTerrain


from isaacgym import terrain_utils 
import numpy as np
import torch

class StairTerrain(AbstractTerrain):

    def __init__(self, stair_width: float, stair_height: float):
        self.stair_width = stair_width
        self.stair_height = stair_height
    
    def create_heightfield(self, patch_width: int,
                           horizontal_scale: float,
                           vertical_scale: float) -> np.ndarray:

        subterrain = self.subterrain(patch_width, horizontal_scale, vertical_scale)
        return terrain_utils.stairs_terrain(subterrain, step_width = self.stair_width, step_height=self.stair_height).height_field_raw
