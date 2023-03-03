from isaacgym import terrain_utils
import numpy as np

from legup.common.abstract_terrain import AbstractTerrain




class SlopedTerrain(AbstractTerrain):
    def __init__(self, slope: float):
        self.slope = slope

    def create_heightfield(self, patch_width: int, horizontal_scale: float, vertical_scale: float) -> np.ndarray:
        subterrain = self.subterrain(patch_width, horizontal_scale, vertical_scale)

        # For some reason terrain_utils.sloped_terrain() wants slope to be an int, but it should be a float so we are ignoring
        return terrain_utils.sloped_terrain(subterrain, slope = self.slope).height_field_raw # type: ignore