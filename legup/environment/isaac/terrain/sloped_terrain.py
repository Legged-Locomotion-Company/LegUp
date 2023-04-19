from isaacgym import terrain_utils
from dataclasses import dataclass
import numpy as np

from legup.abstract.abstract_terrain import AbstractTerrain


@dataclass
class SlopedTerrain(AbstractTerrain):
    num_robots: int
    num_patches: int
    slope: float

    def create_heightfield(self, patch_width: int, horizontal_scale: float, vertical_scale: float) -> np.ndarray:
        subterrain = self.subterrain(
            patch_width, horizontal_scale, vertical_scale)

        # For some reason terrain_utils.sloped_terrain() wants slope to be an int, but it should be a float so we are ignoring
        # type: ignore
        return terrain_utils.sloped_terrain(subterrain, slope=self.slope).height_field_raw

    def get_num_robots(self) -> int:
        return self.num_robots

    def get_num_patches(self) -> int:
        return self.num_patches
