from legup.common.abstract_terrain import AbstractTerrain

from isaacgym import terrain_utils
import numpy as np


class RoughTerrain(AbstractTerrain):
    def __init__(self, min_height: float, max_height: float, step: float, downsampled_scale: float):
        self.min_height = min_height
        self.max_height = max_height
        self.step = step
        self.downsampled_scale = downsampled_scale

    def create_heightfield(self, patch_width: int, horizontal_scale: float, vertical_scale: float) -> np.ndarray:
        subterrain = self.subterrain(
            patch_width, horizontal_scale, vertical_scale)

        return terrain_utils.random_uniform_terrain(subterrain, min_height=self.min_height, max_height=self.max_height, step=self.step, downsampled_scale=self.downsampled_scale).height_field_raw  # type: ignore
