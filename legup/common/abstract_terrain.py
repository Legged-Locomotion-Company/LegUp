from abc import ABC, abstractmethod

from isaacgym import terrain_utils

import numpy as np


class AbstractTerrain(ABC):

    def subterrain(self, patch_width: int, horizontal_scale: float, vertical_scale: float):
        return terrain_utils.SubTerrain(width=patch_width, length=patch_width, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
    
    @abstractmethod
    def get_num_robots(self) -> int:
        pass
    
    @abstractmethod
    def get_num_patches(self) -> int:
        pass

    @abstractmethod
    def create_heightfield(self, patch_width: int, horizontal_scale: float, vertical_scale: float) -> np.ndarray:
        pass
