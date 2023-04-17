from abc import ABC, abstractmethod

from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class FlatTerrain(ABC):
    num_robots: int
    num_patches: int
    
    def create_heightfield(self, patch_width: int, horizontal_scale: float, vertical_scale: float) -> np.ndarray:
        return np.zeros((patch_width, patch_width))

    def get_num_robots(self) -> int:
        return self.num_robots
    
    def get_num_patches(self) -> int:
        return self.num_patches