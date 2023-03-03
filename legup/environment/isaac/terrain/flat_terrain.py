from abc import ABC, abstractmethod

import numpy as np
import torch
class FlatTerrain(ABC):
    def create_heightfield(self, patch_width: int, horizontal_scale: float, vertical_scale: float) -> np.ndarray:
        return np.zeros((patch_width, patch_width))
