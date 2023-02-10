from abc import ABC, abstractmethod

class AbstractTerrain(ABC):

    @abstractmethod
    def create_heightfield(self):
        pass