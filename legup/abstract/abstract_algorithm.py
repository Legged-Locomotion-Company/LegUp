from abc import ABC, abstractmethod

from legup.abstract.abstract_env import AbstractEnv
from legup.agents import AbstractAgent


class AbstractAlgorithm(ABC):

    def __init__(self, env: AbstractEnv, agent: AbstractAgent):
        self.env = env
        self.agent = agent

    def collect_rollout(self):
        self.env.reset()

    @abstractmethod
    def learn(self):
        pass
