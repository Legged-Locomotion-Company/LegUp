from abc import ABC, abstractmethod

from legup.common.abstract_env import AbstractEnv
from legup.common.abstract_agent import AbstractAgent


class AbstractAlgorithm(ABC):

    def __init__(self, env: AbstractEnv, agent: AbstractAgent):
        self.env = env
        self.agent = agent

    def collect_rollout(self):
        self.env.reset()
