from abc import ABC, abstractmethod

from evo_swarm.core.interfaces.agent import Agent


class Scheduler(ABC):
    """
    Coordinates the execution of Agents in the swarm.
    """
    @abstractmethod
    def register_agent(self, agent: Agent):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass
