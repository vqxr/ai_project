from abc import ABC, abstractmethod
from collections.abc import Callable

from evo_swarm.core.events import Event


class Agent(ABC):
    """
    Base Agent Interface.
    Every agent must be able to handle events and potentially emit events.
    """
    def __init__(self, name: str):
        self.name = name
        self.event_bus: Callable[[Event], None] | None = None # Will be injected by the scheduler/runner
        
    def set_event_bus(self, commit_event: Callable[[Event], None]):
        """Inject a callback to allow the agent to publish events."""
        self.event_bus = commit_event
        
    def publish(self, event: Event):
        if self.event_bus:
            self.event_bus(event)
            
    @abstractmethod
    def handle_event(self, event: Event):
        """
        Process an incoming event.
        The agent should decide if it cares about this event and take action.
        """
        pass
