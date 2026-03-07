from evo_swarm.core.interfaces.agent import Agent
from evo_swarm.core.events import Event, EventType

class CuratorAgent(Agent):
    """
    Manages datasets and tasks.
    Triggers new generations by broadcasting a dataset update.
    """
    def __init__(self, name: str = "Curator"):
        super().__init__(name)

    def handle_event(self, event: Event):
        # In a real system the curator might listen for failure reports 
        # and adjust the sampling or curriculum accordingly.
        pass

    def trigger_new_generation(self, generation: int):
        """Called externally (e.g., by main runner) to kick off a loop."""
        print(f"\n=========================================")
        print(f"[{self.name}] Initiating Generation {generation}")
        print(f"=========================================\n")
        
        update_event = Event(
            event_type=EventType.DATASET_UPDATE,
            sender=self.name,
            generation=generation,
            payload={
                "dataset_uri": f"local://math_gen_{generation}",
                "sample_size": 1000
            }
        )
        self.publish(update_event)
