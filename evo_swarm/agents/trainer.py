from evo_swarm.core.interfaces.agent import Agent
from evo_swarm.core.events import Event, EventType
import time
import random

class TrainerAgent(Agent):
    """
    Trains or fine-tunes candidate systems.
    Listens for PROPOSAL.
    """
    def __init__(self, name: str = "Trainer"):
        super().__init__(name)

    def handle_event(self, event: Event):
        if event.event_type == EventType.PROPOSAL:
            self._train_candidate(event)

    def _train_candidate(self, event: Event):
        candidate_data = event.payload.get("candidate", {})
        candidate_id = event.candidate_id
        
        print(f"[{self.name}] Started training candidate {str(candidate_id)[:8]}...")
        
        start_event = Event(
            event_type=EventType.TRAINING_STARTED,
            sender=self.name,
            candidate_id=candidate_id,
            generation=event.generation
        )
        self.publish(start_event)
        
        # Simulate training delay and outcome
        time.sleep(1.0) 
        
        # Mock metrics from training
        train_loss = random.uniform(0.1, 0.5)
        
        print(f"[{self.name}] Completed training {str(candidate_id)[:8]} (loss={train_loss:.4f})")
        
        completed_event = Event(
            event_type=EventType.TRAINING_COMPLETED,
            sender=self.name,
            candidate_id=candidate_id,
            generation=event.generation,
            payload={"train_loss": train_loss, "candidate": candidate_data}
        )
        self.publish(completed_event)
