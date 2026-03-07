from evo_swarm.core.interfaces.agent import Agent
from evo_swarm.core.events import Event, EventType
import time
import random

class EvaluatorAgent(Agent):
    """
    Benchmarks candidates.
    Listens for TRAINING_COMPLETED.
    """
    def __init__(self, name: str = "Evaluator"):
        super().__init__(name)

    def handle_event(self, event: Event):
        if event.event_type == EventType.TRAINING_COMPLETED:
            self._evaluate_candidate(event)

    def _evaluate_candidate(self, event: Event):
        candidate_data = event.payload.get("candidate", {})
        train_loss = event.payload.get("train_loss", 0.5)
        candidate_id = event.candidate_id
        
        print(f"[{self.name}] Evaluating candidate {str(candidate_id)[:8]}...")
        
        start_event = Event(
            event_type=EventType.EVALUATION_STARTED,
            sender=self.name,
            candidate_id=candidate_id,
            generation=event.generation
        )
        self.publish(start_event)
        
        # Simulate evaluation
        time.sleep(1.0)
        
        # Mock calculation: 
        # higher fitness is better. Let's base it slightly on train_loss + randomness
        base_performance = max(0.0, 1.0 - train_loss)
        math_accuracy = min(1.0, max(0.0, base_performance + random.uniform(-0.1, 0.2)))
        
        fitness = (math_accuracy * 0.7) + (random.uniform(0, 0.3)) # abstract combined fitness
        
        print(f"[{self.name}] Evaluated {str(candidate_id)[:8]}: fitness={fitness:.4f}")
        
        result_event = Event(
            event_type=EventType.EVALUATION_RESULT,
            sender=self.name,
            candidate_id=candidate_id,
            generation=event.generation,
            payload={
                "fitness": fitness, 
                "metrics": {"math_accuracy": math_accuracy, "train_loss": train_loss},
                "candidate": candidate_data
            }
        )
        self.publish(result_event)
