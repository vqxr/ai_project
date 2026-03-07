from evo_swarm.core.interfaces.agent import Agent
from evo_swarm.core.events import Event, EventType
from evo_swarm.core.registry.registry import Registry
from evo_swarm.core.genomes import Candidate

class GenerationManager(Agent):
    """
    Acts as the main orchestrator for evolution tracking.
    Listens to the event stream, updates candidate statuses, and writes them to the Registry.
    """
    def __init__(self, registry: Registry, max_generations: int = 5, name: str = "GenerationManager"):
        super().__init__(name)
        self.registry = registry
        self.max_generations = max_generations
        self.current_generation = 0

    def handle_event(self, event: Event):
        # Always intercept PROPOSAL to store new genomes
        if event.event_type == EventType.PROPOSAL:
            data = event.payload.get("candidate", {})
            candidate = Candidate(**data)
            self.registry.save_candidate(candidate)
            print(f"[{self.name}] Registered new Candidate {candidate.id[:8]} to LocalRegistry.")
        
        # Intercept results to update fitness and state
        elif event.event_type == EventType.EVALUATION_RESULT:
            candidate_id = event.candidate_id
            candidate = self.registry.get_candidate(candidate_id)
            if candidate:
                candidate.status = "evaluated"
                candidate.fitness_score = event.payload.get("fitness")
                candidate.metrics = event.payload.get("metrics", {})
                self.registry.save_candidate(candidate)
                print(f"[{self.name}] Updated Candidate {candidate_id[:8]} in LocalRegistry with metrics.")
        
        # Track generations
        elif event.event_type == EventType.CHAMPION_SELECTED:
            print(f"[{self.name}] Evolution complete! A champion was found.")
            # We could stop the system here by gracefully letting the queue empty.
            self.publish(Event(event_type="SYSTEM_HALT", sender=self.name))
