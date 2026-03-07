from evo_swarm.core.interfaces.agent import Agent
from evo_swarm.core.events import Event, EventType

class CriticMutatorAgent(Agent):
    """
    Analyzes failure modes and proposes mutations.
    Listens for EVALUATION_RESULT.
    """
    def __init__(self, name: str = "CriticMutator", target_fitness: float = 0.85):
        super().__init__(name)
        self.target_fitness = target_fitness

    def handle_event(self, event: Event):
        if event.event_type == EventType.EVALUATION_RESULT:
            self._analyze_result(event)

    def _analyze_result(self, event: Event):
        fitness = event.payload.get("fitness", 0.0)
        candidate_id = event.candidate_id
        
        if fitness >= self.target_fitness:
            print(f"[{self.name}] Candidate {str(candidate_id)[:8]} is a CHAMPION! (Fitness: {fitness:.4f})")
            champ_event = Event(
                event_type=EventType.CHAMPION_SELECTED,
                sender=self.name,
                candidate_id=candidate_id,
                generation=event.generation,
                payload=event.payload
            )
            self.publish(champ_event)
        else:
            print(f"[{self.name}] Candidate {str(candidate_id)[:8]} failed target criteria (Fitness: {fitness:.4f}). Suggesting mutation...")
            mutation_plan = {
                "num_layers": 4, # naive constant mutation for V1 mock
                "learning_rate": 0.0005,
                "reasoning": "Fitness too low, increasing model depth slightly."
            }
            
            mutation_event = Event(
                event_type=EventType.MUTATION_PLAN,
                sender=self.name,
                candidate_id=candidate_id,
                generation=event.generation,
                payload={"mutations": mutation_plan, "parent_fitness": fitness}
            )
            self.publish(mutation_event)
