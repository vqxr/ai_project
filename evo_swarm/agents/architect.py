from typing import Dict, Any
from evo_swarm.core.interfaces.agent import Agent
from evo_swarm.core.events import Event, EventType
from evo_swarm.core.genomes import Genome, Candidate
import uuid

class ArchitectAgent(Agent):
    """
    Proposes new candidate system designs.
    Listens for DATASET_UPDATE (to start generation) or MUTATION_PLAN.
    """
    def __init__(self, name: str = "Architect"):
        super().__init__(name)

    def handle_event(self, event: Event):
        if event.event_type == EventType.DATASET_UPDATE:
            self._propose_initial_candidate(event)
        elif event.event_type == EventType.MUTATION_PLAN:
            self._propose_mutated_candidate(event)

    def _propose_initial_candidate(self, event: Event):
        # Creates a naive baseline candidate (Generation 0)
        genome = Genome(
            model_family="small_transformer",
            num_layers=2,
            hidden_dimension=64,
            optimizer="adamw",
            learning_rate=1e-3,
            batch_size=16,
            memory_policy="none",
            retrieval_policy="none",
            curriculum_strategy="random",
            loss_weighting={"ce_loss": 1.0}
        )
        
        candidate = Candidate(
            id=str(uuid.uuid4()),
            parent_ids=[],
            generation=0,
            genome=genome,
            status="proposed"
        )
        
        print(f"[{self.name}] Proposed Gen 0 Candidate: {candidate.id[:8]}")
        
        proposal_event = Event(
            event_type=EventType.PROPOSAL,
            sender=self.name,
            candidate_id=candidate.id,
            generation=0,
            payload={"candidate": candidate.model_dump()}
        )
        self.publish(proposal_event)

    def _propose_mutated_candidate(self, event: Event):
        # Applies genome mutations based on the critic's plan
        mutations = event.payload.get("mutations", {})
        parent_id = event.candidate_id
        parent_gen = event.generation
        
        # Mock logic: we would normally load the parent genome and apply changes
        genome = Genome(
            model_family="small_transformer",
            num_layers=mutations.get("num_layers", 4), # e.g. increased layers
            hidden_dimension=mutations.get("hidden_dimension", 128),
            optimizer="adamw",
            learning_rate=mutations.get("learning_rate", 5e-4),
            batch_size=32,
            memory_policy="none",
            retrieval_policy="none",
            curriculum_strategy="hard_examples_first",
            loss_weighting={"ce_loss": 1.0}
        )
        
        candidate = Candidate(
            id=str(uuid.uuid4()),
            parent_ids=[parent_id] if parent_id else [],
            generation=parent_gen + 1 if parent_gen is not None else 1,
            genome=genome,
            status="proposed"
        )
        
        print(f"[{self.name}] Proposed Mutated Candidate {candidate.id[:8]} (Parent: {str(parent_id)[:8]})")
        
        proposal_event = Event(
            event_type=EventType.PROPOSAL,
            sender=self.name,
            candidate_id=candidate.id,
            generation=candidate.generation,
            payload={"candidate": candidate.model_dump()}
        )
        self.publish(proposal_event)
