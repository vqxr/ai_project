from __future__ import annotations

import random
import uuid
from typing import Any

from evo_swarm.core.events import Event, EventType
from evo_swarm.core.genomes import Candidate, Genome
from evo_swarm.core.interfaces.agent import Agent

# Default values for initial genome fields
_DEFAULT_GENOME_POOL: dict[str, list[Any]] = {
    "model_family": ["small_transformer", "mini_mlp", "tiny_rnn"],
    "num_layers": [2, 3, 4, 6],
    "hidden_dimension": [64, 128, 256],
    "optimizer": ["adamw", "adam", "sgd"],
    "learning_rate": [1e-2, 3e-3, 1e-3, 5e-4, 1e-4],
    "batch_size": [8, 16, 32],
    "memory_policy": ["none"],
    "retrieval_policy": ["none"],
    "curriculum_strategy": ["random", "hard_examples_first", "easy_to_hard"],
}


class ArchitectAgent(Agent):
    """
    Proposes new candidate system designs.
    - On DATASET_UPDATE: proposes N initial candidates (population seeding)
    - On MUTATION_PLAN: creates a mutated child from a parent genome
    """
    def __init__(self, name: str = "Architect", population_size: int = 3):
        super().__init__(name)
        self.population_size = population_size

    def handle_event(self, event: Event):
        if event.event_type == EventType.DATASET_UPDATE:
            self._propose_initial_population(event)
        elif event.event_type == EventType.MUTATION_PLAN:
            self._propose_mutated_candidate(event)

    def _propose_initial_population(self, event: Event):
        """Seed the generation with diverse candidates."""
        generation = event.generation or 0
        for i in range(self.population_size):
            genome = Genome(
                model_family=random.choice(_DEFAULT_GENOME_POOL["model_family"]),
                num_layers=random.choice(_DEFAULT_GENOME_POOL["num_layers"]),
                hidden_dimension=random.choice(_DEFAULT_GENOME_POOL["hidden_dimension"]),
                optimizer=random.choice(_DEFAULT_GENOME_POOL["optimizer"]),
                learning_rate=random.choice(_DEFAULT_GENOME_POOL["learning_rate"]),
                batch_size=random.choice(_DEFAULT_GENOME_POOL["batch_size"]),
                memory_policy=random.choice(_DEFAULT_GENOME_POOL["memory_policy"]),
                retrieval_policy=random.choice(_DEFAULT_GENOME_POOL["retrieval_policy"]),
                curriculum_strategy=random.choice(_DEFAULT_GENOME_POOL["curriculum_strategy"]),
                loss_weighting={"ce_loss": 1.0},
            )

            candidate = Candidate(
                id=str(uuid.uuid4()),
                parent_ids=[],
                generation=generation,
                genome=genome,
                status="proposed",
            )

            print(f"[{self.name}] Proposed Gen {generation} Candidate {i+1}/{self.population_size}: {candidate.id[:8]}")

            self.publish(Event(
                event_type=EventType.PROPOSAL,
                sender=self.name,
                candidate_id=candidate.id,
                generation=generation,
                payload={"candidate": candidate.model_dump()},
            ))

    def _propose_mutated_candidate(self, event: Event):
        """Build a mutated child from the parent genome + mutation plan (or crossover)."""
        mutations = event.payload.get("mutations", {})
        parent_genome = event.payload.get("parent_genome", {})
        parent_ids = event.payload.get("parent_ids", [])
        method = event.payload.get("method", "mutation")
        parent_gen = event.generation or 0

        # Fall back to event candidate_id if no parent_ids provided
        if not parent_ids and event.candidate_id:
            parent_ids = [event.candidate_id]

        # Start from parent genome, apply mutations on top
        genome_dict: dict[str, Any] = {**parent_genome}
        for key, value in mutations.items():
            genome_dict[key] = value

        # Ensure required fields have values
        genome_dict.setdefault("model_family", "small_transformer")
        genome_dict.setdefault("loss_weighting", {"ce_loss": 1.0})
        genome_dict.setdefault("routing_graph", None)

        genome = Genome(**genome_dict)
        candidate = Candidate(
            id=str(uuid.uuid4()),
            parent_ids=parent_ids,
            generation=parent_gen + 1,
            genome=genome,
            status="proposed",
        )

        parent_str = "+".join(p[:8] for p in parent_ids)
        print(f"[{self.name}] Proposed {method} child {candidate.id[:8]} "
              f"(Parents: {parent_str}, Gen {candidate.generation})")

        self.publish(Event(
            event_type=EventType.PROPOSAL,
            sender=self.name,
            candidate_id=candidate.id,
            generation=candidate.generation,
            payload={"candidate": candidate.model_dump()},
        ))
