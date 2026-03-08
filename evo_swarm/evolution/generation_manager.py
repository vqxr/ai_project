from __future__ import annotations

import random

from evo_swarm.core.events import Event, EventType
from evo_swarm.core.genomes import Candidate
from evo_swarm.core.interfaces.agent import Agent
from evo_swarm.core.registry.registry import Registry


def _crossover_genomes(genome_a: dict, genome_b: dict) -> dict:
    """Single-point crossover: for each key, randomly pick from parent A or B."""
    child: dict = {}
    all_keys = set(genome_a.keys()) | set(genome_b.keys())
    for key in all_keys:
        if random.random() < 0.5 and key in genome_a:
            child[key] = genome_a[key]
        elif key in genome_b:
            child[key] = genome_b[key]
        elif key in genome_a:
            child[key] = genome_a[key]
    return child


class GenerationManager(Agent):
    """
    Sole controller of the evolution loop.
    
    Flow:
    1. Tracks proposals and evaluations per generation
    2. When all candidates in a generation are evaluated:
       - Picks top-K parents
       - Emits MUTATION_PLAN (with crossover when 2+ parents)
       - Enforces max_generations
    3. The CriticMutator only emits FAILURE_REPORT (analysis), not breeding commands
    """
    def __init__(
        self,
        registry: Registry,
        population_size: int = 3,
        top_k: int = 2,
        max_generations: int = 5,
        target_fitness: float = 0.90,
        crossover_rate: float = 0.3,
        name: str = "GenerationManager",
    ):
        super().__init__(name)
        self.registry = registry
        self.population_size = population_size
        self.top_k = top_k
        self.max_generations = max_generations
        self.target_fitness = target_fitness
        self.crossover_rate = crossover_rate

        self.current_generation = 0
        # Per-generation tracking
        self._gen_proposed: dict[int, int] = {}   # gen -> count of proposals
        self._gen_evaluated: dict[int, int] = {}   # gen -> count of evaluations
        self._gen_advanced: set = set()             # generations that already triggered breeding
        self._champion_found = False
        # Collect failure reports for mutation suggestions
        self._failure_reports: dict[int, list[dict]] = {}  # gen -> list of {candidate_id, suggested_mutations, ...}

    def handle_event(self, event: Event):
        if self._champion_found:
            return  # Stop processing after champion
        if event.event_type == EventType.PROPOSAL:
            self._on_proposal(event)
        elif event.event_type == EventType.EVALUATION_RESULT:
            self._on_evaluation(event)
        elif event.event_type == EventType.FAILURE_REPORT:
            self._on_failure_report(event)
        elif event.event_type == EventType.CHAMPION_SELECTED:
            self._on_champion(event)

    def _on_proposal(self, event: Event):
        data = event.payload.get("candidate", {})
        try:
            candidate = Candidate(**data)
            self.registry.save_candidate(candidate)
            gen = candidate.generation
            self._gen_proposed[gen] = self._gen_proposed.get(gen, 0) + 1
            print(f"[{self.name}] Registered {candidate.id[:8]} "
                  f"(Gen {gen}, {self._gen_proposed[gen]} proposed)")
        except Exception as e:
            print(f"[{self.name}] Error registering candidate: {e}")

    def _on_evaluation(self, event: Event):
        candidate_id = event.candidate_id
        if not candidate_id:
            return
        candidate = self.registry.get_candidate(candidate_id)
        if not candidate:
            return
        candidate.status = "evaluated"
        candidate.fitness_score = event.payload.get("fitness")
        candidate.metrics = event.payload.get("metrics", {})
        self.registry.save_candidate(candidate)

        gen = candidate.generation
        self._gen_evaluated[gen] = self._gen_evaluated.get(gen, 0) + 1
        proposed = self._gen_proposed.get(gen, 0)
        evaluated = self._gen_evaluated[gen]

        print(f"[{self.name}] Evaluated {candidate_id[:8]}: "
              f"fitness={candidate.fitness_score:.4f} (Gen {gen}: {evaluated}/{proposed})")

        # Check if generation is complete
        if evaluated >= proposed and gen not in self._gen_advanced:
            self._on_generation_complete(gen)

    def _on_failure_report(self, event: Event):
        """Collect failure analysis from the Critic for later use during breeding."""
        gen = event.generation or 0
        if gen not in self._failure_reports:
            self._failure_reports[gen] = []
        self._failure_reports[gen].append({
            "candidate_id": event.candidate_id,
            "suggested_mutations": event.payload.get("suggested_mutations", {}),
            "parent_genome": event.payload.get("parent_genome", {}),
            "parent_fitness": event.payload.get("parent_fitness", 0),
        })

    def _on_champion(self, event: Event):
        self._champion_found = True
        print(f"\n[{self.name}] 🏆 EVOLUTION COMPLETE — Champion found!")
        self.publish(Event(event_type="SYSTEM_HALT", sender=self.name))

    def _on_generation_complete(self, gen: int):
        """All candidates in generation `gen` have been evaluated. Breed the next generation."""
        self._gen_advanced.add(gen)

        # Get all evaluated candidates for this generation
        gen_candidates = self.registry.get_generation(gen)
        evaluated = [c for c in gen_candidates if c.status == "evaluated" and c.fitness_score is not None]
        evaluated.sort(key=lambda c: c.fitness_score or 0, reverse=True)

        if not evaluated:
            print(f"[{self.name}] No evaluated candidates for Gen {gen}")
            return

        best = evaluated[0]
        print(f"\n{'='*50}")
        print(f"[{self.name}] Generation {gen} complete!")
        print(f"  Population: {len(evaluated)} candidates")
        print(f"  Best:       {best.id[:8]} (fitness={best.fitness_score:.4f})")
        print(f"  Worst:      {evaluated[-1].id[:8]} (fitness={evaluated[-1].fitness_score:.4f})")
        print(f"{'='*50}\n")

        # Check max generations
        next_gen = gen + 1
        if next_gen >= self.max_generations:
            print(f"[{self.name}] Reached max_generations={self.max_generations}. Halting.")
            self.publish(Event(event_type="SYSTEM_HALT", sender=self.name))
            return

        # Select top-K parents
        parents = evaluated[:self.top_k]
        self.current_generation = next_gen

        # Build children via mutation and crossover
        children_to_create = self.population_size
        created = 0

        for i in range(children_to_create):
            parent = parents[i % len(parents)]
            parent_genome = parent.genome.model_dump()

            # Decide: crossover or mutation?
            if len(parents) >= 2 and random.random() < self.crossover_rate:
                # CROSSOVER: pick two parents, combine genomes, then apply mutations
                other = random.choice([p for p in parents if p.id != parent.id] or parents)
                base_genome = _crossover_genomes(parent_genome, other.genome.model_dump())
                parent_ids = [parent.id, other.id]
                method = "crossover"
            else:
                # MUTATION: use failure report suggestions if available
                base_genome = parent_genome
                parent_ids = [parent.id]
                method = "mutation"

            # Get mutation suggestions from the Critic's failure report for this parent
            mutations = {}
            reports = self._failure_reports.get(gen, [])
            matching = [r for r in reports if r["candidate_id"] == parent.id]
            if matching:
                mutations = matching[0].get("suggested_mutations", {})
            else:
                # Use any available report's mutations as fallback
                if reports:
                    mutations = reports[0].get("suggested_mutations", {})

            print(f"[{self.name}] Breeding child {created+1}/{children_to_create} via {method} "
                  f"(parent: {parent.id[:8]})")

            self.publish(Event(
                event_type=EventType.MUTATION_PLAN,
                sender=self.name,
                candidate_id=parent.id,
                generation=gen,  # parent's generation — architect adds +1
                payload={
                    "mutations": mutations,
                    "parent_genome": base_genome,
                    "parent_ids": parent_ids,
                    "method": method,
                },
            ))
            created += 1
