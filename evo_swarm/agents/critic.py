from __future__ import annotations

import random

from evo_swarm.core.events import Event, EventType
from evo_swarm.core.interfaces.agent import Agent


def _perturb_float(value: float, lo: float, hi: float, scale: float = 0.25) -> float:
    """Perturb a float by up to ±scale fraction, clamped to [lo, hi]."""
    delta = value * scale * random.uniform(-1, 1)
    return round(max(lo, min(hi, value + delta)), 8)


def _perturb_int(value: int, lo: int, hi: int, step: int = 1) -> int:
    """Shift an int by ±step, clamped to [lo, hi]."""
    return max(lo, min(hi, value + random.choice([-step, 0, step])))


OPTIMIZER_POOL = ["adamw", "adam", "sgd", "rmsprop"]
CURRICULUM_POOL = ["random", "hard_examples_first", "easy_to_hard", "mixed", "interleaved"]
MEMORY_POOL = ["none", "recent_failures", "top_k_exemplars"]


class CriticMutatorAgent(Agent):
    """
    Analyzes failure modes and suggests mutations.
    
    IMPORTANT: This agent emits FAILURE_REPORT (not MUTATION_PLAN).
    The GenerationManager is the sole authority that decides when to breed
    and emits MUTATION_PLAN after a full generation has been evaluated.
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
        metrics = event.payload.get("metrics", {})
        candidate_data = event.payload.get("candidate", {})
        parent_genome = candidate_data.get("genome", {}) if candidate_data else {}

        if fitness >= self.target_fitness:
            print(f"[{self.name}] 🏆 Candidate {str(candidate_id)[:8]} is a CHAMPION! (Fitness: {fitness:.4f})")
            champ_event = Event(
                event_type=EventType.CHAMPION_SELECTED,
                sender=self.name,
                candidate_id=candidate_id,
                generation=event.generation,
                payload=event.payload,
            )
            self.publish(champ_event)
        else:
            # Emit FAILURE_REPORT with analysis — do NOT emit MUTATION_PLAN
            suggested_mutations = self._build_mutation_plan(parent_genome, fitness, metrics)
            reasoning = self._build_reasoning(fitness, metrics, suggested_mutations)

            print(f"[{self.name}] {str(candidate_id)[:8]} below target "
                  f"(fitness={fitness:.4f}). Analysis: {reasoning}")

            self.publish(Event(
                event_type=EventType.FAILURE_REPORT,
                sender=self.name,
                candidate_id=candidate_id,
                generation=event.generation,
                payload={
                    "suggested_mutations": suggested_mutations,
                    "parent_fitness": fitness,
                    "parent_genome": parent_genome,
                    "reasoning": reasoning,
                },
            ))

    def _build_mutation_plan(self, parent_genome: dict, fitness: float, metrics: dict) -> dict:
        """Generate genome-aware mutations based on parent genome and performance."""
        mutations: dict = {}

        cur_layers = int(parent_genome.get("num_layers", 2))
        cur_hidden = int(parent_genome.get("hidden_dimension", 64))
        cur_lr = float(parent_genome.get("learning_rate", 1e-3))
        cur_optimizer = str(parent_genome.get("optimizer", "adamw"))
        cur_curriculum = str(parent_genome.get("curriculum_strategy", "random"))
        cur_memory = str(parent_genome.get("memory_policy", "none"))

        # Always perturb core hyperparams
        mutations["num_layers"] = _perturb_int(cur_layers, lo=1, hi=12, step=random.choice([1, 2]))
        mutations["hidden_dimension"] = _perturb_int(cur_hidden, lo=32, hi=512, step=random.choice([16, 32, 64]))
        mutations["learning_rate"] = _perturb_float(cur_lr, lo=1e-5, hi=1e-1, scale=0.3)
        mutations["batch_size"] = random.choice([8, 16, 32, 64])

        if random.random() < 0.3:
            mutations["optimizer"] = random.choice([o for o in OPTIMIZER_POOL if o != cur_optimizer] or [cur_optimizer])
        if random.random() < 0.4:
            mutations["curriculum_strategy"] = random.choice(
                [c for c in CURRICULUM_POOL if c != cur_curriculum] or [cur_curriculum]
            )
        if random.random() < 0.2:
            mutations["memory_policy"] = random.choice(
                [m for m in MEMORY_POOL if m != cur_memory] or [cur_memory]
            )

        # Fitness-dependent aggression
        if fitness < 0.4:
            mutations["num_layers"] = _perturb_int(cur_layers, lo=1, hi=12, step=random.choice([2, 3]))
            mutations["hidden_dimension"] = _perturb_int(cur_hidden, lo=32, hi=512, step=random.choice([64, 128]))
            mutations["learning_rate"] = _perturb_float(cur_lr, lo=1e-5, hi=1e-1, scale=0.5)

        return mutations

    def _build_reasoning(self, fitness: float, metrics: dict, mutations: dict) -> str:
        parts = []
        if fitness < 0.4:
            parts.append("Very low fitness → aggressive mutations")
        elif fitness < 0.7:
            parts.append("Moderate fitness → targeted perturbations")
        else:
            parts.append("Near-target fitness → fine-tuning")
        parts.append(f"Mutated: {', '.join(mutations.keys())}")
        return ". ".join(parts)
