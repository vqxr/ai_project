from __future__ import annotations

from evo_swarm.benchmarks.math.benchmark import score_genome
from evo_swarm.core.events import Event, EventType
from evo_swarm.core.interfaces.agent import Agent


class EvaluatorAgent(Agent):
    """
    Benchmarks candidates using the math benchmark.
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
        val_loss = (event.payload.get("train_metrics") or {}).get("val_loss")
        candidate_id = event.candidate_id
        genome = candidate_data.get("genome", {}) if candidate_data else {}

        print(f"[{self.name}] Evaluating candidate {str(candidate_id)[:8]}...")

        self.publish(Event(
            event_type=EventType.EVALUATION_STARTED,
            sender=self.name,
            candidate_id=candidate_id,
            generation=event.generation,
        ))

        # Run the deterministic math benchmark on the genome
        bench_scores = score_genome(genome)

        # Combine benchmark score with training signal
        bench_fitness = bench_scores["overall_fitness"]
        
        # If we have real training metrics, blend them in
        loss_for_scoring = train_loss
        try:
            if val_loss is not None:
                loss_for_scoring = float(val_loss)
        except (TypeError, ValueError):
            pass
        
        # Training bonus: lower loss -> higher fitness contribution
        training_signal = max(0.0, 1.0 - float(loss_for_scoring))
        
        # Final fitness: 70% benchmark, 30% training signal
        fitness = 0.70 * bench_fitness + 0.30 * training_signal

        metrics = {
            **bench_scores,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "training_signal": round(training_signal, 4),
            "fitness": round(fitness, 4),
        }

        print(f"[{self.name}] {str(candidate_id)[:8]}: fitness={fitness:.4f} "
              f"(bench={bench_fitness:.4f} train_sig={training_signal:.4f})")

        self.publish(Event(
            event_type=EventType.EVALUATION_RESULT,
            sender=self.name,
            candidate_id=candidate_id,
            generation=event.generation,
            payload={
                "fitness": fitness,
                "metrics": metrics,
                "candidate": candidate_data,
            },
        ))
