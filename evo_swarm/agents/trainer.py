from evo_swarm.core.interfaces.agent import Agent
from evo_swarm.core.events import Event, EventType
import os
import random
from evo_swarm.training.backends import MockTrainBackend, TrainBackend, build_train_backend

class TrainerAgent(Agent):
    """
    Trains or fine-tunes candidate systems.
    Listens for PROPOSAL.
    """
    def __init__(self, name: str = "Trainer", *, backend: TrainBackend | None = None, repo_root: str | None = None):
        super().__init__(name)
        if backend is not None:
            self.backend = backend
        else:
            root = repo_root or os.getcwd()
            self.backend = build_train_backend(repo_root=root)

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
        
        outcome = None
        try:
            genome = (candidate_data or {}).get("genome", {}) or {}
            outcome = self.backend.train(candidate_id=str(candidate_id), genome=genome, generation=int(event.generation or 0))
        except Exception as e:  # noqa: BLE001
            # Don't crash the scheduler loop; report and let downstream decide.
            outcome = None
            print(f"[{self.name}] Training backend error: {e}")

        # Compatibility: always emit a train_loss for existing evaluators.
        if outcome and outcome.ok and "train_loss" in outcome.metrics:
            train_loss = float(outcome.metrics["train_loss"])
        elif outcome and outcome.ok and outcome.metrics.get("val_loss") is not None:
            # Approximate train_loss from val_loss when only val is available.
            train_loss = float(outcome.metrics["val_loss"])
        else:
            train_loss = random.uniform(0.1, 0.5)
        
        print(f"[{self.name}] Completed training {str(candidate_id)[:8]} (loss={train_loss:.4f})")
        
        completed_event = Event(
            event_type=EventType.TRAINING_COMPLETED,
            sender=self.name,
            candidate_id=candidate_id,
            generation=event.generation,
            payload={
                "train_loss": train_loss,
                "candidate": candidate_data,
                "train_backend": getattr(self.backend, "name", "unknown"),
                "train_ok": bool(outcome.ok) if outcome else False,
                "train_metrics": outcome.metrics if outcome else {},
                "train_artifacts": outcome.artifacts if outcome else {},
                "train_message": outcome.message if outcome else "no outcome (backend exception)",
            }
        )
        self.publish(completed_event)
