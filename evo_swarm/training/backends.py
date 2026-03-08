from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrainOutcome:
    ok: bool
    metrics: Dict[str, Any]
    artifacts: Dict[str, str]
    message: str = ""


class TrainBackend:
    name: str

    def train(self, *, candidate_id: str, genome: Dict[str, Any], generation: int) -> TrainOutcome:  # pragma: no cover
        raise NotImplementedError


class MockTrainBackend(TrainBackend):
    name = "mock"

    def train(self, *, candidate_id: str, genome: Dict[str, Any], generation: int) -> TrainOutcome:
        # Keep deterministic-ish behavior out of the core loop for now; the evaluator can add noise.
        # This backend is intentionally fast and dependency-free.
        lr = float(genome.get("learning_rate", 1e-3))
        layers = int(genome.get("num_layers", 2))
        hidden = int(genome.get("hidden_dimension", 64))
        synthetic_loss = max(0.05, min(2.0, (1.2 / max(1, layers)) + (256 / max(32, hidden)) * 0.05 + (1e-3 / max(lr, 1e-6)) * 0.02))
        return TrainOutcome(ok=True, metrics={"train_loss": float(synthetic_loss)}, artifacts={}, message="mock training")


class LocalLLMTrainBackend(TrainBackend):
    """
    Runs the small-from-scratch trainer in `ai/local_llm/scripts/train.py` as an external process.

    This is best-effort: if deps/data aren't present, it returns `ok=False` and the swarm can fall
    back to mock metrics (or treat as failure) without crashing the scheduler loop.
    """

    name = "local_llm"

    def __init__(
        self,
        *,
        repo_root: str,
        data_dir: str,
        runs_dir: str,
        max_steps: int = 200,
        eval_every: int = 100,
        save_every: int = 200,
    ):
        self.repo_root = Path(repo_root)
        self.data_dir = Path(data_dir)
        self.runs_dir = Path(runs_dir)
        self.max_steps = int(max_steps)
        self.eval_every = int(eval_every)
        self.save_every = int(save_every)

    def _read_last_val_loss(self, metrics_path: Path) -> Optional[float]:
        if not metrics_path.exists():
            return None
        last = None
        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "val" in obj:
                    last = obj.get("val")
        try:
            return float(last) if last is not None else None
        except (TypeError, ValueError):
            return None

    def train(self, *, candidate_id: str, genome: Dict[str, Any], generation: int) -> TrainOutcome:
        train_script = self.repo_root / "ai" / "local_llm" / "scripts" / "train.py"
        if not train_script.exists():
            return TrainOutcome(ok=False, metrics={}, artifacts={}, message=f"missing train script: {train_script}")

        # Map genome -> local_llm hyperparameters.
        n_layer = int(genome.get("num_layers", 2))
        n_embd = int(genome.get("hidden_dimension", 64))
        n_head = 8
        if n_embd % n_head != 0:
            n_embd = ((n_embd // n_head) + 1) * n_head

        lr = float(genome.get("learning_rate", 3e-4))

        run_dir = self.runs_dir / f"gen{generation}" / candidate_id[:8]
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(train_script),
            "--data_dir",
            str(self.data_dir),
            "--out_dir",
            str(run_dir),
            "--max_steps",
            str(self.max_steps),
            "--eval_every",
            str(self.eval_every),
            "--save_every",
            str(self.save_every),
            "--n_layer",
            str(n_layer),
            "--n_head",
            str(n_head),
            "--n_embd",
            str(n_embd),
            "--lr",
            str(lr),
        ]

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except Exception as e:  # noqa: BLE001
            return TrainOutcome(ok=False, metrics={}, artifacts={}, message=f"failed to spawn trainer: {e}")

        val_loss = self._read_last_val_loss(run_dir / "metrics.jsonl")
        metrics: Dict[str, Any] = {
            "backend": self.name,
            "seconds": time.time() - t0,
            "val_loss": val_loss,
        }
        artifacts = {
            "run_dir": str(run_dir),
            "stdout": str(run_dir / "stdout.txt"),
        }
        # Always persist stdout for debugging.
        (run_dir / "stdout.txt").write_text(proc.stdout or "", encoding="utf-8")

        if proc.returncode != 0:
            return TrainOutcome(
                ok=False,
                metrics=metrics | {"returncode": proc.returncode},
                artifacts=artifacts,
                message="local_llm training failed (see stdout.txt)",
            )

        if val_loss is None:
            # Training may have completed without an eval step; treat as soft-failure.
            return TrainOutcome(
                ok=False,
                metrics=metrics,
                artifacts=artifacts,
                message="local_llm training finished but produced no val_loss (metrics.jsonl missing/empty)",
            )

        return TrainOutcome(
            ok=True,
            metrics=metrics,
            artifacts=artifacts,
            message="local_llm training ok",
        )


def build_train_backend(*, repo_root: str) -> TrainBackend:
    """
    Select a training backend via env var.

    - `EVO_SWARM_TRAIN_BACKEND=mock` (default)
    - `EVO_SWARM_TRAIN_BACKEND=local_llm`
    """

    backend = os.getenv("EVO_SWARM_TRAIN_BACKEND", "mock").strip().lower()
    if backend in {"mock", "none", ""}:
        return MockTrainBackend()
    if backend == "local_llm":
        data_dir = os.getenv("LOCAL_LLM_DATA_DIR", "ai/local_llm/data/processed")
        runs_dir = os.getenv("LOCAL_LLM_RUNS_DIR", "ai/local_llm/runs/evo_swarm")
        max_steps = int(os.getenv("LOCAL_LLM_MAX_STEPS", "200"))
        eval_every = int(os.getenv("LOCAL_LLM_EVAL_EVERY", "100"))
        save_every = int(os.getenv("LOCAL_LLM_SAVE_EVERY", str(max_steps)))
        return LocalLLMTrainBackend(
            repo_root=repo_root,
            data_dir=str(Path(repo_root) / data_dir),
            runs_dir=str(Path(repo_root) / runs_dir),
            max_steps=max_steps,
            eval_every=eval_every,
            save_every=save_every,
        )
    raise ValueError(f"Unknown EVO_SWARM_TRAIN_BACKEND={backend!r}")

