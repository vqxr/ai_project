from __future__ import annotations

from pydantic import BaseModel, Field


class OfflineSwarmConfig(BaseModel):
    # Where ingested papers and indexes live.
    db_path: str = Field(default="offline_swarm.db")

    # Safety guard for shell execution: only allow these command prefixes.
    # Start narrow; expand deliberately.
    allowed_cmd_prefixes: list[str] = Field(
        default_factory=lambda: ["python", "python3", "./venv/bin/python", "rg", "ls", "cat"]
    )

    # Chunking controls for ingestion.
    chunk_chars: int = 3500
    chunk_overlap_chars: int = 300

    # Auto-training: run a fine-tune every N logged assistant replies (0 disables).
    auto_train_every: int = 0

    # Training output location for adapters/checkpoints.
    train_out_dir: str = Field(default="offline_training_out")
