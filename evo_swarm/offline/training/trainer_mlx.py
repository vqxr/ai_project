from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainResult:
    ok: bool
    message: str


def train_with_mlx(*, dataset_jsonl_path: str, out_dir: str) -> TrainResult:
    """
    Stub hook for Apple MLX fine-tuning.

    This repo intentionally avoids pulling heavy deps automatically. The goal is:
    - capture and version the dataset locally
    - provide a single place to integrate real MLX LoRA training when you’re ready
    """
    if not os.path.exists(dataset_jsonl_path):
        return TrainResult(ok=False, message=f"dataset not found: {dataset_jsonl_path}")

    os.makedirs(out_dir, exist_ok=True)
    return TrainResult(
        ok=False,
        message=(
            "MLX training not wired yet.\n"
            "Next: add MLX deps and run a LoRA fine-tune using this JSONL dataset.\n"
            f"Dataset: {dataset_jsonl_path}\n"
            f"Out dir: {out_dir}"
        ),
    )

