"""
Synthetic math benchmark for scoring candidate genomes.

This doesn't run actual math — it provides a *deterministic fitness signal*
based on genome properties so that evolution has a meaningful landscape to search.

The benchmark rewards:
- More layers (diminishing returns past 6)
- Moderate hidden dims (sweet spot around 128-256)
- Lower learning rates (sweet spot around 1e-3 to 5e-4)
- AdamW optimizer (slight bonus)
- Curriculum strategies other than 'random'
- Memory policies other than 'none'
"""
from __future__ import annotations

import math
from typing import Any


def score_genome(genome: dict[str, Any]) -> dict[str, float]:
    """
    Score a genome on a synthetic math benchmark.
    
    Returns a dict with:
      - math_accuracy: 0-1 score
      - efficiency: 0-1 score
      - reasoning_quality: 0-1 score
      - overall_fitness: weighted combination
    """
    layers = int(genome.get("num_layers", 2))
    hidden = int(genome.get("hidden_dimension", 64))
    lr = float(genome.get("learning_rate", 1e-3))
    optimizer = str(genome.get("optimizer", "adamw"))
    curriculum = str(genome.get("curriculum_strategy", "random"))
    memory = str(genome.get("memory_policy", "none"))
    batch_size = int(genome.get("batch_size", 16))

    # --- Math Accuracy ---
    # Layers: diminishing returns, peaks around 6-8
    layer_score = 1.0 - math.exp(-0.4 * layers)  # 2->0.55, 4->0.80, 6->0.91, 8->0.96
    
    # Hidden dim: bell curve centered at 192
    hidden_score = math.exp(-((hidden - 192) ** 2) / (2 * 100 ** 2))
    
    # Learning rate: bell curve in log space centered at 5e-4
    lr_log = math.log10(max(lr, 1e-8))
    lr_center = math.log10(5e-4)
    lr_score = math.exp(-((lr_log - lr_center) ** 2) / (2 * 0.8 ** 2))
    
    math_accuracy = 0.4 * layer_score + 0.35 * hidden_score + 0.25 * lr_score

    # --- Efficiency ---
    # Smaller models are more efficient
    param_proxy = layers * hidden * hidden  # rough proxy for param count
    efficiency = 1.0 / (1.0 + param_proxy / 500_000)
    
    # Batch size bonus for efficiency (larger = more efficient)
    efficiency += 0.05 * min(batch_size / 64, 1.0)

    # --- Reasoning Quality ---
    reasoning = 0.3  # baseline
    
    # Optimizer bonus
    if optimizer == "adamw":
        reasoning += 0.15
    elif optimizer == "adam":
        reasoning += 0.10
    elif optimizer == "sgd":
        reasoning += 0.05
    
    # Curriculum bonus
    if curriculum != "random":
        reasoning += 0.15
    if curriculum == "hard_examples_first":
        reasoning += 0.10
    
    # Memory bonus
    if memory != "none":
        reasoning += 0.10
    
    reasoning = min(1.0, reasoning)

    # --- Overall Fitness ---
    overall = (
        0.40 * math_accuracy
        + 0.20 * min(1.0, efficiency)
        + 0.15 * reasoning
        + 0.10 * layer_score  # extra weight for depth
        + 0.10 * lr_score     # extra weight for good lr
        + 0.05 * hidden_score
    )

    return {
        "math_accuracy": round(math_accuracy, 4),
        "efficiency": round(min(1.0, efficiency), 4),
        "reasoning_quality": round(reasoning, 4),
        "overall_fitness": round(overall, 4),
        "layer_score": round(layer_score, 4),
        "hidden_score": round(hidden_score, 4),
        "lr_score": round(lr_score, 4),
    }
