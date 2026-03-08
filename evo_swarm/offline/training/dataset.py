from __future__ import annotations

import json
from dataclasses import asdict

from evo_swarm.offline.training.store import Interaction


def interactions_to_jsonl(interactions: list[Interaction]) -> str:
    """
    Produces a simple instruction-tuning JSONL format:
    {"instruction": ..., "input": ..., "output": ...}

    - instruction: the user request
    - input: retrieved context (paper chunks etc.)
    - output: the assistant response
    """
    lines: list[str] = []
    for i in interactions:
        obj = {
            "instruction": i.user_text,
            "input": i.retrieved_context,
            "output": i.assistant_text,
        }
        lines.append(json.dumps(obj, ensure_ascii=True))
    return "\n".join(lines) + ("\n" if lines else "")

