import json
import os
from typing import List, Optional
from evo_swarm.core.genomes import Candidate
from evo_swarm.core.registry.registry import Registry

class LocalRegistry(Registry):
    """
    A simple JSON-file-based implementation of the Registry for V1.
    """
    def __init__(self, storage_dir: str = "local_registry"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        
    def _candidate_path(self, candidate_id: str) -> str:
        return os.path.join(self.storage_dir, f"{candidate_id}.json")

    def save_candidate(self, candidate: Candidate):
        path = self._candidate_path(candidate.id)
        with open(path, 'w') as f:
            f.write(candidate.model_dump_json(indent=2))

    def get_candidate(self, candidate_id: str) -> Optional[Candidate]:
        path = self._candidate_path(candidate_id)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            data = json.load(f)
            return Candidate(**data)

    def get_generation(self, generation_id: int) -> List[Candidate]:
        candidates = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.storage_dir, filename), 'r') as f:
                    data = json.load(f)
                    if data.get("generation") == generation_id:
                        candidates.append(Candidate(**data))
        return candidates

    def get_best_candidates(self, limit: int = 5) -> List[Candidate]:
        all_candidates = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.storage_dir, filename), 'r') as f:
                    data = json.load(f)
                    # Exclude un-evaluated candidates based on status
                    if data.get("status") in ("completed", "evaluated") and data.get("fitness_score") is not None:
                        all_candidates.append(Candidate(**data))
        
        # Sort by fitness score descending
        all_candidates.sort(key=lambda c: c.fitness_score, reverse=True)
        return all_candidates[:limit]

    def get_lineage_tree(self, candidate_id: str) -> List[Candidate]:
        """Not implemented for the JSON-file registry."""
        return []

