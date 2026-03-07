from abc import ABC, abstractmethod
from typing import List, Optional
from evo_swarm.core.genomes import Candidate

class Registry(ABC):
    """
    Storage abstraction for Candidates, Lineage, and Metrics.
    """
    @abstractmethod
    def save_candidate(self, candidate: Candidate):
        pass

    @abstractmethod
    def get_candidate(self, candidate_id: str) -> Optional[Candidate]:
        pass

    @abstractmethod
    def get_generation(self, generation_id: int) -> List[Candidate]:
        pass

    @abstractmethod
    def get_best_candidates(self, limit: int = 5) -> List[Candidate]:
        pass
