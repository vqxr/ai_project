from abc import ABC, abstractmethod

from evo_swarm.core.genomes import Candidate


class Registry(ABC):
    """
    Storage abstraction for Candidates, Lineage, and Metrics.
    """
    @abstractmethod
    def save_candidate(self, candidate: Candidate):
        pass

    @abstractmethod
    def get_candidate(self, candidate_id: str) -> Candidate | None:
        pass

    @abstractmethod
    def get_generation(self, generation_id: int) -> list[Candidate]:
        pass

    @abstractmethod
    def get_best_candidates(self, limit: int = 5) -> list[Candidate]:
        pass

    @abstractmethod
    def get_lineage_tree(self, candidate_id: str) -> list[Candidate]:
        """Return the full ancestry chain from the given candidate back to the root."""
        pass

