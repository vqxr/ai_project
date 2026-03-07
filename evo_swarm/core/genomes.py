from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Genome(BaseModel):
    """
    A structured description of a candidate system.
    The genome represents what can evolve.
    """
    model_family: str
    num_layers: int
    hidden_dimension: int
    optimizer: str
    learning_rate: float
    batch_size: int
    memory_policy: str
    retrieval_policy: str
    curriculum_strategy: str
    loss_weighting: Dict[str, float]
    routing_graph: Optional[Dict[str, Any]] = None

class Candidate(BaseModel):
    """
    An instantiated genome plus its metrics, artifacts, lineage, and run history.
    """
    id: str
    parent_ids: List[str]
    generation: int
    genome: Genome
    fitness_score: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    run_history: List[str] = Field(default_factory=list)
    artifacts: Dict[str, str] = Field(default_factory=dict)
    status: str = "proposed" # proposed, training, evaluating, completed, failed
