from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import time
import uuid

class Event(BaseModel):
    """
    Base communication event in the swarm.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    sender: str
    receiver: Optional[str] = None # If None, it's a broadcast
    timestamp: float = Field(default_factory=time.time)
    
    # Event payload
    candidate_id: Optional[str] = None
    generation: Optional[int] = None
    payload: Dict[str, Any] = Field(default_factory=dict)

# Helpful constants for event types
class EventType:
    PROPOSAL = "proposal"
    MUTATION_PLAN = "mutation_plan"
    TRAINING_STARTED = "training_started"
    TRAINING_COMPLETED = "training_completed"
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_RESULT = "evaluation_result"
    FAILURE_REPORT = "failure_report"
    DATASET_UPDATE = "dataset_update"
    CHAMPION_SELECTED = "champion_selected"
    LINEAGE_UPDATE = "lineage_update"
    BENCHMARK_SUMMARY = "benchmark_summary"
