# Evo Swarm — Distributed Evolutionary Swarm Intelligence Framework

A distributed, modular evolutionary framework where specialist agents collaboratively generate, train, evaluate, mutate, and refine candidate learning systems represented as genomes.

## Quickstart

```bash
# 1. Create venv and install deps
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run the evolution loop (mock training, math benchmark)
python main.py

# 3. Inspect the SQLite registry
python scripts/inspect_registry.py evo_swarm.db
```

## PDF Ingestion

```bash
# Batch-convert PDFs to text
python scripts/pdf_to_txt.py --input papers/ --output papers_txt/

# Or ingest directly (PDFs handled automatically)
python -m evo_swarm.offline ingest papers/
```

## Directory Structure

```
evo_swarm/
  core/              # Interfaces, events, registry, scheduler
  agents/            # Architect, Trainer, Evaluator, Curator, Critic
  evolution/         # Generation manager, mutation, crossover
  training/          # Pluggable training backends (mock, local_llm)
  offline/           # Offline swarm with LLM roles, knowledge store, tools
  benchmarks/        # Domain-specific evaluation (math, neuro, philosophy)
  models/            # Pluggable model implementations
  memory/            # Fast, episodic, knowledge memory
  tracking/          # Experiments, metrics, lineage
  infra/             # Local, distributed, edge compute
  api/               # REST API (future)
  dashboard/         # Web dashboard (future)
ai/local_llm/        # From-scratch GPT trainer (PyTorch)
scripts/             # Utility scripts (PDF converter, registry inspector)
```

## Architecture

The system uses an **event-driven** architecture where specialist agents communicate through typed events:

1. **Curator** prepares datasets and kicks off generations
2. **Architect** proposes candidate genome configurations
3. **Trainer** trains candidates (mock or real backends)
4. **Evaluator** benchmarks candidates and computes fitness
5. **Critic/Mutator** analyzes failures and proposes mutations
6. **GenerationManager** tracks lineage in SQLite

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `EVO_SWARM_TRAIN_BACKEND` | `mock` | Training backend (`mock`, `local_llm`) |
| `LOCAL_LLM_DATA_DIR` | `ai/local_llm/data/processed` | Data dir for local_llm backend |
| `LOCAL_LLM_MAX_STEPS` | `200` | Max training steps per candidate |
