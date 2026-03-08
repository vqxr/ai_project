# Evo Swarm — Distributed Evolutionary Swarm Intelligence Framework

A distributed, modular evolutionary framework where specialist agents collaboratively generate, train, evaluate, mutate, and refine candidate learning systems represented as genomes.

## Quickstart

```bash
# 1. Create venv and install deps
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run the evolution loop (mock training, math benchmark)
python3 main.py

# 3. Inspect the SQLite registry
python3 scripts/inspect_registry.py evo_swarm.db
```

## PDF Ingestion

```bash
# Batch-convert PDFs to text
python3 scripts/pdf_to_txt.py --input papers/ --output papers_txt/

# Or ingest directly (PDFs handled automatically)
python3 -m evo_swarm.offline.cli ingest papers/
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
| `LOCAL_LLM_EVAL_EVERY` | `100` | Eval interval (steps) for local_llm backend |
| `LOCAL_LLM_SAVE_EVERY` | `LOCAL_LLM_MAX_STEPS` | Checkpoint interval (steps) for local_llm backend |
| `LOCAL_LLM_RUNS_DIR` | `ai/local_llm/runs/evo_swarm` | Output runs dir for local_llm backend |

## Using the Local LLM Trainer (Optional)

The default swarm trainer is `mock` (fast and dependency-free). To train a small GPT-from-scratch model
inside the swarm loop, wire the `ai/local_llm` trainer as the backend.

1) Install local trainer deps:

```bash
python3 -m pip install -r ai/local_llm/requirements.txt
```

2) Put your text data in:

```text
ai/local_llm/data/text/**/*.txt
```

3) Build tokenizer + token dataset:

```bash
python3 ai/local_llm/scripts/prepare_data.py \
  --text_dir ai/local_llm/data/text \
  --out_dir ai/local_llm/data/processed \
  --vocab_size 16000
```

4) Run the swarm with `local_llm` training:

```bash
EVO_SWARM_TRAIN_BACKEND=local_llm python3 main.py
```

Artifacts are written under `ai/local_llm/runs/evo_swarm/` (per generation/candidate).

## Offline Swarm CLI (RAG + Logging)

The `evo_swarm.offline` flow is designed for offline “ingest → retrieve → plan/critic → log for later fine-tuning”.

```bash
# Ingest a folder of .txt/.md papers
python3 -m evo_swarm.offline.cli ingest papers_txt/

# Ask a question grounded in ingested notes
python3 -m evo_swarm.offline.cli ask "Summarize the key findings about X"

# Interactive chat; optionally auto-train every N replies (training is a stub by default)
python3 -m evo_swarm.offline.cli chat --auto-train-every 20 --train-out offline_training_out
```

## Scientific Datasets

If you’re browsing scientific datasets/LLM resources (e.g. the awesome list
[InternScience/Awesome-Scientific-Datasets-and-LLMs](https://github.com/InternScience/Awesome-Scientific-Datasets-and-LLMs)),
use it to pick sources that you can legally download and store locally, then:

- For RAG-style Q&A: ingest extracted `.txt`/`.md` into the offline CLI.
- For training `ai/local_llm`: convert the corpus into `.txt` files under `ai/local_llm/data/text/`, then run `prepare_data.py`.

## Dev Checks (Optional)

If your editor shows “red lines” from lint/type diagnostics, these two commands usually match what’s happening:

```bash
ruff check . --fix
pyright
```

Note: `pyright` will report missing imports for `ai/local_llm/scripts/*` unless you install `ai/local_llm/requirements.txt`
into the same Python environment your editor uses.
