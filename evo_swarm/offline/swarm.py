from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from rich.console import Console

from evo_swarm.offline.config import OfflineSwarmConfig
from evo_swarm.offline.knowledge.ingest import ingest_path
from evo_swarm.offline.knowledge.store import KnowledgeStore
from evo_swarm.offline.llm.base import EchoLLM, LocalLLM
from evo_swarm.offline.tools import ToolCall, ToolRunner

console = Console()


@dataclass
class SwarmContext:
    config: OfflineSwarmConfig
    store: KnowledgeStore
    llm: LocalLLM
    tools: ToolRunner


class CuratorRole:
    def ingest(self, ctx: SwarmContext, path: str) -> dict:
        return ingest_path(ctx.store, ctx.config, path)


class ArchitectRole:
    SYSTEM = (
        "You are the Architect role in an offline agent swarm. "
        "You produce structured plans with acceptance criteria. "
        "If information is missing, request it explicitly."
    )

    def plan(self, ctx: SwarmContext, goal: str) -> str:
        # Keep this text-only for now. You can later enforce JSON plans.
        prompt = (
            "Goal:\n"
            f"{goal}\n\n"
            "Constraints:\n"
            "- Fully offline\n"
            "- Prefer searching ingested papers before making claims\n"
            "- Use only the available tools: search_papers, read_file, write_file, run_cmd\n\n"
            "Output:\n"
            "- A short plan\n"
            "- A list of tool calls you would run (as bullet points)\n"
            "- Acceptance criteria\n"
        )
        return ctx.llm.generate(system=self.SYSTEM, prompt=prompt)


class ExecutorRole:
    def run_research(self, ctx: SwarmContext, query: str) -> str:
        res = ctx.tools.run(ToolCall(name="search_papers", args={"query": query, "limit": 8}))
        return res.output


class EvaluatorRole:
    def score(self, plan_text: str) -> str:
        # Placeholder: in a real system this would run a benchmark suite.
        if "Acceptance" in plan_text or "criteria" in plan_text.lower():
            return "score=0.6 (has acceptance criteria)"
        return "score=0.2 (missing acceptance criteria)"


class CriticRole:
    SYSTEM = (
        "You are the Critic role. You try to falsify plans and outputs. "
        "You demand citations to ingested papers and you flag unsafe steps."
    )

    def review(self, ctx: SwarmContext, text: str) -> str:
        prompt = (
            "Review the following for missing grounding, unsafe actions, and vague steps.\n\n"
            f"{text}\n"
        )
        return ctx.llm.generate(system=self.SYSTEM, prompt=prompt)


class OfflineSwarm:
    def __init__(self, *, config: OfflineSwarmConfig, llm: LocalLLM, workspace_root: str):
        self.config = config
        self.store = KnowledgeStore(config.db_path)
        self.tools = ToolRunner(config=config, store=self.store, workspace_root=workspace_root)
        self.ctx = SwarmContext(config=config, store=self.store, llm=llm, tools=self.tools)

        self.curator = CuratorRole()
        self.architect = ArchitectRole()
        self.executor = ExecutorRole()
        self.evaluator = EvaluatorRole()
        self.critic = CriticRole()

    def close(self) -> None:
        self.store.close()

    def ingest(self, path: str) -> dict:
        return self.curator.ingest(self.ctx, path)

    def ask(self, question: str) -> str:
        # Baseline offline flow:
        # 1) retrieve context
        # 2) generate a plan (if LLM configured)
        # 3) critique it
        console.print("[bold]Retrieval[/bold]")
        retrieval = self.executor.run_research(self.ctx, question)
        console.print(retrieval)

        console.print("\n[bold]Architect Plan[/bold]")
        plan = self.architect.plan(self.ctx, goal=question)
        console.print(plan)

        console.print("\n[bold]Evaluator[/bold]")
        console.print(self.evaluator.score(plan))

        console.print("\n[bold]Critic Review[/bold]")
        critique = self.critic.review(self.ctx, plan + "\n\nRETRIEVAL:\n" + retrieval)
        console.print(critique)

        # Log interaction for later fine-tuning. This makes "it trains on what I feed it" real.
        self.store.training.log_interaction(
            interaction_id=str(uuid.uuid4()),
            ts=time.time(),
            user_text=question,
            assistant_text=plan,
            retrieved_context=retrieval,
        )

        return critique


def default_swarm(workspace_root: str) -> OfflineSwarm:
    config = OfflineSwarmConfig()
    llm: LocalLLM = EchoLLM()
    return OfflineSwarm(config=config, llm=llm, workspace_root=workspace_root)
