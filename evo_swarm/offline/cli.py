from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

from rich.console import Console

from evo_swarm.offline.config import OfflineSwarmConfig
from evo_swarm.offline.llm.base import EchoLLM, LocalLLM
from evo_swarm.offline.llm.llama_cpp_server import LlamaCppServerLLM
from evo_swarm.offline.swarm import OfflineSwarm
from evo_swarm.offline.training.dataset import interactions_to_jsonl
from evo_swarm.offline.training.trainer_mlx import train_with_mlx


console = Console()


def build_llm(args: argparse.Namespace) -> LocalLLM:
    if args.llm_backend == "echo":
        return EchoLLM()
    if args.llm_backend == "llama.cpp":
        return LlamaCppServerLLM(base_url=args.llm_url)
    raise SystemExit(f"Unknown llm backend: {args.llm_backend}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evo_swarm.offline")
    parser.add_argument("--db", default="offline_swarm.db", help="SQLite DB path")
    parser.add_argument(
        "--llm-backend",
        default="echo",
        choices=["echo", "llama.cpp"],
        help="Local LLM backend",
    )
    parser.add_argument("--llm-url", default="http://127.0.0.1:8080", help="llama.cpp server base URL")

    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ingest = sub.add_parser("ingest", help="Ingest a folder of text papers (.txt/.md)")
    p_ingest.add_argument("path", help="Folder/file to ingest")

    p_ask = sub.add_parser("ask", help="Ask a question grounded in ingested papers")
    p_ask.add_argument("question", help="Question to answer")

    p_chat = sub.add_parser("chat", help="Interactive chat (logs for training)")
    p_chat.add_argument("--auto-train-every", type=int, default=0, help="Fine-tune every N assistant turns (0 disables)")
    p_chat.add_argument("--train-out", default="offline_training_out", help="Training output directory")

    p_export = sub.add_parser("export-dataset", help="Export logged interactions to JSONL")
    p_export.add_argument("--out", default="offline_dataset.jsonl", help="Output JSONL path")
    p_export.add_argument("--limit", type=int, default=2000, help="Max interactions to export")

    p_train = sub.add_parser("train", help="Run (or stub) a local fine-tune from logged interactions")
    p_train.add_argument("--out-dir", default="offline_training_out", help="Output directory for adapters/checkpoints")
    p_train.add_argument("--limit", type=int, default=2000, help="Max interactions to include")

    args = parser.parse_args(argv)

    auto_train_every = 0
    train_out_dir = "offline_training_out"
    if args.cmd == "chat":
        auto_train_every = int(args.auto_train_every)
        train_out_dir = str(args.train_out)

    config = OfflineSwarmConfig(
        db_path=args.db,
        auto_train_every=auto_train_every,
        train_out_dir=train_out_dir,
    )
    llm = build_llm(args)
    swarm = OfflineSwarm(config=config, llm=llm, workspace_root=os.getcwd())
    try:
        if args.cmd == "ingest":
            res = swarm.ingest(args.path)
            console.print(res)
            return 0 if not res["errors"] else 2
        if args.cmd == "ask":
            swarm.ask(args.question)
            return 0
        if args.cmd == "chat":
            auto_every = int(args.auto_train_every)
            train_out = str(args.train_out)
            console.print("Type '/exit' to quit.")
            while True:
                try:
                    user = input("> ").strip()
                except EOFError:
                    break
                if not user:
                    continue
                if user == "/exit":
                    break
                swarm.ask(user)

                if auto_every > 0:
                    n = swarm.store.training.count_interactions()
                    if n % auto_every == 0:
                        console.print(f"[bold]Auto-train triggered[/bold] (n={n})")
                        jsonl = interactions_to_jsonl(swarm.store.training.iter_interactions(limit=2000))
                        dataset_path = os.path.join(train_out, "dataset.jsonl")
                        os.makedirs(train_out, exist_ok=True)
                        with open(dataset_path, "w", encoding="utf-8") as f:
                            f.write(jsonl)
                        res = train_with_mlx(dataset_jsonl_path=dataset_path, out_dir=train_out)
                        console.print(res.message)
            return 0
        if args.cmd == "export-dataset":
            interactions = swarm.store.training.iter_interactions(limit=int(args.limit))
            jsonl = interactions_to_jsonl(interactions)
            out_path = os.path.abspath(str(args.out))
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(jsonl)
            console.print(f"WROTE {out_path} ({len(interactions)} interactions)")
            return 0
        if args.cmd == "train":
            out_dir = os.path.abspath(str(args.out_dir))
            os.makedirs(out_dir, exist_ok=True)
            interactions = swarm.store.training.iter_interactions(limit=int(args.limit))
            dataset_path = os.path.join(out_dir, "dataset.jsonl")
            with open(dataset_path, "w", encoding="utf-8") as f:
                f.write(interactions_to_jsonl(interactions))
            res = train_with_mlx(dataset_jsonl_path=dataset_path, out_dir=out_dir)
            console.print(res.message)
            return 0 if res.ok else 3
        raise SystemExit("unreachable")
    finally:
        swarm.close()


if __name__ == "__main__":
    raise SystemExit(main())
