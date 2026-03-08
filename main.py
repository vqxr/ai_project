from __future__ import annotations

import argparse
import threading
import time

from evo_swarm.agents.architect import ArchitectAgent
from evo_swarm.agents.critic import CriticMutatorAgent
from evo_swarm.agents.curator import CuratorAgent
from evo_swarm.agents.evaluator import EvaluatorAgent
from evo_swarm.agents.trainer import TrainerAgent
from evo_swarm.core.registry.sqlite_registry import SqliteRegistry
from evo_swarm.core.scheduler.local_scheduler import LocalEventScheduler
from evo_swarm.evolution.generation_manager import GenerationManager


def main():
    parser = argparse.ArgumentParser(description="Evo Swarm — Evolutionary Intelligence Framework")
    parser.add_argument("--db", default="evo_swarm.db", help="SQLite database path (default: evo_swarm.db)")
    parser.add_argument("--population-size", type=int, default=3, help="Candidates per generation (default: 3)")
    parser.add_argument("--max-generations", type=int, default=5, help="Max generations to evolve (default: 5)")
    parser.add_argument("--target-fitness", type=float, default=0.90, help="Fitness threshold for champion (default: 0.90)")
    parser.add_argument("--top-k", type=int, default=2, help="Top-K parents to breed from each generation (default: 2)")
    parser.add_argument("--timeout", type=int, default=30, help="Max seconds to run (default: 30)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Evo Swarm — Distributed Evolutionary Intelligence")
    print("=" * 60)
    print(f"  Population size:  {args.population_size}")
    print(f"  Max generations:  {args.max_generations}")
    print(f"  Target fitness:   {args.target_fitness}")
    print(f"  Top-K parents:    {args.top_k}")
    print(f"  Database:         {args.db}")
    print("=" * 60)

    # 1. Setup Core Infrastructure
    registry = SqliteRegistry(db_path=args.db)
    scheduler = LocalEventScheduler()

    # 2. Initialize Agents
    architect = ArchitectAgent(population_size=args.population_size)
    trainer = TrainerAgent()
    evaluator = EvaluatorAgent()
    critic = CriticMutatorAgent(target_fitness=args.target_fitness)
    curator = CuratorAgent()
    generation_manager = GenerationManager(
        registry=registry,
        population_size=args.population_size,
        top_k=args.top_k,
        max_generations=args.max_generations,
        target_fitness=args.target_fitness,
    )

    # 3. Register Agents with Scheduler
    for agent in [architect, trainer, evaluator, critic, curator, generation_manager]:
        scheduler.register_agent(agent)

    # 4. Start Scheduler loop in a background thread
    scheduler_thread = threading.Thread(target=scheduler.start, daemon=True)
    scheduler_thread.start()
    time.sleep(0.3)

    # 5. Kick off Generation 0
    print("\nStarting the Swarm...\n")
    curator.trigger_new_generation(generation=0)

    # Wait for the system to run
    try:
        time.sleep(args.timeout)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        scheduler.stop()
        scheduler_thread.join(timeout=2.0)

    # 6. Print Results
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    best = registry.get_best_candidates(limit=5)
    if not best:
        print("  No candidates were fully evaluated.")
    else:
        champion = best[0]
        print(f"  🏆 Top Candidate: {champion.id[:8]}")
        print(f"     Fitness:       {champion.fitness_score:.4f}")
        print(f"     Generation:    {champion.generation}")
        print(f"     Parents:       {len(champion.parent_ids)}")
        print(f"     Architecture:  {champion.genome.model_family}")
        print(f"     Layers:        {champion.genome.num_layers}")
        print(f"     Hidden dim:    {champion.genome.hidden_dimension}")
        print(f"     Learning rate: {champion.genome.learning_rate}")
        print(f"     Optimizer:     {champion.genome.optimizer}")
        print(f"     Curriculum:    {champion.genome.curriculum_strategy}")

        # Show ancestry chain
        lineage = registry.get_lineage_tree(champion.id)
        if len(lineage) > 1:
            print(f"\n  --- Ancestry Chain ({len(lineage)} nodes) ---")
            for i, ancestor in enumerate(lineage):
                prefix = "  └─" if i == len(lineage) - 1 else "  ├─"
                score = f"{ancestor.fitness_score:.4f}" if ancestor.fitness_score else "N/A"
                print(f"  {prefix} Gen {ancestor.generation} | {ancestor.id[:8]} | "
                      f"fitness={score} | layers={ancestor.genome.num_layers} lr={ancestor.genome.learning_rate}")

        # Show top-5 leaderboard
        if len(best) > 1:
            print(f"\n  --- Top {len(best)} Leaderboard ---")
            for rank, c in enumerate(best, 1):
                score = f"{c.fitness_score:.4f}" if c.fitness_score else "N/A"
                print(f"  {rank}. {c.id[:8]} | Gen {c.generation} | fitness={score} | "
                      f"{c.genome.model_family} L{c.genome.num_layers} H{c.genome.hidden_dimension}")

    print("=" * 60)
    registry.close()


if __name__ == "__main__":
    main()
