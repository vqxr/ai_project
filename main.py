import sys
import threading
import time
from evo_swarm.core.scheduler.local_scheduler import LocalEventScheduler
from evo_swarm.core.registry.sqlite_registry import SqliteRegistry
from evo_swarm.agents.architect import ArchitectAgent
from evo_swarm.agents.trainer import TrainerAgent
from evo_swarm.agents.evaluator import EvaluatorAgent
from evo_swarm.agents.critic import CriticMutatorAgent
from evo_swarm.agents.curator import CuratorAgent
from evo_swarm.evolution.generation_manager import GenerationManager

def main():
    print("Initializing Evo Swarm...")
    
    # 1. Setup Core Infrastructure
    registry = SqliteRegistry(db_path="evo_swarm.db")
    scheduler = LocalEventScheduler()
    
    # 2. Initialize Agents
    architect = ArchitectAgent()
    trainer = TrainerAgent()
    evaluator = EvaluatorAgent()
    critic = CriticMutatorAgent(target_fitness=0.90)  # High to force some mutations
    curator = CuratorAgent()
    generation_manager = GenerationManager(registry=registry)
    
    # 3. Register Agents with Scheduler
    scheduler.register_agent(architect)
    scheduler.register_agent(trainer)
    scheduler.register_agent(evaluator)
    scheduler.register_agent(critic)
    scheduler.register_agent(curator)
    scheduler.register_agent(generation_manager)
    
    # 4. Start Scheduler loop in a background thread
    scheduler_thread = threading.Thread(target=scheduler.start, daemon=True)
    scheduler_thread.start()
    
    # Give the scheduler a moment to start
    time.sleep(0.5)
    
    # 5. Kick off Generation 0
    # We do this by telling the Curator to emit a DATASET_UPDATE
    print("\nStarting the Swarm...")
    curator.trigger_new_generation(generation=0)
    
    # Wait for the system to run and events to propagate
    # In V1, we'll just sleep for a sufficiently long time to simulate 1-3 loops.
    # The queue processes things instantly, but the trainer/evaluator have 1s sleeps.
    try:
        # Give it roughly enough time for 3 mutational loops
        # 3 loops = 3 * (1s + 1s) = 6 seconds
        time.sleep(7.0)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        scheduler.stop()
        scheduler_thread.join(timeout=2.0)
        
    # 6. Print Results
    print("\n--- Final Lineage Status ---")
    best = registry.get_best_candidates(limit=3)
    if not best:
        print("No candidates were fully evaluated.")
    else:
        champion = best[0]
        print(f"Top Candidate: {champion.id[:8]}")
        print(f"Fitness: {champion.fitness_score:.4f}")
        print(f"Generation: {champion.generation}")
        print(f"Lineage depth (parents): {len(champion.parent_ids)}")
        print(f"Model Genotype: layers={champion.genome.num_layers}, lr={champion.genome.learning_rate}")
        
        # Show ancestry chain from the SQLite lineage table
        lineage = registry.get_lineage_tree(champion.id)
        if len(lineage) > 1:
            print(f"\n--- Ancestry Chain ({len(lineage)} nodes) ---")
            for i, ancestor in enumerate(lineage):
                prefix = "└─" if i == len(lineage) - 1 else "├─"
                score = f"{ancestor.fitness_score:.4f}" if ancestor.fitness_score else "N/A"
                print(f"  {prefix} Gen {ancestor.generation} | {ancestor.id[:8]} | fitness={score}")
    
    registry.close()

if __name__ == "__main__":
    main()
