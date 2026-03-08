#!/usr/bin/env python3
"""
Inspect the Evo Swarm SQLite registry: show candidates, leaderboard, and lineage trees.

Usage:
    python scripts/inspect_registry.py evo_swarm.db
    python scripts/inspect_registry.py evo_swarm.db --lineage <candidate_id_prefix>
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

console = Console()


def get_all_candidates(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, generation, status, fitness_score, genome_json, metrics_json, created_at "
        "FROM candidates ORDER BY generation, fitness_score DESC"
    ).fetchall()
    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "generation": r[1],
            "status": r[2],
            "fitness": r[3],
            "genome": json.loads(r[4]) if r[4] else {},
            "metrics": json.loads(r[5]) if r[5] else {},
            "created_at": r[6],
        })
    return results


def get_lineage(conn: sqlite3.Connection, candidate_id: str) -> list[dict]:
    chain = []
    current = candidate_id
    visited = set()
    while current and current not in visited:
        visited.add(current)
        row = conn.execute(
            "SELECT id, generation, status, fitness_score, genome_json FROM candidates WHERE id = ?",
            (current,)
        ).fetchone()
        if not row:
            break
        chain.append({
            "id": row[0], "generation": row[1], "status": row[2],
            "fitness": row[3], "genome": json.loads(row[4]) if row[4] else {},
        })
        parent = conn.execute(
            "SELECT parent_id FROM lineage WHERE candidate_id = ? LIMIT 1", (current,)
        ).fetchone()
        current = parent[0] if parent else None
    return chain


def show_table(candidates: list[dict]):
    table = Table(title="📊 Evo Swarm Registry", show_lines=True)
    table.add_column("Gen", style="cyan", justify="center")
    table.add_column("ID", style="bold")
    table.add_column("Status", style="green")
    table.add_column("Fitness", style="magenta", justify="right")
    table.add_column("Model", style="yellow")
    table.add_column("Layers", justify="center")
    table.add_column("Hidden", justify="center")
    table.add_column("LR", justify="right")
    table.add_column("Optimizer")

    for c in candidates:
        g = c["genome"]
        fitness_str = f"{c['fitness']:.4f}" if c["fitness"] is not None else "—"
        table.add_row(
            str(c["generation"]),
            c["id"][:8],
            c["status"],
            fitness_str,
            g.get("model_family", "?"),
            str(g.get("num_layers", "?")),
            str(g.get("hidden_dimension", "?")),
            f"{g.get('learning_rate', '?'):.1e}" if isinstance(g.get("learning_rate"), (int, float)) else "?",
            g.get("optimizer", "?"),
        )
    console.print(table)


def show_lineage_tree(chain: list[dict]):
    if not chain:
        console.print("[red]No lineage found.[/red]")
        return

    root_label = f"[bold]Lineage for {chain[0]['id'][:8]}[/bold]"
    tree = Tree(root_label)
    for c in chain:
        fitness_str = f"{c['fitness']:.4f}" if c['fitness'] is not None else "N/A"
        g = c["genome"]
        label = (
            f"Gen {c['generation']} | {c['id'][:8]} | "
            f"fitness={fitness_str} | "
            f"{g.get('model_family', '?')} L{g.get('num_layers', '?')} H{g.get('hidden_dimension', '?')} "
            f"lr={g.get('learning_rate', '?')}"
        )
        tree.add(label)
    console.print(tree)


def main():
    parser = argparse.ArgumentParser(description="Inspect the Evo Swarm SQLite registry")
    parser.add_argument("db", help="Path to evo_swarm.db")
    parser.add_argument("--lineage", help="Show lineage tree for a candidate ID (prefix match)")
    args = parser.parse_args()

    try:
        conn = sqlite3.connect(args.db)
    except Exception as e:
        sys.exit(f"Cannot open database: {e}")

    candidates = get_all_candidates(conn)
    if not candidates:
        console.print("[yellow]No candidates in registry.[/yellow]")
        conn.close()
        return

    show_table(candidates)

    # Stats
    generations = set(c["generation"] for c in candidates)
    evaluated = [c for c in candidates if c["fitness"] is not None]
    best = max(evaluated, key=lambda c: c["fitness"]) if evaluated else None
    console.print(f"\n[bold]Stats:[/bold] {len(candidates)} candidates across {len(generations)} generations, "
                  f"{len(evaluated)} evaluated")
    if best:
        console.print(f"[bold green]Best:[/bold green] {best['id'][:8]} — fitness {best['fitness']:.4f} (Gen {best['generation']})")

    # Lineage
    if args.lineage:
        # prefix match
        match = [c for c in candidates if c["id"].startswith(args.lineage)]
        if match:
            chain = get_lineage(conn, match[0]["id"])
            console.print()
            show_lineage_tree(chain)
        else:
            console.print(f"[red]No candidate matching prefix '{args.lineage}'[/red]")
    elif best:
        chain = get_lineage(conn, best["id"])
        if len(chain) > 1:
            console.print()
            show_lineage_tree(chain)

    conn.close()


if __name__ == "__main__":
    main()
