"""CLI for Graph Nim solver.

Usage:
  python main.py --vertices A B C D E F --edges AB BC CD EF --weights AB:3 BC:2 CD:5 EF:4
  python main.py --json '{"vertices":["A","B","C","D","E","F"],"edges":[["A","B"],["B","C"],["C","D"],["E","F"]],"weights":{"AB":3,"BC":2,"CD":5,"EF":4}}'
"""

from __future__ import annotations
import argparse
import json
import sys

from graph_state import GraphState, canonical_edge
from classifier import classify, remap_state
from solvers.base import Move
from solvers.galaxy import GalaxySolver
from solvers.f1 import F1Solver
from solvers.f2 import F2Solver
from solvers.g2_g3 import G2G3Solver
from solvers.g4 import G4Solver
from solvers.h1 import H1Solver
from retrograde import retrograde_search


SOLVER_MAP = {
    "G1": lambda: GalaxySolver("G1"),
    "H2": lambda: GalaxySolver("H2"),
    "H3": lambda: GalaxySolver("H3"),
    "I1": lambda: GalaxySolver("I1"),
    "I2": lambda: GalaxySolver("I2"),
    "F1": lambda: F1Solver(),
    "F2": lambda: F2Solver(),
    "G2": lambda: G2G3Solver("G2"),
    "G3": lambda: G2G3Solver("G3"),
    "G4": lambda: G4Solver(),
    "H1": lambda: H1Solver(),
}


def parse_edge(s: str) -> tuple[str, str]:
    """Parse edge string like 'AB' into ('A', 'B')."""
    if len(s) == 2:
        return (s[0], s[1])
    raise ValueError(f"Invalid edge format: {s}. Expected 2 characters like 'AB'.")


def parse_weight(s: str) -> tuple[tuple[str, str], int]:
    """Parse weight string like 'AB:3' into (('A', 'B'), 3)."""
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid weight format: {s}. Expected 'AB:3'.")
    edge = parse_edge(parts[0])
    weight = int(parts[1])
    return edge, weight


def build_state_from_args(args) -> GraphState:
    """Build GraphState from CLI arguments."""
    vertices = args.vertices
    edges = [parse_edge(e) for e in args.edges]
    weights = {}
    for w in args.weights:
        edge, weight = parse_weight(w)
        weights[canonical_edge(*edge)] = weight
    return GraphState.create(vertices, edges, weights)


def build_state_from_json(json_str: str) -> GraphState:
    """Build GraphState from JSON string."""
    data = json.loads(json_str)
    vertices = data["vertices"]
    edges = [tuple(e) for e in data["edges"]]
    weights = {}
    for key, val in data["weights"].items():
        if len(key) == 2:
            weights[canonical_edge(key[0], key[1])] = val
        else:
            # Try parsing as list
            e = tuple(key.split(","))
            weights[canonical_edge(e[0], e[1])] = val
    return GraphState.create(vertices, edges, weights)


def format_move(move: Move, label_map_inv: dict) -> str:
    """Format a move for display, converting canonical labels back to user labels."""
    lines = [f"  Select vertex: {label_map_inv.get(move.vertex, move.vertex)}"]
    total = 0
    for edge, amount in sorted(move.reductions.items()):
        u = label_map_inv.get(edge[0], edge[0])
        v = label_map_inv.get(edge[1], edge[1])
        lines.append(f"  Remove from {u}{v}: {amount}")
        total += amount
    lines.append(f"  Total removed: {total}")
    return "\n".join(lines)


def solve(state: GraphState) -> None:
    """Classify graph, evaluate position, and output result."""
    # Classify
    graph_type, label_map = classify(state)
    label_map_inv = {v: k for k, v in label_map.items()}

    # Remap to canonical labels
    canonical_state = remap_state(state, label_map)

    # Get solver
    solver = SOLVER_MAP[graph_type]()

    # Evaluate
    result = solver.evaluate(canonical_state)

    # Try retrograde for UNKNOWN
    retro_move = None
    if result == "UNKNOWN" and graph_type == "H1":
        retro_result, retro_move = retrograde_search(canonical_state)
        if retro_result != "UNKNOWN":
            result = retro_result

    print(f"Graph Type: {graph_type}")
    print(f"Position:   {result}")

    if result == "WINNING":
        move = retro_move or solver.winning_move(canonical_state)
        if move:
            print("Winning Move:")
            print(format_move(move, label_map_inv))
        else:
            print("Winning Move: (could not construct)")


def main():
    parser = argparse.ArgumentParser(description="Graph Nim Solver (4 edges)")
    parser.add_argument("--vertices", nargs="+", help="Vertex labels")
    parser.add_argument("--edges", nargs="+", help="Edges (e.g., AB BC CD EF)")
    parser.add_argument("--weights", nargs="+", help="Weights (e.g., AB:3 BC:2)")
    parser.add_argument("--json", type=str, help="JSON input")

    args = parser.parse_args()

    if args.json:
        state = build_state_from_json(args.json)
    elif args.vertices and args.edges and args.weights:
        state = build_state_from_args(args)
    else:
        parser.print_help()
        sys.exit(1)

    solve(state)


if __name__ == "__main__":
    main()
