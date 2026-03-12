"""Graph state representation for Graph Nim.

A GraphState holds vertices, edges, and positive integer weights.
Edge keys are stored as sorted tuples so (A,B) == (B,A).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


def canonical_edge(u: Any, v: Any) -> tuple:
    """Return edge key with vertices in sorted order."""
    return (min(u, v), max(u, v))


@dataclass(frozen=True)
class GraphState:
    """Immutable snapshot of a Graph Nim position.

    Attributes:
        vertices: list of vertex labels
        edges: list of (u, v) tuples (canonical order)
        weights: dict mapping canonical edge tuple -> positive int weight
    """
    vertices: tuple
    edges: tuple
    weights: dict = field(default_factory=dict)

    @classmethod
    def create(cls, vertices: list, edges: list[tuple], weights: dict) -> GraphState:
        """Construct a GraphState with canonical edge ordering."""
        canon_edges = tuple(sorted(canonical_edge(u, v) for u, v in edges))
        canon_weights = {}
        for key, w in weights.items():
            if isinstance(key, str) and len(key) == 2:
                e = canonical_edge(key[0], key[1])
            elif isinstance(key, (tuple, list)):
                e = canonical_edge(key[0], key[1])
            else:
                e = key
            canon_weights[e] = w
        return cls(
            vertices=tuple(sorted(vertices)),
            edges=canon_edges,
            weights=dict(canon_weights),
        )

    def get_weight(self, u: Any, v: Any) -> int:
        """Get weight of edge (u, v)."""
        return self.weights.get(canonical_edge(u, v), 0)

    def incident_edges(self, v: Any) -> list[tuple]:
        """Return all edges incident to vertex v."""
        return [e for e in self.edges if v in e]

    def total_weight(self) -> int:
        """Sum of all edge weights."""
        return sum(self.weights.values())

    def apply_move(self, move: 'Move') -> GraphState:
        """Return new GraphState after applying a move."""
        new_weights = dict(self.weights)
        for edge, amount in move.reductions.items():
            e = canonical_edge(edge[0], edge[1])
            new_weights[e] = new_weights[e] - amount
        # Remove zero-weight edges
        remaining_edges = tuple(e for e in self.edges if new_weights.get(e, 0) > 0)
        remaining_weights = {e: w for e, w in new_weights.items() if w > 0}
        return GraphState(
            vertices=self.vertices,
            edges=remaining_edges,
            weights=remaining_weights,
        )
