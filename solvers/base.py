"""Base classes for Graph Nim solvers.

Provides the Move dataclass and abstract Solver interface.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from graph_state import GraphState, canonical_edge


@dataclass
class Move:
    """A single Graph Nim move: select a vertex and reduce incident edges.

    Attributes:
        vertex: the selected vertex
        reductions: dict mapping edge (tuple) -> amount to remove (>= 0)
    """
    vertex: Any
    reductions: dict[tuple, int] = field(default_factory=dict)

    def is_valid(self, state: GraphState) -> bool:
        """Check move validity per Definition 1.2.

        Rules:
        - Total removed must be > 0
        - Each reduction >= 0 and <= current edge weight
        - Only edges incident to self.vertex may be reduced
        - Edges not incident to self.vertex must not appear
        """
        total = 0
        for edge, amount in self.reductions.items():
            e = canonical_edge(edge[0], edge[1])
            # Edge must be incident to the selected vertex
            if self.vertex not in e:
                return False
            # Amount must be non-negative
            if amount < 0:
                return False
            # Amount must not exceed current weight
            if amount > state.get_weight(e[0], e[1]):
                return False
            total += amount
        # Total removed must be strictly positive
        return total > 0


class Solver(ABC):
    """Abstract base class for Graph Nim solvers."""

    @abstractmethod
    def evaluate(self, state: GraphState) -> Literal["WINNING", "LOSING", "UNKNOWN"]:
        """Classify the position."""
        ...

    @abstractmethod
    def winning_move(self, state: GraphState) -> Optional[Move]:
        """Return a concrete winning move, or None if LOSING/UNKNOWN."""
        ...
