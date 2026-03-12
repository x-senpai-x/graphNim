"""Galaxy graph solver for G1, H2, H3, I1, I2.

Implements Theorem 2.3 (arXiv:2509.05064v1): a galaxy graph configuration
is losing iff the tuple of star-sums is balanced (XOR = 0 in standard Nim).

G1 (K_{1,4}): single star → single pile → always WINNING (Theorem 4.1i).
H2 (2×K_{1,2}): 2 stars → LOSING iff sum1 == sum2 (Theorem 4.1ii).
H3 (K_{1,3} ∪ K₂): 2 stars → LOSING iff star_sum == edge_weight (Theorem 4.1iii).
I1 (K_{1,2} ∪ 2K₂): 3 stars → LOSING iff (sum, w1, w2) balanced (Theorem 4.1iv).
I2 (4K₂): 4 stars → LOSING iff (w1,w2,w3,w4) balanced (Theorem 4.1v).
"""

from __future__ import annotations
from typing import Literal, Optional

import networkx as nx

from graph_state import GraphState, canonical_edge
from solvers.base import Move, Solver


def _compute_star_sums(state: GraphState) -> list[tuple[list[tuple], int]]:
    """Compute (edges_list, sum_of_weights) for each star component.

    Returns list of (edges_in_star, total_weight) sorted by total_weight descending.
    """
    G = nx.Graph()
    for e in state.edges:
        if state.weights.get(e, 0) > 0:
            G.add_edge(e[0], e[1])
    # Add isolated vertices too
    for v in state.vertices:
        G.add_node(v)

    stars = []
    for comp_nodes in nx.connected_components(G):
        comp_edges = [e for e in state.edges if e[0] in comp_nodes and e[1] in comp_nodes]
        total = sum(state.weights.get(e, 0) for e in comp_edges)
        if total > 0:
            stars.append((comp_edges, total))
    return stars


def _nim_xor(values: list[int]) -> int:
    """Compute XOR of a list of integers."""
    result = 0
    for v in values:
        result ^= v
    return result


def _find_star_center(state: GraphState, edges: list[tuple]) -> Optional[str]:
    """Find the center vertex of a star given its edges.

    For K₂ (single edge), either endpoint works as 'center'.
    For K_{1,n} with n>1, the center is the vertex appearing in all edges.
    """
    if len(edges) == 1:
        return edges[0][0]  # either endpoint
    # Count vertex appearances
    counts: dict = {}
    for e in edges:
        for v in e:
            counts[v] = counts.get(v, 0) + 1
    # Center appears in all edges
    for v, c in counts.items():
        if c == len(edges):
            return v
    return None


class GalaxySolver(Solver):
    """Solver for galaxy graphs G1, H2, H3, I1, I2.

    Implements Theorem 2.3, arXiv:2509.05064v1.
    """

    def __init__(self, graph_type: str):
        self.graph_type = graph_type

    def evaluate(self, state: GraphState) -> Literal["WINNING", "LOSING", "UNKNOWN"]:
        """Classify position using Theorem 2.3.

        Implements Theorem 4.1, arXiv:2509.05064v1.
        """
        # G1 (K_{1,4}): all configs winning — single pile Nim
        if self.graph_type == "G1":
            return "WINNING"

        stars = _compute_star_sums(state)
        sums = [s for _, s in stars]
        nim_sum = _nim_xor(sums)

        if nim_sum == 0:
            return "LOSING"
        return "WINNING"

    def winning_move(self, state: GraphState) -> Optional[Move]:
        """Compute winning move using standard Nim strategy.

        For galaxy Nim, reduce one star's total so that the resulting
        tuple of star-sums is balanced (XOR = 0).

        Implements constructive strategy from Theorem 2.3, arXiv:2509.05064v1.
        """
        if self.evaluate(state) != "WINNING":
            return None

        stars = _compute_star_sums(state)

        # G1: single pile, always winning. Remove 1 from any edge.
        if self.graph_type == "G1":
            for e in state.edges:
                w = state.weights.get(e, 0)
                if w > 0:
                    center = _find_star_center(state, list(state.edges))
                    return Move(vertex=center, reductions={e: 1})
            return None

        sums = [s for _, s in stars]
        nim_sum = _nim_xor(sums)

        # Find a star where (sum XOR nim_sum) < sum
        target_idx = None
        for i, s in enumerate(sums):
            new_s = s ^ nim_sum
            if new_s < s:
                target_idx = i
                break

        if target_idx is None:
            return None

        target_edges, current_sum = stars[target_idx]
        new_sum = current_sum ^ nim_sum
        reduction_needed = current_sum - new_sum  # total to remove from this star

        # For a star, we can select a leaf and reduce its single edge,
        # or select the center and reduce multiple edges.
        # Strategy: select a leaf of one edge and reduce it by the full amount
        # if possible, otherwise use center.

        if len(target_edges) == 1:
            # Single edge star (K₂): select either endpoint
            edge = target_edges[0]
            vertex = edge[0]
            return Move(vertex=vertex, reductions={edge: reduction_needed})

        # Multi-edge star: select center and reduce one edge by the needed amount
        center = _find_star_center(state, target_edges)
        if center is None:
            return None

        # Try to reduce a single edge that has enough weight
        for edge in target_edges:
            w = state.weights.get(edge, 0)
            if w >= reduction_needed:
                return Move(vertex=center, reductions={edge: reduction_needed})

        # Need to reduce across multiple edges from center
        remaining = reduction_needed
        reductions = {}
        for edge in target_edges:
            w = state.weights.get(edge, 0)
            if remaining <= 0:
                break
            take = min(w, remaining)
            if take > 0:
                reductions[edge] = take
                remaining -= take
        return Move(vertex=center, reductions=reductions)
