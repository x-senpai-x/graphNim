"""Bounded retrograde analysis for H1 UNKNOWN configurations.

When the H1 solver returns UNKNOWN (beyond Theorem 4.10's scope),
this module performs a bounded search to attempt classification.

A state is LOSING if ALL moves lead to WINNING states.
A state is WINNING if SOME move leads to a LOSING state.
"""

from __future__ import annotations
from typing import Literal, Optional

from graph_state import GraphState, canonical_edge
from solvers.base import Move


def _generate_moves(state: GraphState) -> list[Move]:
    """Generate all 'reasonable' moves for bounded search.

    For each vertex, enumerate reductions of incident edges up to
    a bounded maximum. Since the move space is technically infinite
    (any reduction amounts), we bound by current edge weights.
    """
    moves = []
    for v in state.vertices:
        incident = state.incident_edges(v)
        if not incident:
            continue

        if len(incident) == 1:
            # Single edge: reduce by 1 to full weight
            e = incident[0]
            w = state.weights.get(e, 0)
            for amount in range(1, w + 1):
                moves.append(Move(vertex=v, reductions={e: amount}))
        elif len(incident) == 2:
            e1, e2 = incident
            w1 = state.weights.get(e1, 0)
            w2 = state.weights.get(e2, 0)
            for a1 in range(w1 + 1):
                for a2 in range(w2 + 1):
                    if a1 + a2 > 0:
                        reductions = {}
                        if a1 > 0:
                            reductions[e1] = a1
                        if a2 > 0:
                            reductions[e2] = a2
                        moves.append(Move(vertex=v, reductions=reductions))
        elif len(incident) == 3:
            e1, e2, e3 = incident
            w1 = state.weights.get(e1, 0)
            w2 = state.weights.get(e2, 0)
            w3 = state.weights.get(e3, 0)
            for a1 in range(w1 + 1):
                for a2 in range(w2 + 1):
                    for a3 in range(w3 + 1):
                        if a1 + a2 + a3 > 0:
                            reductions = {}
                            if a1 > 0:
                                reductions[e1] = a1
                            if a2 > 0:
                                reductions[e2] = a2
                            if a3 > 0:
                                reductions[e3] = a3
                            moves.append(Move(vertex=v, reductions=reductions))

    return moves


def _state_key(state: GraphState) -> tuple:
    """Create a hashable key for memoization."""
    return tuple(sorted(state.weights.items()))


def retrograde_search(
    state: GraphState,
    max_total_weight: int = 50,
) -> tuple[Literal["WINNING", "LOSING", "UNKNOWN"], Optional[Move]]:
    """Bounded retrograde analysis.

    Returns (result, winning_move_or_None).
    Only attempts search if total weight is within max_total_weight.
    """
    if state.total_weight() > max_total_weight:
        return "UNKNOWN", None

    memo: dict[tuple, Literal["WINNING", "LOSING"]] = {}

    def solve(st: GraphState) -> Literal["WINNING", "LOSING"]:
        key = _state_key(st)
        if key in memo:
            return memo[key]

        # Terminal: no moves possible = all weights 0
        if st.total_weight() == 0:
            # Player to move loses (no move available)
            memo[key] = "LOSING"
            return "LOSING"

        moves = _generate_moves(st)
        if not moves:
            memo[key] = "LOSING"
            return "LOSING"

        # WINNING if any move leads to LOSING
        for move in moves:
            next_st = st.apply_move(move)
            if solve(next_st) == "LOSING":
                memo[key] = "WINNING"
                return "WINNING"

        memo[key] = "LOSING"
        return "LOSING"

    result = solve(state)

    # Find winning move if WINNING
    if result == "WINNING":
        moves = _generate_moves(state)
        for move in moves:
            next_st = state.apply_move(move)
            if solve(next_st) == "LOSING":
                return "WINNING", move

    return result, None
