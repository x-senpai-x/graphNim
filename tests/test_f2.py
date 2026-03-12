"""Tests for F2 solver (Triangle + pendant)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from graph_state import GraphState
from classifier import classify, remap_state
from solvers.f2 import F2Solver


def _make(ab, bc, cd, db):
    """Create F2 state with canonical labels.

    F2: pendant A-B, triangle B-C-D.
    Edges: AB, BC, CD, DB.
    """
    state = GraphState.create(
        ["A", "B", "C", "D"],
        [("A", "B"), ("B", "C"), ("C", "D"), ("D", "B")],
        {("A", "B"): ab, ("B", "C"): bc, ("C", "D"): cd, ("D", "B"): db},
    )
    gtype, label_map = classify(state)
    assert gtype == "F2"
    canonical = remap_state(state, label_map)
    return F2Solver(), canonical


class TestF2:
    """Test F2 = Triangle + pendant (Theorem 4.3)."""

    def test_losing_condition(self):
        """LOSING iff BC=DB and CD=AB+BC.

        Example: AB=2, BC=3, CD=5, DB=3 → BC=DB=3, CD=5=2+3 → LOSING.
        """
        solver, state = _make(2, 3, 5, 3)
        assert solver.evaluate(state) == "LOSING"

    def test_losing_simple(self):
        """AB=1, BC=1, CD=2, DB=1 → LOSING."""
        solver, state = _make(1, 1, 2, 1)
        assert solver.evaluate(state) == "LOSING"

    def test_winning_bc_ne_db(self):
        """AB=2, BC=3, CD=5, DB=4 → BC≠DB → WINNING."""
        solver, state = _make(2, 3, 5, 4)
        assert solver.evaluate(state) == "WINNING"

    def test_winning_cd_ne_sum(self):
        """AB=2, BC=3, CD=4, DB=3 → CD≠AB+BC → WINNING."""
        solver, state = _make(2, 3, 4, 3)
        assert solver.evaluate(state) == "WINNING"

    def test_winning_move_valid(self):
        solver, state = _make(2, 3, 5, 4)
        move = solver.winning_move(state)
        assert move is not None
        assert move.is_valid(state)

    def test_winning_move_leads_to_losing(self):
        solver, state = _make(2, 3, 5, 4)
        move = solver.winning_move(state)
        new_state = state.apply_move(move)
        new_solver = F2Solver()
        assert new_solver.evaluate(new_state) == "LOSING"
