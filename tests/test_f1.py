"""Tests for F1 solver (C₄ cycle)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from graph_state import GraphState
from classifier import classify, remap_state
from solvers.f1 import F1Solver


def _make(ab, bc, cd, da):
    state = GraphState.create(
        ["A", "B", "C", "D"],
        [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")],
        {("A", "B"): ab, ("B", "C"): bc, ("C", "D"): cd, ("D", "A"): da},
    )
    gtype, label_map = classify(state)
    assert gtype == "F1"
    canonical = remap_state(state, label_map)
    return F1Solver(), canonical


class TestF1:
    """Test F1 = C₄ (Theorem 4.2)."""

    def test_losing_opposite_equal(self):
        """(3,4,3,4) → LOSING: AB=CD=3, BC=DA=4."""
        solver, state = _make(3, 4, 3, 4)
        assert solver.evaluate(state) == "LOSING"

    def test_losing_all_equal(self):
        """(5,5,5,5) → LOSING."""
        solver, state = _make(5, 5, 5, 5)
        assert solver.evaluate(state) == "LOSING"

    def test_winning_unequal(self):
        """(3,4,3,5) → WINNING."""
        solver, state = _make(3, 4, 3, 5)
        assert solver.evaluate(state) == "WINNING"

    def test_winning_move_valid(self):
        solver, state = _make(3, 4, 3, 5)
        move = solver.winning_move(state)
        assert move is not None
        assert move.is_valid(state)

    def test_winning_move_leads_to_losing(self):
        """After winning move, opposite edges should be equal."""
        solver, state = _make(3, 4, 3, 5)
        move = solver.winning_move(state)
        new_state = state.apply_move(move)
        # Check opposite edges equal
        new_solver = F1Solver()
        assert new_solver.evaluate(new_state) == "LOSING"

    def test_winning_asymmetric(self):
        solver, state = _make(1, 2, 3, 4)
        assert solver.evaluate(state) == "WINNING"
        move = solver.winning_move(state)
        assert move is not None
        assert move.is_valid(state)
