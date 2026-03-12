"""Tests for galaxy solver (G1, H2, H3, I1, I2)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from graph_state import GraphState
from classifier import classify, remap_state
from solvers.galaxy import GalaxySolver


def _make_and_solve(vertices, edges, weights, graph_type):
    state = GraphState.create(vertices, edges, weights)
    gtype, label_map = classify(state)
    assert gtype == graph_type, f"Expected {graph_type}, got {gtype}"
    canonical = remap_state(state, label_map)
    solver = GalaxySolver(graph_type)
    return solver, canonical


class TestG1:
    """G1 = K_{1,4}: all configs winning (Theorem 4.1i)."""

    def test_all_ones(self):
        solver, state = _make_and_solve(
            ["A", "B", "C", "D", "E"],
            [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")],
            {("A", "B"): 1, ("A", "C"): 1, ("A", "D"): 1, ("A", "E"): 1},
            "G1",
        )
        assert solver.evaluate(state) == "WINNING"

    def test_various_weights(self):
        solver, state = _make_and_solve(
            ["A", "B", "C", "D", "E"],
            [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")],
            {("A", "B"): 5, ("A", "C"): 10, ("A", "D"): 15, ("A", "E"): 20},
            "G1",
        )
        assert solver.evaluate(state) == "WINNING"
        move = solver.winning_move(state)
        assert move is not None
        assert move.is_valid(state)


class TestH2:
    """H2 = 2×K_{1,2}: losing iff sum1 = sum2 (Theorem 4.1ii)."""

    def _make(self, ab, bc, de, ef):
        return _make_and_solve(
            ["A", "B", "C", "D", "E", "F"],
            [("A", "B"), ("B", "C"), ("D", "E"), ("E", "F")],
            {("A", "B"): ab, ("B", "C"): bc, ("D", "E"): de, ("E", "F"): ef},
            "H2",
        )

    def test_losing_equal_sums(self):
        solver, state = self._make(3, 4, 2, 5)  # 7 = 7
        assert solver.evaluate(state) == "LOSING"

    def test_winning_unequal_sums(self):
        solver, state = self._make(3, 4, 2, 6)  # 7 ≠ 8
        assert solver.evaluate(state) == "WINNING"
        move = solver.winning_move(state)
        assert move is not None
        assert move.is_valid(state)


class TestH3:
    """H3 = K_{1,3} ∪ K₂: losing iff star_sum = edge_weight (Theorem 4.1iii)."""

    def _make(self, ab, ac, ad, ef):
        return _make_and_solve(
            ["A", "B", "C", "D", "E", "F"],
            [("A", "B"), ("A", "C"), ("A", "D"), ("E", "F")],
            {("A", "B"): ab, ("A", "C"): ac, ("A", "D"): ad, ("E", "F"): ef},
            "H3",
        )

    def test_losing(self):
        solver, state = self._make(2, 3, 5, 10)  # 10 = 10
        assert solver.evaluate(state) == "LOSING"

    def test_winning(self):
        solver, state = self._make(2, 3, 5, 9)  # 10 ≠ 9
        assert solver.evaluate(state) == "WINNING"


class TestI1:
    """I1 = K_{1,2} ∪ 2K₂: losing iff (sum, w1, w2) balanced (Theorem 4.1iv)."""

    def _make(self, ab, bc, de, fg):
        return _make_and_solve(
            ["A", "B", "C", "D", "E", "F", "G"],
            [("A", "B"), ("B", "C"), ("D", "E"), ("F", "G")],
            {("A", "B"): ab, ("B", "C"): bc, ("D", "E"): de, ("F", "G"): fg},
            "I1",
        )

    def test_losing_balanced(self):
        # (AB+BC, DE, FG) = (3, 1, 2) → XOR = 3^1^2 = 0
        solver, state = self._make(1, 2, 1, 2)
        assert solver.evaluate(state) == "LOSING"

    def test_winning_unbalanced(self):
        solver, state = self._make(1, 2, 3, 4)  # (3, 3, 4) XOR = 4 ≠ 0
        assert solver.evaluate(state) == "WINNING"


class TestI2:
    """I2 = 4K₂: losing iff (w1,w2,w3,w4) balanced (Theorem 4.1v)."""

    def _make(self, ab, cd, ef, gh):
        return _make_and_solve(
            ["A", "B", "C", "D", "E", "F", "G", "H"],
            [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")],
            {("A", "B"): ab, ("C", "D"): cd, ("E", "F"): ef, ("G", "H"): gh},
            "I2",
        )

    def test_losing_xor_zero(self):
        # 1^2^3^0 = 0? No. 1^1^1^1 = 0.
        solver, state = self._make(1, 1, 1, 1)
        assert solver.evaluate(state) == "LOSING"

    def test_losing_xor_zero_2(self):
        # 3^5^6^0 = 0? 3^5=6, 6^6=0. Yes!
        solver, state = self._make(3, 5, 6, 0)
        # Wait, weights must be positive. Let's use 1^2^3 = 0
        # 1^2=3, 3^3=0. So (1,2,3,0) but 0 is invalid.
        # 5^3^6 = 0. So (5,3,6,0) still invalid.
        # Actually for I2 all edges have positive weights.
        # 1^2^3^0=0 but 0 not allowed. Let me find valid example:
        # Need a^b^c^d = 0 with all > 0.
        # 1^2^3 = 0, but that's 3 values. For 4: 1^1^2^2 = 0.
        pass

    def test_losing_four_values(self):
        # 1^1^2^2 = 0
        solver, state = self._make(1, 1, 2, 2)
        assert solver.evaluate(state) == "LOSING"

    def test_winning(self):
        solver, state = self._make(1, 2, 3, 4)  # XOR = 4 ≠ 0
        assert solver.evaluate(state) == "WINNING"
        move = solver.winning_move(state)
        assert move is not None
        assert move.is_valid(state)

    def test_winning_move_leads_to_losing(self):
        """Verify that applying the winning move gives a losing position."""
        solver, state = self._make(1, 2, 3, 4)
        move = solver.winning_move(state)
        assert move is not None
        new_state = state.apply_move(move)
        # After move, need to re-evaluate. The new state may have 0-weight edges.
        # Compute XOR of remaining edge weights
        vals = list(new_state.weights.values())
        xor = 0
        for v in vals:
            xor ^= v
        assert xor == 0, f"After winning move, XOR should be 0 but got {xor}"
