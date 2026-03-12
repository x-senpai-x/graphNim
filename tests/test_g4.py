"""Tests for G4 solver (Triangle ∪ K₂)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from graph_state import GraphState
from classifier import classify, remap_state
from solvers.g4 import G4Solver, is_special, _find_m_for_k


def _make(ab, bc, ca, de):
    """Create G4 state: triangle A-B-C + isolated edge D-E."""
    state = GraphState.create(
        ["A", "B", "C", "D", "E"],
        [("A", "B"), ("B", "C"), ("C", "A"), ("D", "E")],
        {("A", "B"): ab, ("B", "C"): bc, ("C", "A"): ca, ("D", "E"): de},
    )
    gtype, label_map = classify(state)
    assert gtype == "G4"
    canonical = remap_state(state, label_map)
    return G4Solver(), canonical


class TestSpecialMultiset:
    """Test Definition 4.5: (k,ℓ,m,i)-special multisets."""

    def test_find_m(self):
        """m(m+1)/2 ≤ k ≤ m(m+3)/2."""
        assert _find_m_for_k(1) == 1  # 1 ≤ 1 ≤ 2
        assert _find_m_for_k(2) == 1  # 1 ≤ 2 ≤ 2
        assert _find_m_for_k(3) == 2  # 3 ≤ 3 ≤ 5
        assert _find_m_for_k(5) == 2  # 3 ≤ 5 ≤ 5
        assert _find_m_for_k(6) == 3  # 6 ≤ 6 ≤ 9

    def test_special_k1_m1(self):
        """k=1, m=1, i∈{1,2}, ℓ=0: {2, 1+i, 1+3-i} = {2, 1+i, 4-i}.
        i=1: {2,2,3}. i=2: {2,3,2} = {2,2,3}."""
        assert is_special([2, 2, 3], 1) == True

    def test_special_k1_ell1(self):
        """k=1, m=1, ℓ=1: {k+1+(m+1)ℓ, k+i+(m+1)ℓ, k+m+2-i+(m+1)ℓ}
        = {4, 3+i, 6-i}.
        i=1: {4,4,5}. i=2: {4,5,4}={4,4,5}."""
        assert is_special([4, 4, 5], 1) == True
        assert is_special([4, 4, 4], 1) == False

    def test_not_special(self):
        assert is_special([1, 2, 3], 1) == False
        assert is_special([1, 1, 1], 1) == False

    def test_special_k3(self):
        """k=3, m=2, i∈{1,2,3}, ℓ=0: {base+1, base+i, base+m+2-i}
        base=3. {4, 3+i, 3+4-i} = {4, 3+i, 7-i}.
        i=1: {4,4,6}. i=2: {4,5,5}. i=3: {4,6,4}={4,4,6}."""
        assert is_special([4, 4, 6], 3) == True
        assert is_special([4, 5, 5], 3) == True
        assert is_special([4, 4, 5], 3) == False


class TestG4Solver:
    """Test G4 solver (Theorem 4.6)."""

    def test_a1_losing(self):
        """A1: triangle multiset is special for k=DE.
        k=1, special={2,2,3}: AB=2, BC=2, CA=3, DE=1."""
        solver, state = _make(2, 2, 3, 1)
        assert solver.evaluate(state) == "LOSING"

    def test_a2_losing(self):
        """A2: DE = sum(tri), not all equal, not special.
        AB=1, BC=2, CA=3 → sum=6. DE=6. Not all equal. Is {1,2,3} special for k=6?
        k=6, m=3 (6≤6≤9). Special: {7+4ℓ, 6+i+4ℓ, 6+5-i+4ℓ}={7,6+i,11-i} for ℓ=0.
        i=1:{7,7,10}. i=2:{7,8,9}. i=3:{7,9,8}={7,8,9}. i=4:{7,10,7}={7,7,10}.
        {1,2,3} ≠ any of these. So not special. → A2 applies → LOSING."""
        solver, state = _make(1, 2, 3, 6)
        assert solver.evaluate(state) == "LOSING"

    def test_d1_winning(self):
        """D1 sandwich: min(tri) ≤ k ≤ sum-min(tri).
        AB=2, BC=3, CA=5, DE=4. min=2, sum-min=8. 2≤4≤8 → WINNING."""
        solver, state = _make(2, 3, 5, 4)
        assert solver.evaluate(state) == "WINNING"
        move = solver.winning_move(state)
        assert move is not None
        assert move.is_valid(state)

    def test_all_equal_k_sum(self):
        """All triangle weights equal, k = sum → WINNING (A2 doesn't apply).
        AB=BC=CA=3, DE=9. All equal → not A2. {3,3,3} special for k=9?
        m=4 (10≤9? No). m=3: 6≤9≤9 → m=3. Special: {10+4ℓ, 9+i+4ℓ, 13-i+4ℓ}.
        For ℓ=0: {10, 9+i, 13-i}. {3,3,3} ≠ any. Not special.
        But A2 requires not all equal. So neither A1 nor A2 → WINNING."""
        solver, state = _make(3, 3, 3, 9)
        assert solver.evaluate(state) == "WINNING"

    def test_winning_move_leads_to_losing(self):
        """Verify winning move creates a losing position."""
        solver, state = _make(2, 3, 5, 4)
        move = solver.winning_move(state)
        assert move is not None
        new_state = state.apply_move(move)
        new_solver = G4Solver()
        result = new_solver.evaluate(new_state)
        assert result == "LOSING", f"Expected LOSING after winning move, got {result}"
