"""Tests for H1 solver (Path + isolated edge).

Tests cover Lemmas 4.7, 4.9, 4.11 and Theorems 4.8, 4.10.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from graph_state import GraphState
from classifier import classify, remap_state
from solvers.h1 import (
    H1Solver, f, block_size, decompose,
    check_lemma_4_7, check_theorem_4_8, check_lemma_4_9,
    check_lemma_4_11, check_theorem_4_10,
)


def _make(ab, bc, cd, ef):
    """Create H1 state: path A-B-C-D + isolated edge E-F."""
    state = GraphState.create(
        ["A", "B", "C", "D", "E", "F"],
        [("A", "B"), ("B", "C"), ("C", "D"), ("E", "F")],
        {("A", "B"): ab, ("B", "C"): bc, ("C", "D"): cd, ("E", "F"): ef},
    )
    gtype, label_map = classify(state)
    assert gtype == "H1"
    canonical = remap_state(state, label_map)
    return H1Solver(), canonical


class TestHelpers:
    """Test f(k), block_size, decompose."""

    def test_f_values(self):
        assert f(1) == 0
        assert f(2) == 1
        assert f(3) == 1
        assert f(4) == 2
        assert f(5) == 2
        assert f(7) == 2
        assert f(8) == 3
        assert f(15) == 3
        assert f(16) == 4

    def test_f_zero_raises(self):
        with pytest.raises(ValueError):
            f(0)

    def test_block_size(self):
        assert block_size(1) == 2
        assert block_size(2) == 4
        assert block_size(3) == 4
        assert block_size(4) == 8
        assert block_size(7) == 8
        assert block_size(8) == 16
        assert block_size(15) == 16
        assert block_size(16) == 32

    def test_decompose(self):
        assert decompose(10, 3) == (2, 2)  # 10 = 4*2 + 2
        assert decompose(7, 1) == (3, 1)   # 7 = 2*3 + 1
        assert decompose(0, 5) == (0, 0)
        assert decompose(16, 4) == (2, 0)  # 16 = 8*2 + 0


class TestLemma47:
    """Lemma 4.7: sandwich conditions."""

    def test_ab_le_k_le_ab_plus_bc(self):
        """w(AB)=2, w(BC)=3, w(CD)=5, w(EF)=4. AB=2 ≤ 4 ≤ 5=AB+BC → WINNING."""
        assert check_lemma_4_7(2, 3, 5, 4) == True

    def test_cd_le_k_le_bc_plus_cd(self):
        """w(AB)=10, w(BC)=3, w(CD)=2, w(EF)=4. CD=2 ≤ 4 ≤ 5=BC+CD → WINNING."""
        assert check_lemma_4_7(10, 3, 2, 4) == True

    def test_not_sandwich(self):
        """Neither condition holds."""
        assert check_lemma_4_7(10, 1, 20, 5) == False


class TestTheorem48:
    """Theorem 4.8: binary parity."""

    def test_b1_s_empty(self):
        """S empty and BC ≥ 1: k=5=101, AB=3=011, CD=6=110.
        Bits: a=(1,0,1), b=(0,1,1), c=(0,1,1). (reversed for bit positions 0,1,2)
        Actually: k=5=101, AB=3=011, CD=6=110.
        pos 0: 1+1+0=2 (even). pos 1: 0+1+1=2 (even). pos 2: 1+0+1=2 (even).
        S = empty. BC≥1 → WINNING."""
        assert check_theorem_4_8(3, 1, 6, 5) == True

    def test_b2_b_i_equals_1(self):
        """S non-empty with b_I=1."""
        # k=4=100, AB=3=011, CD=4=100
        # pos 0: 0+1+0=1 (odd) → in S
        # pos 1: 0+1+0=1 (odd) → in S
        # pos 2: 1+0+1=2 (even)
        # I = max(S) = 1, b_1 = 1 → WINNING
        assert check_theorem_4_8(3, 0, 4, 4) == True

    def test_not_winning(self):
        """S non-empty but b_I=c_I=0 → NOT winning by this theorem.
        This would mean a_I=1 (since sum is odd) and b_I=c_I=0.
        k=4=100, AB=0=000, CD=0=000 → but these must be positive.
        k=1=1, AB=2=10, CD=2=10.
        pos 0: 1+0+0=1 (odd). I=0, b_0=0, c_0=0 → not winning."""
        assert check_theorem_4_8(2, 1, 2, 1) == False


class TestLemma49:
    """Lemma 4.9: quotient/remainder mismatch."""

    def test_m1_ne_m2(self):
        """m1 ≠ m2 → WINNING."""
        # k=3, block=4. AB=5=4*1+1, CD=1=4*0+1. m1=1≠m2=0.
        assert check_lemma_4_9(5, 0, 1, 3) == True

    def test_min_ell_ge_k(self):
        """min(ℓ1,ℓ2) ≥ k → WINNING."""
        # k=2, block=4. AB=7=4*1+3, CD=6=4*1+2. m1=m2=1, ℓ1=3,ℓ2=2, min=2≥2.
        assert check_lemma_4_9(7, 0, 6, 2) == True

    def test_k_in_ells(self):
        """k ∈ {ℓ1, ℓ2} with min>0 → WINNING."""
        # k=3, block=4. AB=7=4*1+3, CD=5=4*1+1. k=3=ℓ1, min(ℓ)=1>0.
        assert check_lemma_4_9(7, 0, 5, 3) == True


class TestLemma411:
    """Lemma 4.11: large BC."""

    def test_bc_gt_k(self):
        """w_BC > k → WINNING."""
        assert check_lemma_4_11(10, 6, 10, 5) == True

    def test_bc_gt_k_minus_ell_min(self):
        """m1=m2, ℓ_min < k, BC > k - ℓ_min → WINNING."""
        # k=5, block=8. AB=9=8*1+1, CD=10=8*1+2. m1=m2=1, ℓ1=1,ℓ2=2, min=1<5.
        # BC > 5-1=4, so BC=5 → WINNING.
        assert check_lemma_4_11(9, 5, 10, 5) == True


class TestTheorem410:
    """Theorem 4.10: losing conditions."""

    def test_part_i_r0(self):
        """Part (i) r=0: {AB,CD}={Bm, Bm+k-s}, BC=s.
        k=5, block=8, m=1. r=0, s=3: AB=8, CD=8+5-0-3=10, BC=3.
        Check: config (8, 3, 10, 5) → LOSING."""
        assert check_theorem_4_10(8, 3, 10, 5) == True

    def test_part_i_r1(self):
        """Part (i) r=1: {AB,CD}={Bm+1, Bm+k-1-s}, BC=s.
        k=5, block=8, m=0. r=1, s=1: AB=1, CD=5-1-1=3, BC=1.
        Config (1, 1, 3, 5) → LOSING."""
        assert check_theorem_4_10(1, 1, 3, 5) == True

    def test_part_i_r3(self):
        """Part (i) r=3: {AB,CD}={Bm+3, Bm+k-3-s}, BC=s.
        k=8, block=16, m=0. r=3, s=1: AB=3, CD=8-3-1=4, BC=1.
        Config (3, 1, 4, 8) → LOSING."""
        assert check_theorem_4_10(3, 1, 4, 8) == True

    def test_not_losing(self):
        """A winning configuration should not match Theorem 4.10."""
        # k=4, AB=2, BC=3, CD=5: Lemma 4.7 applies (AB=2 ≤ 4 ≤ 5=AB+BC) → WINNING
        assert check_theorem_4_10(2, 3, 5, 4) == False

    def test_open_case(self):
        """H1 open case: AB=5, BC=1, CD=6, EF=11 → must NOT be WINNING.
        From paper: this is losing but not covered by Theorem 4.10.
        Expect LOSING or UNKNOWN."""
        solver, state = _make(5, 1, 6, 11)
        result = solver.evaluate(state)
        assert result in ("LOSING", "UNKNOWN"), \
            f"Open case (5,1,6,11) should be LOSING or UNKNOWN, got {result}"


class TestH1Solver:
    """Integration tests for H1 solver."""

    def test_k_zero_winning(self):
        """k=0 → WINNING (Remark 4.12)."""
        # Can't use _make because EF=0 means edge doesn't exist in state
        # Let's create manually
        state = GraphState.create(
            ["A", "B", "C", "D", "E", "F"],
            [("A", "B"), ("B", "C"), ("C", "D")],
            {("A", "B"): 3, ("B", "C"): 2, ("C", "D"): 5},
        )
        solver = H1Solver()
        # Manually construct with EF=0 weight
        from graph_state import canonical_edge
        state2 = GraphState(
            vertices=tuple(sorted(["A", "B", "C", "D", "E", "F"])),
            edges=(("A", "B"), ("B", "C"), ("C", "D"), ("E", "F")),
            weights={("A", "B"): 3, ("B", "C"): 2, ("C", "D"): 5, ("E", "F"): 0},
        )
        # Actually k=0 means EF has weight 0. But the paper assumes positive weights.
        # Remark 4.12 handles this as a special case.
        # For our solver, if EF weight = 0, it's WINNING.
        assert solver.evaluate(state2) == "WINNING"

    def test_sandwich_winning(self):
        """Lemma 4.7 sandwich: AB=2, BC=3, CD=5, EF=4."""
        solver, state = _make(2, 3, 5, 4)
        assert solver.evaluate(state) == "WINNING"
        move = solver.winning_move(state)
        assert move is not None
        assert move.is_valid(state)

    def test_theorem_4_10_losing(self):
        """Theorem 4.10(i) r=0: (8, 3, 10, 5) → LOSING."""
        solver, state = _make(8, 3, 10, 5)
        assert solver.evaluate(state) == "LOSING"

    def test_winning_move_leads_to_losing(self):
        """After winning move, the resulting position should be LOSING."""
        solver, state = _make(2, 3, 5, 4)
        move = solver.winning_move(state)
        assert move is not None
        new_state = state.apply_move(move)
        # Re-evaluate with H1 solver or check directly
        new_solver = H1Solver()
        result = new_solver.evaluate(new_state)
        # The result should be LOSING (or at least not WINNING)
        assert result in ("LOSING", "UNKNOWN"), \
            f"After winning move, expected LOSING/UNKNOWN but got {result}"

    def test_boundary_k_values(self):
        """Test with k = 1, 2, 3, 4, 5, 7, 8, 15, 16 (block boundaries)."""
        solver = H1Solver()
        # k=1: block=2. Config (2,1,2,1) r=0,s=1 → LOSING
        _, state = _make(2, 1, 2, 1)
        assert solver.evaluate(state) == "LOSING"

        # k=2: block=4. Config (4,1,5,2) r=0,s=1 → AB=4=4*1+0, CD=5=4*1+1.
        # r=0, lmax=k-0-1=1. CD remainder=1. ✓ LOSING.
        _, state = _make(4, 1, 5, 2)
        assert solver.evaluate(state) == "LOSING"

    def test_bc_zero_galaxy(self):
        """When BC=0, reduces to galaxy {AB, CD, EF}."""
        state = GraphState(
            vertices=tuple(sorted(["A", "B", "C", "D", "E", "F"])),
            edges=(("A", "B"), ("B", "C"), ("C", "D"), ("E", "F")),
            weights={("A", "B"): 3, ("B", "C"): 0, ("C", "D"): 3, ("E", "F"): 3},
        )
        solver = H1Solver()
        # Galaxy {3, 3, 3}: 3^3^3 = 3 ≠ 0 → WINNING... wait
        # Actually BC=0 means the edge doesn't contribute.
        # But our state has BC in edges with weight 0.
        # Let's handle this: galaxy {AB=3, CD=3, EF=3}
        # 3^3=0, 0^3=3 → XOR=3 ≠ 0 → WINNING
        # Wait: 3 XOR 3 XOR 3 = (3 XOR 3) XOR 3 = 0 XOR 3 = 3. WINNING.
        result = solver.evaluate(state)
        # Should handle degenerate case
        assert result == "WINNING"
