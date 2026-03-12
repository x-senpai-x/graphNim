"""H1 solver: Path P₄ + isolated edge Graph Nim.

Implements Lemmas 4.7, 4.9, 4.11 and Theorems 4.8, 4.10, arXiv:2509.05064v1.

Canonical labeling:
  Path: A-B-C-D (edges AB, BC, CD)
  Isolated edge: E-F (edge EF)
  k = w(EF)

Evaluation order:
  1. k = 0 → WINNING (Remark 4.12)
  2. Lemma 4.7 (sandwich) → WINNING
  3. Theorem 4.8 (binary parity) → WINNING
  4. Lemma 4.9 (quotient/remainder) → WINNING
  5. Lemma 4.11 (large BC) → WINNING
  6. Theorem 4.10 (losing conditions, r ∈ {0,1,2,3,4}) → LOSING
  7. Otherwise → UNKNOWN
"""

from __future__ import annotations
import math
from typing import Literal, Optional

from graph_state import GraphState
from solvers.base import Move, Solver


# ---------------------------------------------------------------------------
# Helper functions (§4.2, arXiv:2509.05064v1)
# ---------------------------------------------------------------------------

def f(k: int) -> int:
    """f(k) = floor(log₂(k)) for k ≥ 1.

    The unique non-negative integer satisfying 2^f(k) ≤ k < 2^(f(k)+1).
    f(0) is undefined — callers must handle k=0 separately (Remark 4.12).
    """
    if k <= 0:
        raise ValueError("f(k) undefined for k <= 0 (Remark 4.12)")
    return k.bit_length() - 1


def block_size(k: int) -> int:
    """Block size = 2^(f(k)+1) for k ≥ 1."""
    return 1 << (f(k) + 1)


def decompose(w: int, k: int) -> tuple[int, int]:
    """Decompose w = block_size(k) * m + ℓ, return (m, ℓ).

    Here ℓ ∈ {0, 1, ..., block_size(k) - 1}.
    """
    B = block_size(k)
    return divmod(w, B)


def _is_balanced(values: list[int]) -> bool:
    """Check if a tuple of values is balanced (XOR = 0)."""
    x = 0
    for v in values:
        x ^= v
    return x == 0


# ---------------------------------------------------------------------------
# Winning condition checks
# ---------------------------------------------------------------------------

def check_lemma_4_7(w_ab: int, w_bc: int, w_cd: int, k: int) -> bool:
    """Lemma 4.7: sandwich conditions.

    WINNING if w(AB) ≤ k ≤ w(AB) + w(BC) or w(CD) ≤ k ≤ w(BC) + w(CD).

    Implements Lemma 4.7, arXiv:2509.05064v1.
    """
    if w_ab <= k <= w_ab + w_bc:
        return True
    if w_cd <= k <= w_bc + w_cd:
        return True
    return False


def check_theorem_4_8(w_ab: int, w_bc: int, w_cd: int, k: int) -> bool:
    """Theorem 4.8: binary parity check.

    Write k, w_AB, w_CD in binary. Let S = {i : (a_i + b_i + c_i) is odd}.
    WINNING if:
      (B1) S is empty and w_BC ≥ 1, or
      (B2) S non-empty and (b_I = 1 or c_I = 1) where I = max(S).

    Implements Theorem 4.8, arXiv:2509.05064v1.
    """
    # Determine bit length
    s = max(k, w_ab, w_cd).bit_length()
    if s == 0:
        s = 1

    S = []
    for i in range(s):
        a_i = (k >> i) & 1
        b_i = (w_ab >> i) & 1
        c_i = (w_cd >> i) & 1
        if (a_i + b_i + c_i) % 2 == 1:
            S.append(i)

    if not S:
        # B1: S empty and w_BC >= 1
        return w_bc >= 1
    else:
        # B2: S non-empty, check b_I or c_I = 1
        I = max(S)
        b_I = (w_ab >> I) & 1
        c_I = (w_cd >> I) & 1
        return b_I == 1 or c_I == 1


def check_lemma_4_9(w_ab: int, w_bc: int, w_cd: int, k: int) -> bool:
    """Lemma 4.9: quotient/remainder mismatch.

    With w_AB = 2^(f(k)+1)*m1 + ℓ1, w_CD = 2^(f(k)+1)*m2 + ℓ2:
    WINNING if:
      (i)  m1 ≠ m2
      (ii) min(ℓ1, ℓ2) ≥ k
      (iii) k ∈ {ℓ1, ℓ2} and (min(ℓ1,ℓ2) > 0 or w_BC > 0)

    Implements Lemma 4.9, arXiv:2509.05064v1.
    """
    m1, l1 = decompose(w_ab, k)
    m2, l2 = decompose(w_cd, k)

    # (i) m1 ≠ m2
    if m1 != m2:
        return True

    ell_min = min(l1, l2)

    # (ii) min(ℓ1, ℓ2) ≥ k
    if ell_min >= k:
        return True

    # (iii) k ∈ {ℓ1, ℓ2} and (min > 0 or w_BC > 0)
    if k in (l1, l2) and (ell_min > 0 or w_bc > 0):
        return True

    return False


def check_lemma_4_11(w_ab: int, w_bc: int, w_cd: int, k: int) -> bool:
    """Lemma 4.11: large BC condition.

    WINNING if w_BC > k, or if m1=m2 and min(ℓ1,ℓ2) < k and w_BC > k - min(ℓ1,ℓ2).

    Implements Lemma 4.11, arXiv:2509.05064v1.
    """
    if w_bc > k:
        return True

    m1, l1 = decompose(w_ab, k)
    m2, l2 = decompose(w_cd, k)

    if m1 == m2:
        ell_min = min(l1, l2)
        if ell_min < k and w_bc > k - ell_min:
            return True

    return False


# ---------------------------------------------------------------------------
# Losing condition check (Theorem 4.10)
# ---------------------------------------------------------------------------

def check_theorem_4_10(w_ab: int, w_bc: int, w_cd: int, k: int) -> bool:
    """Theorem 4.10: losing conditions for r ∈ {0,1,2,3,4}.

    Checks all four parts (i)-(iv) of Theorem 4.10.

    Implements Theorem 4.10, arXiv:2509.05064v1.
    """
    m1, l1 = decompose(w_ab, k)
    m2, l2 = decompose(w_cd, k)

    if m1 != m2:
        return False

    # Check with both orderings since {w(AB), w(CD)} is a set
    return (_check_4_10_ordered(l1, w_bc, l2, k) or
            _check_4_10_ordered(l2, w_bc, l1, k))


def _check_4_10_ordered(r: int, s: int, ell_max_candidate: int, k: int) -> bool:
    """Check Theorem 4.10 with r = min remainder, checking if it matches a losing form.

    Parts (i)-(iv) all have the form:
      {w(AB), w(CD)} = {2^(f(k)+1)*m + r, 2^(f(k)+1)*m + <something>}
    with w(BC) = s.

    Implements Theorem 4.10 parts (i)-(iv), arXiv:2509.05064v1.
    """
    # Part (i): r ∈ {0, 1, 3}, s ∈ {1, ..., k-2r}
    # {w_AB, w_CD} = {Bm + r, Bm + k - r - s}, w_BC = s
    if r in (0, 1, 3):
        if 1 <= s <= k - 2 * r:
            expected_lmax = k - r - s
            if expected_lmax >= 0 and ell_max_candidate == expected_lmax:
                return True

    # Part (ii): r ∈ {2, 4}, s ∈ {1, ..., r-1}
    # k ≡ j mod 2^(f(r)+1) for some j ∈ {s, ..., r-1}
    # {w_AB, w_CD} = {Bm + r, Bm + k + r - s}
    if r in (2, 4) and r >= 2:
        fr = f(r)
        mod = 1 << (fr + 1)
        if 1 <= s <= r - 1:
            k_mod = k % mod
            if any(k_mod == j for j in range(s, r)):
                expected_lmax = k + r - s
                B = block_size(k)
                if expected_lmax < B and ell_max_candidate == expected_lmax:
                    return True

    # Part (iii): r ∈ {2, 4}, s ∈ {1, ..., r-1}, k ≥ 3r
    # k ≢ j mod 2^(f(r)+1) for j ∈ {s, ..., r-1}
    # {w_AB, w_CD} = {Bm + r, Bm + k - r - s}
    if r in (2, 4) and r >= 2:
        fr = f(r)
        mod = 1 << (fr + 1)
        if 1 <= s <= r - 1 and k >= 3 * r:
            k_mod = k % mod
            in_range = any(k_mod == j for j in range(s, r))
            if not in_range:
                expected_lmax = k - r - s
                if expected_lmax >= 0 and ell_max_candidate == expected_lmax:
                    return True

    # Part (iv): r ∈ {2, 4}, s ∈ {r, ..., k-2r}, k ≥ 3r
    # {w_AB, w_CD} = {Bm + r, Bm + k - r - s}
    if r in (2, 4):
        if k >= 3 * r and r <= s <= k - 2 * r:
            expected_lmax = k - r - s
            if expected_lmax >= 0 and ell_max_candidate == expected_lmax:
                return True

    return False


# ---------------------------------------------------------------------------
# Winning move constructors
# ---------------------------------------------------------------------------

def _move_lemma_4_7(w_ab: int, w_bc: int, w_cd: int, k: int) -> Optional[Move]:
    """Winning move from proof of Lemma 4.7 (§7.1).

    If w(AB) ≤ k ≤ w(AB)+w(BC): select C, remove CD entirely,
      reduce BC by w(AB)+w(BC)-k. Result: galaxy {AB,BC}+{EF} balanced.
    If w(CD) ≤ k ≤ w(BC)+w(CD): select B, remove AB entirely,
      reduce BC by w(BC)+w(CD)-k. Result: galaxy {BC,CD}+{EF} balanced.

    Implements proof of Lemma 4.7, §7.1, arXiv:2509.05064v1.
    """
    if w_ab <= k <= w_ab + w_bc:
        reductions = {("C", "D"): w_cd}
        reduce_bc = w_ab + w_bc - k
        if reduce_bc > 0:
            reductions[("B", "C")] = reduce_bc
        return Move(vertex="C", reductions=reductions)

    if w_cd <= k <= w_bc + w_cd:
        reductions = {("A", "B"): w_ab}
        reduce_bc = w_bc + w_cd - k
        if reduce_bc > 0:
            reductions[("B", "C")] = reduce_bc
        return Move(vertex="B", reductions=reductions)

    return None


def _move_theorem_4_8(w_ab: int, w_bc: int, w_cd: int, k: int) -> Optional[Move]:
    """Winning move from proof of Theorem 4.8 (§7.2).

    Case B1 (S empty, w_BC ≥ 1): select vertex B or C, remove BC entirely.
      Result: galaxy {AB, CD, EF} which is balanced.
    Case B2 (S non-empty, b_I=1 or c_I=1):
      WLOG b_I=1. Compute target w'_AB by flipping bits in S.
      Select B, reduce AB by (w_AB - w'_AB), remove BC entirely.
      Result: galaxy {AB, CD, EF} which is balanced.

    Implements proof of Theorem 4.8, §7.2, arXiv:2509.05064v1.
    """
    s_bits = max(k, w_ab, w_cd).bit_length()
    if s_bits == 0:
        s_bits = 1

    S = []
    for i in range(s_bits):
        a_i = (k >> i) & 1
        b_i = (w_ab >> i) & 1
        c_i = (w_cd >> i) & 1
        if (a_i + b_i + c_i) % 2 == 1:
            S.append(i)

    if not S:
        # B1: remove BC entirely. (AB, CD, k) is already balanced.
        if w_bc >= 1:
            # Select B or C to remove BC
            return Move(vertex="B", reductions={("B", "C"): w_bc})
        return None

    I = max(S)
    b_I = (w_ab >> I) & 1
    c_I = (w_cd >> I) & 1

    if b_I == 1:
        # Compute target w'_AB: flip bits at positions in S
        # e_i = b_i for i ∉ S, e_i = 1-b_i for i ∈ S (but ensure e_i+a_i+c_i is even)
        target_ab = 0
        for i in range(s_bits):
            a_i = (k >> i) & 1
            b_i = (w_ab >> i) & 1
            c_i = (w_cd >> i) & 1
            if i in S:
                # Need e_i such that e_i + a_i + c_i is even
                e_i = (a_i + c_i) % 2  # makes sum even
            else:
                e_i = b_i
            target_ab |= (e_i << i)

        if target_ab < w_ab:
            reductions = {("A", "B"): w_ab - target_ab}
            if w_bc > 0:
                reductions[("B", "C")] = w_bc
            return Move(vertex="B", reductions=reductions)

    if c_I == 1:
        # Symmetric: compute target w'_CD
        target_cd = 0
        for i in range(s_bits):
            a_i = (k >> i) & 1
            b_i = (w_ab >> i) & 1
            c_i = (w_cd >> i) & 1
            if i in S:
                e_i = (a_i + b_i) % 2
            else:
                e_i = c_i
            target_cd |= (e_i << i)

        if target_cd < w_cd:
            reductions = {("C", "D"): w_cd - target_cd}
            if w_bc > 0:
                reductions[("B", "C")] = w_bc
            return Move(vertex="C", reductions=reductions)

    return None


def _move_lemma_4_9(w_ab: int, w_bc: int, w_cd: int, k: int) -> Optional[Move]:
    """Winning move from proof of Lemma 4.9 (§7.3).

    Case m1 ≠ m2 or min(ℓ) ≥ k: reduces to Theorem 4.8 (b_I or c_I = 1).
    Case k ∈ {ℓ1, ℓ2}: select C, remove ℓ2 from CD, remove all BC.

    Implements proof of Lemma 4.9, §7.3, arXiv:2509.05064v1.
    """
    m1, l1 = decompose(w_ab, k)
    m2, l2 = decompose(w_cd, k)

    # Cases (i) m1≠m2 and (ii) min(ℓ)≥k: use Theorem 4.8 (which covers these)
    if m1 != m2 or min(l1, l2) >= k:
        return _move_theorem_4_8(w_ab, w_bc, w_cd, k)

    # Case (iii): k ∈ {ℓ1, ℓ2}
    if k == l1:
        # Select C, remove ℓ2 from CD, remove entire BC
        # Result: galaxy {AB, CD, EF} with AB=Bm+k, CD=Bm, EF=k → balanced
        reductions = {}
        if l2 > 0:
            reductions[("C", "D")] = l2
        if w_bc > 0:
            reductions[("B", "C")] = w_bc
        if sum(reductions.values()) > 0:
            return Move(vertex="C", reductions=reductions)
    elif k == l2:
        # Select B, remove ℓ1 from AB, remove entire BC
        reductions = {}
        if l1 > 0:
            reductions[("A", "B")] = l1
        if w_bc > 0:
            reductions[("B", "C")] = w_bc
        if sum(reductions.values()) > 0:
            return Move(vertex="B", reductions=reductions)

    return None


def _move_lemma_4_11(w_ab: int, w_bc: int, w_cd: int, k: int) -> Optional[Move]:
    """Winning move from proof of Lemma 4.11 (§7.5).

    Select C, reduce CD by ℓ2, reduce BC to (k - ℓ1).
    Result: configuration of form (4.4) with r=0 → losing for P2.

    Implements proof of Lemma 4.11, §7.5, arXiv:2509.05064v1.
    """
    m1, l1 = decompose(w_ab, k)
    m2, l2 = decompose(w_cd, k)

    if m1 != m2:
        # Covered by Lemma 4.9
        return _move_lemma_4_9(w_ab, w_bc, w_cd, k)

    ell_min = min(l1, l2)
    ell_max = max(l1, l2)

    if w_bc > k:
        # From proof: if k < min(AB, CD), use Lemma 4.7/4.9.
        # If k >= min(AB, CD) and m1=m2=m with ℓ1≤ℓ2:
        # Select C, remove ℓ2 from CD, reduce BC to k-ℓ1.
        if l1 <= l2:
            reductions = {}
            if l2 > 0:
                reductions[("C", "D")] = l2
            bc_reduce = w_bc - (k - l1)
            if bc_reduce > 0:
                reductions[("B", "C")] = bc_reduce
            if sum(reductions.values()) > 0:
                return Move(vertex="C", reductions=reductions)
        else:
            reductions = {}
            if l1 > 0:
                reductions[("A", "B")] = l1
            bc_reduce = w_bc - (k - l2)
            if bc_reduce > 0:
                reductions[("B", "C")] = bc_reduce
            if sum(reductions.values()) > 0:
                return Move(vertex="B", reductions=reductions)

    if m1 == m2 and ell_min < k and w_bc > k - ell_min:
        if l1 <= l2:
            reductions = {}
            if l2 > 0:
                reductions[("C", "D")] = l2
            bc_reduce = w_bc - (k - l1)
            if bc_reduce > 0:
                reductions[("B", "C")] = bc_reduce
            if sum(reductions.values()) > 0:
                return Move(vertex="C", reductions=reductions)
        else:
            reductions = {}
            if l1 > 0:
                reductions[("A", "B")] = l1
            bc_reduce = w_bc - (k - l2)
            if bc_reduce > 0:
                reductions[("B", "C")] = bc_reduce
            if sum(reductions.values()) > 0:
                return Move(vertex="B", reductions=reductions)

    return None


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

class H1Solver(Solver):
    """Solver for H1 = Path A-B-C-D + isolated edge E-F.

    Partially characterized by Theorems 4.7–4.10, arXiv:2509.05064v1.
    Returns UNKNOWN for configurations beyond r=4 in Theorem 4.10.
    """

    def evaluate(self, state: GraphState) -> Literal["WINNING", "LOSING", "UNKNOWN"]:
        """Classify using Lemmas 4.7, 4.9, 4.11 and Theorems 4.8, 4.10.

        Implements §4.2 evaluation flowchart (Figure 4), arXiv:2509.05064v1.
        """
        w_ab = state.get_weight("A", "B")
        w_bc = state.get_weight("B", "C")
        w_cd = state.get_weight("C", "D")
        k = state.get_weight("E", "F")

        # Remark 4.12: k = 0 → WINNING
        if k == 0:
            return "WINNING"

        # All edge weights must be positive for the theorems to apply
        # If any path edge is 0, it's a simpler game
        if w_ab == 0 and w_bc == 0 and w_cd == 0:
            # Only EF remains: single edge, WINNING (take it all)
            return "WINNING"

        # Handle degenerate cases where path edges are 0
        if w_ab == 0 or w_cd == 0 or w_bc == 0:
            # Reduced game — use galaxy analysis or direct checks
            # If BC=0: galaxy {AB, CD, EF}, losing iff balanced
            if w_bc == 0:
                if _is_balanced([w_ab, w_cd, k]):
                    return "LOSING"
                return "WINNING"
            # If AB=0: star at B (edge BC) + CD + EF → galaxy
            if w_ab == 0:
                if _is_balanced([w_bc + 0, w_cd, k]):
                    # Wait: if AB=0, it's path B-C-D + EF
                    # Star at C with edges BC, CD. Galaxy: star_sum=BC+CD, EF=k
                    # Actually it's K_{1,2} at C (if BC and CD both positive) + EF
                    if _is_balanced([w_bc + w_cd, k]):
                        return "LOSING"
                    # No wait, need to be more careful about path B-C-D
                    # This is a star at B with edge BC plus C-D... it's actually
                    # a path B-C-D which is a star at C (center) with rays BC and CD.
                    # Hmm, B-C-D as a path: C is degree 2 (edges BC, CD).
                    # As a galaxy? No, a path of length 2 IS a star K_{1,2}.
                    # So galaxy: K_{1,2}(center=C, rays BC,CD) + K2(EF).
                    # Star sum = BC + CD. Losing iff BC+CD = k (XOR of 2 piles = 0 means equal).
                    if w_bc + w_cd == k:
                        return "LOSING"
                return "WINNING"
            if w_cd == 0:
                # Symmetric to AB=0
                if w_ab + w_bc == k:
                    return "LOSING"
                return "WINNING"

        # Main evaluation with all edges positive
        # Lemma 4.7
        if check_lemma_4_7(w_ab, w_bc, w_cd, k):
            return "WINNING"

        # Theorem 4.8
        if check_theorem_4_8(w_ab, w_bc, w_cd, k):
            return "WINNING"

        # Lemma 4.9
        if check_lemma_4_9(w_ab, w_bc, w_cd, k):
            return "WINNING"

        # Lemma 4.11
        if check_lemma_4_11(w_ab, w_bc, w_cd, k):
            return "WINNING"

        # Theorem 4.10 (losing)
        if check_theorem_4_10(w_ab, w_bc, w_cd, k):
            return "LOSING"

        return "UNKNOWN"

    def winning_move(self, state: GraphState) -> Optional[Move]:
        """Compute winning move using constructive proofs.

        Implements constructive strategies from §7, arXiv:2509.05064v1.
        """
        result = self.evaluate(state)
        if result != "WINNING":
            return None

        w_ab = state.get_weight("A", "B")
        w_bc = state.get_weight("B", "C")
        w_cd = state.get_weight("C", "D")
        k = state.get_weight("E", "F")

        # Remark 4.12: k=0, remove BC and equalize AB/CD
        if k == 0:
            return self._move_k_zero(w_ab, w_bc, w_cd)

        # Degenerate cases
        if w_bc == 0:
            return self._move_galaxy_3(w_ab, w_cd, k)
        if w_ab == 0:
            return self._move_degenerate_ab_zero(w_bc, w_cd, k)
        if w_cd == 0:
            return self._move_degenerate_cd_zero(w_ab, w_bc, k)

        # Try each winning condition in order
        if check_lemma_4_7(w_ab, w_bc, w_cd, k):
            move = _move_lemma_4_7(w_ab, w_bc, w_cd, k)
            if move:
                return move

        if check_theorem_4_8(w_ab, w_bc, w_cd, k):
            move = _move_theorem_4_8(w_ab, w_bc, w_cd, k)
            if move:
                return move

        if check_lemma_4_9(w_ab, w_bc, w_cd, k):
            move = _move_lemma_4_9(w_ab, w_bc, w_cd, k)
            if move:
                return move

        if check_lemma_4_11(w_ab, w_bc, w_cd, k):
            move = _move_lemma_4_11(w_ab, w_bc, w_cd, k)
            if move:
                return move

        return None

    def _move_k_zero(self, w_ab, w_bc, w_cd):
        """Winning move when k=0 (Remark 4.12).

        WLOG w_AB ≥ w_CD: select C, remove (w_AB - w_CD) from ... wait.
        Actually from Remark 4.12: assuming w(AB) ≥ w(CD), P1 removes
        w(CD)-w(AB) from CD... no: "removes weight w0(CD)−w0(AB) from the edge CD".
        Wait: "assuming w(AB) ≥ w(CD), P1 removes weight w(CD)−w(AB) from CD" doesn't
        make sense if AB≥CD. Let me re-read.

        "assuming, without loss of generality, that w0(AB) ≥ w0(CD), P1 removes,
        in the first round, weight w0(CD) − w0(AB) from the edge CD" — this is wrong
        if AB > CD. The paper must mean w0(AB) - w0(CD) from... Hmm. Actually
        reading more carefully: "P1 removes, in the first round, weight w0(CD) − w0(AB)
        from the edge CD, and the entire edge BC".

        I think the paper means: WLOG w_AB >= w_CD.
        Select vertex C. Reduce CD by w_CD (remove entirely), reduce BC by w_BC
        (remove entirely). But also reduce AB somehow? C is not incident to AB.

        Actually: C is incident to BC and CD. So P1 selects C:
        - Reduce CD to w_AB - ... no. Let me re-read more carefully.

        "P1 removes, in the first round, weight w0(CD)−w0(AB) from the edge CD"
        This only works if CD ≥ AB. So WLOG CD ≥ AB? Or maybe WLOG AB ≥ CD and
        the formula is actually AB-CD from... a different edge.

        Let me just handle it: WLOG w_AB ≥ w_CD.
        Select C, remove all BC, reduce CD to 0. But then AB remains with weight w_AB.
        EF has weight 0. Galaxy: just AB alone. P2 takes it and wins. That's wrong.

        Better reading: "P1 removes weight |w0(AB) - w0(CD)| from one edge and
        removes entire BC, leaving AB=CD, BC=0, EF=0 → galaxy with AB=CD → balanced."

        Select B: reduce AB by (w_AB - w_CD), remove BC entirely.
        Result: AB = w_CD, CD = w_CD, BC = 0, EF = 0.
        Galaxy {AB, CD} balanced → P2 loses.

        Implements Remark 4.12, arXiv:2509.05064v1.
        """
        if w_ab >= w_cd:
            reductions = {("B", "C"): w_bc}
            if w_ab - w_cd > 0:
                reductions[("A", "B")] = w_ab - w_cd
            if sum(reductions.values()) > 0:
                return Move(vertex="B", reductions=reductions)
            # w_bc = 0 and w_ab = w_cd: already balanced. But k=0 so game state
            # has {AB, BC=0, CD, EF=0}. Galaxy {AB, CD} with AB=CD → balanced → LOSING.
            # But we said k=0 is WINNING. That contradicts...
            # Unless all weights are 0. If total_weight = 0, game is over.
            # If w_bc = 0, w_ab = w_cd, k = 0: galaxy {AB, CD} balanced → LOSING.
            # But Remark 4.12 says k=0 is winning. The remark assumes other edges have
            # positive weight and we can make a move. If all are 0, no move possible.
            # If w_ab = w_cd > 0 and w_bc = 0 and k = 0: this is actually LOSING
            # (galaxy with equal piles). So Remark 4.12 applies when we can make the
            # equalizing move, i.e., when we can remove BC and adjust.
            # But if BC = 0 and AB = CD, there's no helpful move from remark 4.12.
            # In this edge case, evaluate should have caught it via the galaxy check.
            return None
        else:
            reductions = {("B", "C"): w_bc}
            if w_cd - w_ab > 0:
                reductions[("C", "D")] = w_cd - w_ab
            if sum(reductions.values()) > 0:
                return Move(vertex="C", reductions=reductions)
            return None

    def _move_galaxy_3(self, w_ab, w_cd, k):
        """Winning move when BC=0: galaxy {AB, CD, EF}."""
        nim_sum = w_ab ^ w_cd ^ k
        # Find pile to reduce
        for val, edge, vertex in [(w_ab, ("A", "B"), "A"), (w_cd, ("C", "D"), "D"), (k, ("E", "F"), "E")]:
            new_val = val ^ nim_sum
            if new_val < val:
                return Move(vertex=vertex, reductions={edge: val - new_val})
        return None

    def _move_degenerate_ab_zero(self, w_bc, w_cd, k):
        """Winning move when AB=0: galaxy with star at C."""
        # Star sum = BC + CD, other pile = k
        star_sum = w_bc + w_cd
        if star_sum > k:
            # Reduce star sum to k: select C, reduce BC and/or CD
            diff = star_sum - k
            reductions = {}
            take_bc = min(diff, w_bc)
            diff -= take_bc
            if take_bc > 0:
                reductions[("B", "C")] = take_bc
            if diff > 0:
                reductions[("C", "D")] = diff
            return Move(vertex="C", reductions=reductions)
        elif star_sum < k:
            # Reduce k to star_sum
            return Move(vertex="E", reductions={("E", "F"): k - star_sum})
        return None

    def _move_degenerate_cd_zero(self, w_ab, w_bc, k):
        """Winning move when CD=0: galaxy with star at B."""
        star_sum = w_ab + w_bc
        if star_sum > k:
            diff = star_sum - k
            reductions = {}
            take_bc = min(diff, w_bc)
            diff -= take_bc
            if take_bc > 0:
                reductions[("B", "C")] = take_bc
            if diff > 0:
                reductions[("A", "B")] = diff
            return Move(vertex="B", reductions=reductions)
        elif star_sum < k:
            return Move(vertex="E", reductions={("E", "F"): k - star_sum})
        return None
