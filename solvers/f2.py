"""F2 solver: Triangle + pendant Graph Nim.

Implements Theorem 4.3, arXiv:2509.05064v1:
An initial weight configuration (w(AB), w(BC), w(CD), w(DB)) on F₂
is losing iff w(BC) = w(DB) and w(CD) = w(AB) + w(BC).

Canonical labeling (from Theorem 4.3):
  Vertices: A, B, C, D
  Edges: AB (pendant), BC, CD, DB (triangle is B-C-D)
  B is degree 3 (incident to AB, BC, DB)
  A is degree 1 (pendant leaf)
"""

from __future__ import annotations
from typing import Literal, Optional

from graph_state import GraphState
from solvers.base import Move, Solver


class F2Solver(Solver):
    """Solver for F2 = Triangle + pendant.

    Implements Theorem 4.3, arXiv:2509.05064v1.
    """

    def evaluate(self, state: GraphState) -> Literal["WINNING", "LOSING", "UNKNOWN"]:
        """LOSING iff w(BC) = w(DB) and w(CD) = w(AB) + w(BC).

        Implements Theorem 4.3, arXiv:2509.05064v1.
        """
        ab = state.get_weight("A", "B")
        bc = state.get_weight("B", "C")
        cd = state.get_weight("C", "D")
        db = state.get_weight("D", "B")

        if bc == db and cd == ab + bc:
            return "LOSING"
        return "WINNING"

    def winning_move(self, state: GraphState) -> Optional[Move]:
        """Constructive winning move from proof of Theorem 4.3.

        From the paper's proof:
        Case 1: w(BC) ≠ w(DB). WLOG w(BC) > w(DB) (swap B↔D role if needed).
          If w(CD) >= w(DB) + w(AB): select C, reduce BC by (BC-DB), reduce CD by CD-(DB+AB).
          If w(CD) < w(DB) + w(AB): select B, reduce BC by (BC-DB), reduce AB by AB-(CD-DB).
        Case 2: w(BC) = w(DB) but w(CD) ≠ w(AB) + w(BC).
          If w(CD) > w(AB) + w(BC): select C or D, remove w(CD) - (AB+BC) from CD.
          If w(CD) < w(AB) + w(BC): select A or B, remove AB - (CD-BC) from AB.

        Implements constructive proof of Theorem 4.3, arXiv:2509.05064v1.
        """
        if self.evaluate(state) != "WINNING":
            return None

        ab = state.get_weight("A", "B")
        bc = state.get_weight("B", "C")
        cd = state.get_weight("C", "D")
        db = state.get_weight("D", "B")

        # Case 1: BC != DB
        if bc != db:
            if bc > db:
                # i = DB, j = BC, k = CD, ℓ = AB (from proof)
                i, j, k, ell = db, bc, cd, ab
                if k >= i + ell:
                    # Select C: reduce BC by (j-i), reduce CD by k-(i+ℓ)
                    reductions = {}
                    r_bc = j - i
                    r_cd = k - (i + ell)
                    if r_bc > 0:
                        reductions[("B", "C")] = r_bc
                    if r_cd > 0:
                        reductions[("C", "D")] = r_cd
                    if sum(reductions.values()) > 0:
                        return Move(vertex="C", reductions=reductions)
                else:
                    # Select B: reduce BC by (j-i), reduce AB by ℓ-(k-i)
                    reductions = {}
                    r_bc = j - i
                    r_ab = ell - (k - i)
                    if r_bc > 0:
                        reductions[("B", "C")] = r_bc
                    if r_ab > 0:
                        reductions[("A", "B")] = r_ab
                    if sum(reductions.values()) > 0:
                        return Move(vertex="B", reductions=reductions)
            else:
                # db > bc: symmetric case
                # Swap roles: i = BC, j = DB, k = CD, ℓ = AB
                i, j, k, ell = bc, db, cd, ab
                if k >= i + ell:
                    # Select D: reduce DB by (j-i), reduce CD by k-(i+ℓ)
                    reductions = {}
                    r_db = j - i
                    r_cd = k - (i + ell)
                    if r_db > 0:
                        reductions[("D", "B")] = r_db
                    if r_cd > 0:
                        reductions[("C", "D")] = r_cd
                    if sum(reductions.values()) > 0:
                        return Move(vertex="D", reductions=reductions)
                else:
                    # Select B: reduce DB by (j-i), reduce AB by ℓ-(k-i)
                    reductions = {}
                    r_db = j - i
                    r_ab = ell - (k - i)
                    if r_db > 0:
                        reductions[("D", "B")] = r_db
                    if r_ab > 0:
                        reductions[("A", "B")] = r_ab
                    if sum(reductions.values()) > 0:
                        return Move(vertex="B", reductions=reductions)

        # Case 2: BC == DB but CD != AB + BC
        elif cd != ab + bc:
            if cd > ab + bc:
                # Remove excess from CD: select C (touches BC and CD) or D (touches CD and DB)
                excess = cd - (ab + bc)
                return Move(vertex="C", reductions={("C", "D"): excess})
            else:
                # CD < AB + BC: reduce AB so new_AB = CD - BC
                excess = ab - (cd - bc)
                if cd > bc:
                    return Move(vertex="A", reductions={("A", "B"): excess})
                else:
                    # cd <= bc means CD - BC <= 0, need AB -> 0 and also adjust
                    # Actually if CD < BC, we need different approach:
                    # target: BC'=DB'=bc, CD'=AB'+BC => CD=AB'+bc => AB'=CD-bc
                    # If CD < bc, then AB' < 0, impossible with current BC.
                    # Need to reduce BC and DB together.
                    # Select B: reduce AB to 0, reduce BC by bc - cd + ab...
                    # Actually: target losing state has BC'=DB', CD'=AB'+BC'.
                    # We can choose any target. Let's try:
                    #   Set AB'=0 (remove all AB via vertex A or B)
                    #   Then need BC'=DB' and CD'=BC'
                    # Select B: reduce AB to 0, reduce BC to min(bc, db) — no wait
                    # Simpler: we want BC=DB (already true), CD = AB+BC.
                    # Since CD < AB+BC, we need to reduce AB by (AB+BC-CD) = AB-(CD-BC).
                    # If CD >= BC: reduce AB by AB-(CD-BC) via vertex A.
                    # If CD < BC: reduce AB fully and reduce BC by BC-(CD-AB)... hmm but then BC≠DB.
                    # Let's try: select D, reduce DB by (bc-cd), reduce CD stays.
                    # Then DB'=cd, BC=bc. Need BC'=DB'=cd and CD'=AB+cd.
                    # But CD' is still cd, not AB+cd. Doesn't work easily.
                    #
                    # Better: select B to reduce all 3 incident edges.
                    # target: BC'=DB'=x, AB'=y, CD'=y+x (CD unchanged)
                    # CD = cd, so y+x = cd. BC'=x <= bc, DB'=x <= db=bc, AB'=y <= ab.
                    # So x = cd - y, need x <= bc => cd - y <= bc => y >= cd - bc.
                    # Also y >= 0 and y <= ab. And x >= 0 => y <= cd.
                    # Pick y = max(0, cd - bc): then x = cd - y = min(cd, bc).
                    y = max(0, cd - bc)
                    x = cd - y
                    if y <= ab and x <= bc and x <= db:
                        reductions = {}
                        if ab - y > 0:
                            reductions[("A", "B")] = ab - y
                        if bc - x > 0:
                            reductions[("B", "C")] = bc - x
                        if db - x > 0:
                            reductions[("D", "B")] = db - x
                        if sum(reductions.values()) > 0:
                            return Move(vertex="B", reductions=reductions)

        return None
