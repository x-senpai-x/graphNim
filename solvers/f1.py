"""F1 solver: C₄ (4-cycle) Graph Nim.

Implements Theorem 4.2 (Theorem 2 of [23]), arXiv:2509.05064v1:
An initial weight configuration (w(AB), w(BC), w(CD), w(DA)) on C₄
is losing if and only if w(AB) = w(CD) and w(BC) = w(DA).

Canonical labeling: cycle A-B-C-D-A with edges AB, BC, CD, DA.
"""

from __future__ import annotations
from typing import Literal, Optional

from graph_state import GraphState
from solvers.base import Move, Solver


class F1Solver(Solver):
    """Solver for F1 = C₄ (4-cycle).

    Implements Theorem 4.2, arXiv:2509.05064v1.
    """

    def evaluate(self, state: GraphState) -> Literal["WINNING", "LOSING", "UNKNOWN"]:
        """LOSING iff opposite edges are equal: w(AB)=w(CD) and w(BC)=w(DA).

        Implements Theorem 4.2, arXiv:2509.05064v1.
        """
        ab = state.get_weight("A", "B")
        bc = state.get_weight("B", "C")
        cd = state.get_weight("C", "D")
        da = state.get_weight("D", "A")

        if ab == cd and bc == da:
            return "LOSING"
        return "WINNING"

    def winning_move(self, state: GraphState) -> Optional[Move]:
        """Find a move to reach a losing position (opposite edges equal).

        Strategy: select a vertex touching two adjacent edges, adjust them
        so that opposite pairs become equal.

        Implements constructive proof of Theorem 4.2, arXiv:2509.05064v1.
        """
        if self.evaluate(state) != "WINNING":
            return None

        ab = state.get_weight("A", "B")
        bc = state.get_weight("B", "C")
        cd = state.get_weight("C", "D")
        da = state.get_weight("D", "A")

        # We need to reach: AB'=CD' and BC'=DA'.
        # Try selecting each vertex and see if we can reach a losing state.

        # Select vertex B (touches AB and BC):
        # Can reduce AB by x and BC by y (x+y > 0).
        # Target: (AB-x) = CD and (BC-y) = DA
        # => x = AB - CD, y = BC - DA
        x = ab - cd
        y = bc - da
        if x >= 0 and y >= 0 and x + y > 0 and x <= ab and y <= bc:
            reductions = {}
            if x > 0:
                reductions[("A", "B")] = x
            if y > 0:
                reductions[("B", "C")] = y
            return Move(vertex="B", reductions=reductions)

        # Select vertex D (touches CD and DA):
        # Target: CD-x = AB, DA-y = BC => x = CD-AB, y = DA-BC
        x = cd - ab
        y = da - bc
        if x >= 0 and y >= 0 and x + y > 0 and x <= cd and y <= da:
            reductions = {}
            if x > 0:
                reductions[("C", "D")] = x
            if y > 0:
                reductions[("D", "A")] = y
            return Move(vertex="D", reductions=reductions)

        # Select vertex A (touches DA and AB):
        # Target: DA-x = BC, AB-y = CD => x = DA-BC, y = AB-CD
        x = da - bc
        y = ab - cd
        if x >= 0 and y >= 0 and x + y > 0 and x <= da and y <= ab:
            reductions = {}
            if x > 0:
                reductions[("D", "A")] = x
            if y > 0:
                reductions[("A", "B")] = y
            return Move(vertex="A", reductions=reductions)

        # Select vertex C (touches BC and CD):
        # Target: BC-x = DA, CD-y = AB => x = BC-DA, y = CD-AB
        x = bc - da
        y = cd - ab
        if x >= 0 and y >= 0 and x + y > 0 and x <= bc and y <= cd:
            reductions = {}
            if x > 0:
                reductions[("B", "C")] = x
            if y > 0:
                reductions[("C", "D")] = y
            return Move(vertex="C", reductions=reductions)

        return None
