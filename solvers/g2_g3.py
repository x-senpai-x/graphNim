"""G2 and G3 solver: all configurations winning.

Implements Theorem 4.4, arXiv:2509.05064v1:
Every initial weight configuration on G2 and G3 is winning.

G2 canonical labeling (from proof of Theorem 4.4):
  Vertices: A, B, C, D, E
  Edges: AB, AC, AD, BE
  A has degree 3 (edges AB, AC, AD). B has degree 2 (edges AB, BE).

G3 canonical labeling (from proof of Theorem 4.4):
  Vertices: A, B, C, D, E
  Edges: AB, BC, CD, DE (path of length 4)
"""

from __future__ import annotations
from typing import Literal, Optional

from graph_state import GraphState
from solvers.base import Move, Solver


class G2G3Solver(Solver):
    """Solver for G2 and G3 — all configurations winning.

    Implements Theorem 4.4, arXiv:2509.05064v1.
    """

    def __init__(self, graph_type: str):
        self.graph_type = graph_type

    def evaluate(self, state: GraphState) -> Literal["WINNING", "LOSING", "UNKNOWN"]:
        """All configurations are WINNING.

        Implements Theorem 4.4, arXiv:2509.05064v1.
        """
        return "WINNING"

    def winning_move(self, state: GraphState) -> Optional[Move]:
        """Constructive winning move from proof of Theorem 4.4.

        Implements constructive proof of Theorem 4.4, arXiv:2509.05064v1.
        """
        if self.graph_type == "G2":
            return self._g2_move(state)
        else:
            return self._g3_move(state)

    def _g2_move(self, state: GraphState) -> Optional[Move]:
        """Winning move for G2 (edges AB, AC, AD, BE).

        From proof of Theorem 4.4:
        (i) If w(BE) >= w(AC)+w(AD): select B, remove AB entirely,
            reduce BE to w(AC)+w(AD). Leaves balanced galaxy.
        (ii) If w(BE) < w(AC)+w(AD): select A, remove AB entirely,
             reduce AC/AD so w1(AC)+w1(AD) = w(BE). Leaves balanced galaxy.

        Implements constructive proof of Theorem 4.4 (G2 case), arXiv:2509.05064v1.
        """
        ab = state.get_weight("A", "B")
        ac = state.get_weight("A", "C")
        ad = state.get_weight("A", "D")
        be = state.get_weight("B", "E")

        if be >= ac + ad:
            # Select B: remove AB, reduce BE by (BE - (AC+AD))
            reductions = {("A", "B"): ab}
            reduce_be = be - (ac + ad)
            if reduce_be > 0:
                reductions[("B", "E")] = reduce_be
            return Move(vertex="B", reductions=reductions)
        else:
            # Select A: remove AB, reduce AC/AD so sum = BE
            target_sum = be
            # Reduce from larger edges first
            reductions = {("A", "B"): ab}
            current_sum = ac + ad
            to_remove = current_sum - target_sum
            # Remove from AD first, then AC
            remove_ad = min(to_remove, ad)
            to_remove -= remove_ad
            remove_ac = to_remove
            if remove_ad > 0:
                reductions[("A", "D")] = remove_ad
            if remove_ac > 0:
                reductions[("A", "C")] = remove_ac
            return Move(vertex="A", reductions=reductions)

    def _g3_move(self, state: GraphState) -> Optional[Move]:
        """Winning move for G3 (path A-B-C-D-E, edges AB, BC, CD, DE).

        From proof of Theorem 4.4:
        (i) w(DE) >= w(AB)+w(BC): select D, remove CD, reduce DE to w(AB)+w(BC).
            (Symmetric: w(AB) >= w(CD)+w(DE): select B, remove BC, reduce AB.)
        (ii) w(AB)+w(BC) > w(DE) >= w(AB): select C, remove CD,
             reduce BC by w(AB)+w(BC)-w(DE).
            (Symmetric: w(CD)+w(DE) > w(AB) >= w(DE): select C, remove BC,
             reduce CD by w(CD)+w(DE)-w(AB).)
        (iii) Remaining cases reduce to (i) or (ii) by symmetry.

        Implements constructive proof of Theorem 4.4 (G3 case), arXiv:2509.05064v1.
        """
        ab = state.get_weight("A", "B")
        bc = state.get_weight("B", "C")
        cd = state.get_weight("C", "D")
        de = state.get_weight("D", "E")

        # Case (i): DE >= AB + BC
        if de >= ab + bc:
            reductions = {("C", "D"): cd}
            reduce_de = de - (ab + bc)
            if reduce_de > 0:
                reductions[("D", "E")] = reduce_de
            return Move(vertex="D", reductions=reductions)

        # Symmetric case (i): AB >= CD + DE
        if ab >= cd + de:
            reductions = {("B", "C"): bc}
            reduce_ab = ab - (cd + de)
            if reduce_ab > 0:
                reductions[("A", "B")] = reduce_ab
            return Move(vertex="B", reductions=reductions)

        # Case (ii): AB + BC > DE >= AB
        if de >= ab:
            # Select C, remove CD entirely, reduce BC
            reduce_bc = ab + bc - de
            reductions = {("C", "D"): cd}
            if reduce_bc > 0:
                reductions[("B", "C")] = reduce_bc
            return Move(vertex="C", reductions=reductions)

        # Symmetric case (ii): CD + DE > AB >= DE
        if ab >= de:
            # But we already handled DE >= AB above, so here AB > DE.
            # Check CD + DE > AB
            if cd + de > ab:
                # Select C, remove BC entirely, reduce CD by CD+DE-AB
                reduce_cd = cd + de - ab
                reductions = {("B", "C"): bc}
                if reduce_cd > 0 and reduce_cd <= cd:
                    reductions[("C", "D")] = reduce_cd
                return Move(vertex="C", reductions=reductions)

        # Fallback: AB > DE and AB >= CD + DE already handled.
        # The above covers all cases since either DE >= AB or AB > DE.
        # If AB > DE and CD+DE <= AB: handled by symmetric case (i).
        # If AB > DE and CD+DE > AB: handled by symmetric case (ii).
        return None
