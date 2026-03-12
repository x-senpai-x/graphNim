"""Tests for graph classifier."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from graph_state import GraphState
from classifier import classify


def _make(vertices, edges, weights):
    return GraphState.create(vertices, edges, weights)


class TestClassifier:
    """Test classification of all 11 graph types."""

    def test_g1_star(self):
        """G1 = K_{1,4}: star with 4 rays."""
        state = _make(
            ["A", "B", "C", "D", "E"],
            [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")],
            {("A", "B"): 1, ("A", "C"): 2, ("A", "D"): 3, ("A", "E"): 4},
        )
        gtype, _ = classify(state)
        assert gtype == "G1"

    def test_h2_two_paths(self):
        """H2 = 2×K_{1,2}: two disjoint paths of length 2."""
        state = _make(
            ["A", "B", "C", "D", "E", "F"],
            [("A", "B"), ("B", "C"), ("D", "E"), ("E", "F")],
            {("A", "B"): 1, ("B", "C"): 2, ("D", "E"): 3, ("E", "F"): 4},
        )
        gtype, _ = classify(state)
        assert gtype == "H2"

    def test_h3_star3_edge(self):
        """H3 = K_{1,3} ∪ K₂."""
        state = _make(
            ["A", "B", "C", "D", "E", "F"],
            [("A", "B"), ("A", "C"), ("A", "D"), ("E", "F")],
            {("A", "B"): 1, ("A", "C"): 2, ("A", "D"): 3, ("E", "F"): 6},
        )
        gtype, _ = classify(state)
        assert gtype == "H3"

    def test_i1_star2_two_edges(self):
        """I1 = K_{1,2} ∪ 2K₂."""
        state = _make(
            ["A", "B", "C", "D", "E", "F", "G"],
            [("A", "B"), ("B", "C"), ("D", "E"), ("F", "G")],
            {("A", "B"): 1, ("B", "C"): 2, ("D", "E"): 3, ("F", "G"): 4},
        )
        gtype, _ = classify(state)
        assert gtype == "I1"

    def test_i2_four_edges(self):
        """I2 = 4K₂: four disjoint edges."""
        state = _make(
            ["A", "B", "C", "D", "E", "F", "G", "H"],
            [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")],
            {("A", "B"): 1, ("C", "D"): 1, ("E", "F"): 1, ("G", "H"): 1},
        )
        gtype, _ = classify(state)
        assert gtype == "I2"

    def test_f1_cycle(self):
        """F1 = C₄: 4-cycle."""
        state = _make(
            ["A", "B", "C", "D"],
            [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")],
            {("A", "B"): 3, ("B", "C"): 4, ("C", "D"): 3, ("D", "A"): 4},
        )
        gtype, _ = classify(state)
        assert gtype == "F1"

    def test_f2_triangle_pendant(self):
        """F2 = Triangle + pendant."""
        state = _make(
            ["A", "B", "C", "D"],
            [("A", "B"), ("B", "C"), ("C", "D"), ("D", "B")],
            {("A", "B"): 1, ("B", "C"): 2, ("C", "D"): 3, ("D", "B"): 2},
        )
        gtype, _ = classify(state)
        assert gtype == "F2"

    def test_g2(self):
        """G2: edges AB, AC, AD, BE."""
        state = _make(
            ["A", "B", "C", "D", "E"],
            [("A", "B"), ("A", "C"), ("A", "D"), ("B", "E")],
            {("A", "B"): 1, ("A", "C"): 2, ("A", "D"): 3, ("B", "E"): 4},
        )
        gtype, _ = classify(state)
        assert gtype == "G2"

    def test_g3_path5(self):
        """G3: path A-B-C-D-E."""
        state = _make(
            ["A", "B", "C", "D", "E"],
            [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")],
            {("A", "B"): 1, ("B", "C"): 2, ("C", "D"): 3, ("D", "E"): 4},
        )
        gtype, _ = classify(state)
        assert gtype == "G3"

    def test_g4_triangle_edge(self):
        """G4 = Triangle ∪ K₂."""
        state = _make(
            ["A", "B", "C", "D", "E"],
            [("A", "B"), ("B", "C"), ("C", "A"), ("D", "E")],
            {("A", "B"): 2, ("B", "C"): 3, ("C", "A"): 4, ("D", "E"): 1},
        )
        gtype, _ = classify(state)
        assert gtype == "G4"

    def test_h1_path_edge(self):
        """H1 = Path A-B-C-D + isolated edge E-F."""
        state = _make(
            ["A", "B", "C", "D", "E", "F"],
            [("A", "B"), ("B", "C"), ("C", "D"), ("E", "F")],
            {("A", "B"): 3, ("B", "C"): 2, ("C", "D"): 5, ("E", "F"): 4},
        )
        gtype, _ = classify(state)
        assert gtype == "H1"

    def test_relabeled_graph(self):
        """Classification works with non-standard vertex labels."""
        # H1 with labels X,Y,Z,W,P,Q instead of A,B,C,D,E,F
        state = _make(
            ["X", "Y", "Z", "W", "P", "Q"],
            [("X", "Y"), ("Y", "Z"), ("Z", "W"), ("P", "Q")],
            {("X", "Y"): 3, ("Y", "Z"): 2, ("Z", "W"): 5, ("P", "Q"): 4},
        )
        gtype, label_map = classify(state)
        assert gtype == "H1"
        # label_map should map user labels to canonical labels
        assert len(label_map) == 6
