"""Graph classifier for 4-edge Graph Nim.

Identifies which of the 11 graph types (G1, H2, H3, I1, I2,
F1, F2, G2, G3, G4, H1) a given graph belongs to, and returns
a canonical vertex labeling for use by solvers.

Graph type structures (from arXiv:2509.05064v1, Figure 2):
  G1:  K_{1,4} — star with 4 rays, 5 vertices, 1 component. (Thm 4.1i)
  H2:  2×K_{1,2} — two disjoint 2-ray stars, 6 vertices. (Thm 4.1ii)
  H3:  K_{1,3} ∪ K₂ — 3-ray star + isolated edge, 6 vertices. (Thm 4.1iii)
  I1:  K_{1,2} ∪ 2K₂ — 2-ray star + 2 isolated edges, 7 vertices. (Thm 4.1iv)
  I2:  4K₂ — 4 disjoint edges, 8 vertices. (Thm 4.1v)
  F1:  C₄ — 4-cycle, 4 vertices. (Thm 4.2)
  F2:  Triangle + pendant, 4 vertices. (Thm 4.3)
  G2:  Edges AB,AC,AD,BE — 5 vertices, 1 component. (Thm 4.4)
  G3:  Path A-B-C-D-E — 5 vertices, 1 component. (Thm 4.4)
  G4:  Triangle ∪ K₂ — 5 vertices, 2 components. (Thm 4.6)
  H1:  Path P₄ ∪ K₂ — 6 vertices, 2 components. (Thm 4.7–4.10)
"""

from __future__ import annotations
from typing import Any

import networkx as nx

from graph_state import GraphState, canonical_edge


# ---------------------------------------------------------------------------
# Canonical reference graphs (unweighted)
# ---------------------------------------------------------------------------

def _make_ref_graphs() -> dict[str, nx.Graph]:
    """Build one canonical (unweighted) graph per type."""
    refs: dict[str, nx.Graph] = {}

    # G1: K_{1,4} — star centre A, leaves B,C,D,E
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")])
    refs["G1"] = g

    # H2: 2×K_{1,2} — stars at B (A-B-C) and E (D-E-F)
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("D", "E"), ("E", "F")])
    refs["H2"] = g

    # H3: K_{1,3} ∪ K₂ — star at A (A-B, A-C, A-D) + edge E-F
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("A", "C"), ("A", "D"), ("E", "F")])
    refs["H3"] = g

    # I1: K_{1,2} ∪ 2K₂ — star at B (A-B-C) + edges D-E, F-G
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("D", "E"), ("F", "G")])
    refs["I1"] = g

    # I2: 4K₂ — 4 disjoint edges
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")])
    refs["I2"] = g

    # F1: C₄ — cycle A-B-C-D-A
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
    refs["F1"] = g

    # F2: Triangle B-C-D + pendant A-B (Theorem 4.3 uses edges AB, BC, CD, DB)
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "B")])
    refs["F2"] = g

    # G2: edges AB, AC, AD, BE (from proof of Theorem 4.4)
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("A", "C"), ("A", "D"), ("B", "E")])
    refs["G2"] = g

    # G3: path A-B-C-D-E (from proof of Theorem 4.4)
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])
    refs["G3"] = g

    # G4: triangle A-B-C + isolated edge D-E
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("D", "E")])
    refs["G4"] = g

    # H1: path A-B-C-D + isolated edge E-F
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("E", "F")])
    refs["H1"] = g

    return refs


_REF_GRAPHS = _make_ref_graphs()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _is_star(component: nx.Graph) -> bool:
    """True if the component is a star graph (includes K₂ as a trivial star)."""
    if component.number_of_edges() == 0:
        return True
    if component.number_of_edges() == 1:
        return True  # K₂ is a star
    # A star K_{1,n} has exactly one vertex with degree n, rest degree 1
    degrees = sorted(component.degree(), key=lambda x: x[1], reverse=True)
    if degrees[0][1] == component.number_of_edges():
        return all(d == 1 for _, d in degrees[1:])
    return False


def _is_galaxy(G: nx.Graph) -> bool:
    """True if G is a galaxy graph (disjoint union of stars)."""
    for comp_nodes in nx.connected_components(G):
        if not _is_star(G.subgraph(comp_nodes)):
            return False
    return True


def _component_signatures(G: nx.Graph) -> list[tuple]:
    """Return sorted list of (num_edges, sorted_degree_seq) per component."""
    sigs = []
    for comp_nodes in nx.connected_components(G):
        sub = G.subgraph(comp_nodes)
        deg_seq = tuple(sorted((d for _, d in sub.degree()), reverse=True))
        sigs.append((sub.number_of_edges(), deg_seq))
    return sorted(sigs, reverse=True)


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify(state: GraphState) -> tuple[str, dict[Any, str]]:
    """Identify graph type and return (type_name, label_map).

    label_map maps user vertex labels → canonical labels used by solvers.
    For example, for H1 the canonical labels are A,B,C,D (path) and E,F (isolated edge).

    Returns:
        (graph_type, label_map) where graph_type is one of:
        G1, H2, H3, I1, I2, F1, F2, G2, G3, G4, H1
    """
    # Build unweighted NetworkX graph from state
    G = nx.Graph()
    G.add_nodes_from(state.vertices)
    for e in state.edges:
        G.add_edge(e[0], e[1])

    # Try isomorphism against each reference graph
    for gtype, ref in _REF_GRAPHS.items():
        gm = nx.isomorphism.GraphMatcher(G, ref)
        if gm.is_isomorphic():
            # gm.mapping: user_vertex -> canonical_vertex
            mapping = gm.mapping
            return gtype, mapping

    raise ValueError(
        f"Could not classify graph with {len(state.vertices)} vertices "
        f"and {len(state.edges)} edges as any known 4-edge type."
    )


def remap_state(state: GraphState, label_map: dict[Any, str]) -> GraphState:
    """Return a new GraphState with vertices relabeled according to label_map."""
    new_verts = [label_map[v] for v in state.vertices]
    new_edges = []
    new_weights = {}
    for e in state.edges:
        u, v = label_map[e[0]], label_map[e[1]]
        ce = canonical_edge(u, v)
        new_edges.append(ce)
        new_weights[ce] = state.weights[e]
    return GraphState(
        vertices=tuple(sorted(new_verts)),
        edges=tuple(sorted(new_edges)),
        weights=new_weights,
    )
