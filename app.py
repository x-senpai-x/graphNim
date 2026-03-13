"""Streamlit web UI for Graph Nim Solver.

Run with:
    streamlit run app.py
"""

from __future__ import annotations
import sys
import os

# Ensure project root is on path when running from any directory
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional

from graph_state import GraphState, canonical_edge
from classifier import classify, remap_state
from solvers.base import Move
from solvers.galaxy import GalaxySolver
from solvers.f1 import F1Solver
from solvers.f2 import F2Solver
from solvers.g2_g3 import G2G3Solver
from solvers.g4 import G4Solver
from solvers.h1 import H1Solver
from retrograde import retrograde_search


# ---------------------------------------------------------------------------
# Solver registry
# ---------------------------------------------------------------------------

SOLVER_MAP = {
    "G1": lambda: GalaxySolver("G1"),
    "H2": lambda: GalaxySolver("H2"),
    "H3": lambda: GalaxySolver("H3"),
    "I1": lambda: GalaxySolver("I1"),
    "I2": lambda: GalaxySolver("I2"),
    "F1": lambda: F1Solver(),
    "F2": lambda: F2Solver(),
    "G2": lambda: G2G3Solver("G2"),
    "G3": lambda: G2G3Solver("G3"),
    "G4": lambda: G4Solver(),
    "H1": lambda: H1Solver(),
}

# ---------------------------------------------------------------------------
# Graph templates: (vertices, edges, default_weights, description)
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, dict] = {
    "H1 — Path + isolated edge": {
        "vertices": ["A", "B", "C", "D", "E", "F"],
        "edges": [("A", "B"), ("B", "C"), ("C", "D"), ("E", "F")],
        "weights": {"AB": 3, "BC": 2, "CD": 5, "EF": 4},
        "desc": "Path A-B-C-D plus isolated edge E-F (partially open)",
    },
    "F1 — 4-cycle C₄": {
        "vertices": ["A", "B", "C", "D"],
        "edges": [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")],
        "weights": {"AB": 3, "BC": 4, "CD": 3, "DA": 4},
        "desc": "Cycle A-B-C-D-A. Losing iff opposite edges are equal.",
    },
    "F2 — Triangle + pendant": {
        "vertices": ["A", "B", "C", "D"],
        "edges": [("A", "B"), ("B", "C"), ("C", "D"), ("D", "B")],
        "weights": {"AB": 2, "BC": 3, "CD": 5, "DB": 3},
        "desc": "Triangle B-C-D with pendant A-B.",
    },
    "G1 — Star K₁,₄": {
        "vertices": ["A", "B", "C", "D", "E"],
        "edges": [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")],
        "weights": {"AB": 2, "AC": 5, "AD": 3, "AE": 7},
        "desc": "Star with centre A and 4 leaves. Always WINNING.",
    },
    "G4 — Triangle + isolated edge": {
        "vertices": ["A", "B", "C", "D", "E"],
        "edges": [("A", "B"), ("B", "C"), ("C", "A"), ("D", "E")],
        "weights": {"AB": 3, "BC": 3, "CA": 3, "DE": 9},
        "desc": "Triangle A-B-C plus isolated edge D-E.",
    },
    "I2 — 4 disjoint edges": {
        "vertices": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "edges": [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")],
        "weights": {"AB": 3, "CD": 5, "EF": 6, "GH": 4},
        "desc": "Four independent edges. Losing iff XOR of all weights = 0.",
    },
    "H2 — Two P₃ paths": {
        "vertices": ["A", "B", "C", "D", "E", "F"],
        "edges": [("A", "B"), ("B", "C"), ("D", "E"), ("E", "F")],
        "weights": {"AB": 3, "BC": 3, "DE": 5, "EF": 5},
        "desc": "Two disjoint 2-star paths. Losing iff star sums are equal.",
    },
    "G2 — Fan graph": {
        "vertices": ["A", "B", "C", "D", "E"],
        "edges": [("A", "B"), ("A", "C"), ("A", "D"), ("B", "E")],
        "weights": {"AB": 4, "AC": 2, "AD": 6, "BE": 3},
        "desc": "Edges AB,AC,AD,BE. Always WINNING.",
    },
    "G3 — Path P₅": {
        "vertices": ["A", "B", "C", "D", "E"],
        "edges": [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")],
        "weights": {"AB": 2, "BC": 5, "CD": 3, "DE": 7},
        "desc": "Path A-B-C-D-E. Always WINNING.",
    },
}

# Fixed layout positions for each template (for visual stability)
FIXED_POS: dict[str, dict] = {
    "H1 — Path + isolated edge": {
        "A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (3, 0), "E": (1.5, -1.2), "F": (2.5, -1.2)
    },
    "F1 — 4-cycle C₄": {
        "A": (0, 0), "B": (1, 0), "C": (1, 1), "D": (0, 1)
    },
    "F2 — Triangle + pendant": {
        "A": (0, 0), "B": (1, 0), "C": (2, 1), "D": (1, 2)
    },
    "G1 — Star K₁,₄": {
        "A": (1, 1), "B": (0, 2), "C": (2, 2), "D": (0, 0), "E": (2, 0)
    },
    "G4 — Triangle + isolated edge": {
        "A": (0, 0), "B": (1, 1.5), "C": (2, 0), "D": (3.5, 0.75), "E": (4.5, 0.75)
    },
    "I2 — 4 disjoint edges": {
        "A": (0, 1), "B": (0, 0), "C": (1.5, 1), "D": (1.5, 0),
        "E": (3, 1), "F": (3, 0), "G": (4.5, 1), "H": (4.5, 0)
    },
    "H2 — Two P₃ paths": {
        "A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (0, -1.2), "E": (1, -1.2), "F": (2, -1.2)
    },
    "G2 — Fan graph": {
        "A": (1, 1), "B": (0, 0), "C": (1, 0), "D": (2, 0), "E": (-1, 0)
    },
    "G3 — Path P₅": {
        "A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (3, 0), "E": (4, 0)
    },
}


# ---------------------------------------------------------------------------
# Graph rendering
# ---------------------------------------------------------------------------

def render_graph(
    vertices: list,
    edges: list[tuple],
    weights: dict[str, int],
    move: Optional[Move] = None,
    label_map_inv: Optional[dict] = None,
    template_name: str = "",
) -> plt.Figure:
    """Draw the weighted graph, optionally highlighting the winning move."""
    G = nx.Graph()
    G.add_nodes_from(vertices)
    for u, v in edges:
        G.add_edge(u, v)

    # Layout
    pos = FIXED_POS.get(template_name)
    if pos is None or not all(v in pos for v in vertices):
        pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Determine highlighted vertex and edges from move
    highlight_vertex = None
    highlight_edges: set[tuple] = set()
    edge_labels_extra: dict[tuple, str] = {}

    if move is not None and label_map_inv is not None:
        highlight_vertex = label_map_inv.get(move.vertex, move.vertex)
        for edge, amount in move.reductions.items():
            u = label_map_inv.get(edge[0], edge[0])
            v = label_map_inv.get(edge[1], edge[1])
            ce = canonical_edge(u, v)
            highlight_edges.add(ce)
            edge_labels_extra[ce] = f"−{amount}"

    # Node colors
    node_colors = []
    for v in G.nodes():
        if v == highlight_vertex:
            node_colors.append("#e05252")  # red for chosen vertex
        else:
            node_colors.append("#4a90d9")  # blue default

    # Edge colors and widths
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        ce = canonical_edge(u, v)
        if ce in highlight_edges:
            edge_colors.append("#f0a500")  # orange for changed edges
            edge_widths.append(3.5)
        else:
            edge_colors.append("#aaaaaa")
            edge_widths.append(1.5)

    nx.draw_networkx(
        G, pos, ax=ax,
        node_color=node_colors,
        edge_color=edge_colors,
        width=edge_widths,
        node_size=700,
        font_color="white",
        font_size=11,
        font_weight="bold",
        with_labels=True,
    )

    # Weight labels (show current weight on each edge)
    weight_labels = {}
    for u, v in G.edges():
        ce = canonical_edge(u, v)
        w = weights.get(f"{ce[0]}{ce[1]}", weights.get(f"{ce[1]}{ce[0]}", 0))
        label = str(w)
        if ce in edge_labels_extra:
            label += f"  {edge_labels_extra[ce]}"
        weight_labels[(u, v)] = label

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=weight_labels, ax=ax,
        font_color="#ffdd99",
        font_size=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="#1e2530", ec="none", alpha=0.8),
    )

    # Legend
    if move is not None:
        patches = [
            mpatches.Patch(color="#e05252", label="Selected vertex"),
            mpatches.Patch(color="#f0a500", label="Reduced edge"),
        ]
        ax.legend(handles=patches, loc="upper right", facecolor="#1e2530",
                  edgecolor="none", labelcolor="white", fontsize=8)

    ax.axis("off")
    plt.tight_layout(pad=0.5)
    return fig


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Graph Nim Solver",
    page_icon="♟",
    layout="wide",
)

st.title("Graph Nim Solver")
st.caption("Based on arXiv:2509.05064v1 — all graphs with exactly 4 edges.")

# ---- Sidebar: template selection ----
with st.sidebar:
    st.header("Graph Setup")
    template_name = st.selectbox("Graph template", list(TEMPLATES.keys()))
    tpl = TEMPLATES[template_name]
    st.caption(tpl["desc"])
    st.divider()

    # Edge weight sliders
    st.subheader("Edge weights")
    edges = tpl["edges"]
    vertices = tpl["vertices"]
    slider_weights: dict[str, int] = {}
    for u, v in edges:
        key = f"{u}{v}"
        default = tpl["weights"].get(key, tpl["weights"].get(f"{v}{u}", 3))
        slider_weights[key] = st.slider(
            f"w({u}–{v})", min_value=1, max_value=20, value=default, key=f"slider_{key}"
        )

    st.divider()
    solve_btn = st.button("Solve", type="primary", use_container_width=True)

# ---- Build state ----
weights_dict = {canonical_edge(k[0], k[1]): v for k, v in slider_weights.items()}
state = GraphState.create(vertices, edges, weights_dict)

# ---- Layout: visualization | result ----
col_viz, col_result = st.columns([3, 2], gap="large")

result = None
move = None
graph_type = None
label_map_inv: dict = {}

if solve_btn or True:  # Always show the graph preview; solve on button click
    try:
        graph_type, label_map = classify(state)
        label_map_inv = {v: k for k, v in label_map.items()}
        canonical_state = remap_state(state, label_map)
        solver = SOLVER_MAP[graph_type]()
        result = solver.evaluate(canonical_state)

        if result == "UNKNOWN" and graph_type == "H1":
            retro_result, retro_move = retrograde_search(canonical_state)
            if retro_result != "UNKNOWN":
                result = retro_result
                move = retro_move

        if result == "WINNING" and move is None:
            move = solver.winning_move(canonical_state)

    except Exception as e:
        st.error(f"Solver error: {e}")

# ---- Visualization ----
with col_viz:
    st.subheader("Graph")
    fig = render_graph(
        vertices, edges, slider_weights,
        move=move if solve_btn else None,
        label_map_inv=label_map_inv if solve_btn else None,
        template_name=template_name,
    )
    st.pyplot(fig)
    plt.close(fig)

# ---- Result panel ----
with col_result:
    st.subheader("Analysis")

    if graph_type:
        st.markdown(f"**Graph type:** `{graph_type}`")

    if result is None:
        st.info("Adjust weights and click **Solve**.")
    elif result == "WINNING":
        st.success("WINNING")
        if move is not None:
            vertex_label = label_map_inv.get(move.vertex, move.vertex)
            st.markdown(f"**Select vertex:** `{vertex_label}`")
            rows = []
            total = 0
            for edge, amount in sorted(move.reductions.items()):
                if amount > 0:
                    u = label_map_inv.get(edge[0], edge[0])
                    v = label_map_inv.get(edge[1], edge[1])
                    rows.append({"Edge": f"{u}–{v}", "Remove": amount})
                    total += amount
            if rows:
                st.table(rows)
            st.markdown(f"**Total removed:** `{total}`")
        else:
            st.caption("(Winning move could not be constructed)")
    elif result == "LOSING":
        st.error("LOSING")
        st.caption("The player to move loses with optimal play from the opponent.")
    elif result == "UNKNOWN":
        st.warning("UNKNOWN")
        st.caption(
            "This H1 position is beyond Theorem 4.10's current scope (r > 4). "
            "The retrograde search also reached its depth limit. "
            "The answer is an open problem."
        )

    # Theorem reference
    THEOREM_MAP = {
        "G1": "Theorem 4.1(i)", "H2": "Theorem 4.1(ii)", "H3": "Theorem 4.1(iii)",
        "I1": "Theorem 4.1(iv)", "I2": "Theorem 4.1(v)", "F1": "Theorem 4.2",
        "F2": "Theorem 4.3", "G2": "Theorem 4.4", "G3": "Theorem 4.4",
        "G4": "Theorem 4.6", "H1": "Theorems 4.7–4.10",
    }
    if graph_type and solve_btn:
        st.divider()
        st.caption(f"Implemented via {THEOREM_MAP.get(graph_type, '—')} · arXiv:2509.05064v1")
