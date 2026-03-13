# Graph Nim Solver

A Python solver for the Game of Graph Nim on all graphs with exactly 4 edges, based on the paper:

> "The Game of Graph Nim on Graphs with Four Edges" — Karmakar, Podder, Roy, Sadhukhan (arXiv:2509.05064v1)

Given a weighted graph with 4 edges, the solver identifies its structure, classifies the position as **WINNING** or **LOSING** (or **UNKNOWN** for unresolved H1 cases), and outputs a concrete winning move.

## Game Rules

- Each edge has a positive integer weight (tokens).
- A move: select one vertex, remove non-negative integers from each incident edge such that the **total removed > 0**.
- The player who makes the last move **wins**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Web UI

```bash
streamlit run app.py
```

Opens a browser with:
- **Sidebar**: pick a graph template (H1, F1, G1, etc.) and adjust edge weights with sliders
- **Graph panel**: NetworkX visualization with edge weight labels; the winning vertex is highlighted red and reduced edges in orange after solving
- **Result panel**: WINNING / LOSING / UNKNOWN badge, the concrete winning move as a table, and the theorem reference

## CLI Usage

```bash
python main.py --vertices A B C D E F --edges AB BC CD EF --weights AB:3 BC:2 CD:5 EF:4
```

**Output:**
```
Graph Type: H1
Position:   WINNING
Winning Move:
  Select vertex: C
  Remove from BC: 1
  Remove from CD: 5
  Total removed: 6
```

### JSON input

```bash
python main.py --json '{"vertices":["A","B","C","D"],"edges":[["A","B"],["B","C"],["C","D"],["D","A"]],"weights":{"AB":3,"BC":4,"CD":3,"DA":4}}'
```

## Graph Types Supported

| Type | Structure | Condition |
|------|-----------|-----------|
| G1 | K₁,₄ (star, 5 vertices) | All configs winning |
| H2 | 2×K₁,₂ (two P₃ paths) | Losing iff star sums equal |
| H3 | K₁,₃ ∪ K₂ | Losing iff star sum = edge weight |
| I1 | K₁,₂ ∪ 2K₂ | Losing iff NIM-XOR balanced |
| I2 | 4K₂ (4 disjoint edges) | Losing iff NIM-XOR = 0 |
| F1 | C₄ (4-cycle) | Losing iff opposite edges equal |
| F2 | Triangle + pendant | Theorem 4.3 condition |
| G2 | Edges AB,AC,AD,BE | All configs winning |
| G3 | Path A-B-C-D-E | All configs winning |
| G4 | Triangle ∪ K₂ | Special multiset condition (Thm 4.6) |
| H1 | Path A-B-C-D ∪ K₂ EF | Partial — open problem (Thm 4.10) |

H1 is partially unresolved. If the theorems can't classify a position, bounded retrograde analysis is attempted; otherwise **UNKNOWN** is returned.

## Running Tests

```bash
source .venv/bin/activate
pytest
```

73 tests covering all graph types, edge cases, and winning move validity.

## Project Structure

```
graphnim/
├── graph_state.py      # GraphState dataclass
├── classifier.py       # Identifies graph type via NetworkX isomorphism
├── main.py             # CLI entry point
├── retrograde.py       # Bounded search for H1 unknowns
├── solvers/
│   ├── base.py         # Move + abstract Solver
│   ├── galaxy.py       # G1, H2, H3, I1, I2
│   ├── f1.py           # C₄ cycle
│   ├── f2.py           # Triangle + pendant
│   ├── g2_g3.py        # All-winning graphs
│   ├── g4.py           # Triangle ∪ K₂
│   └── h1.py           # Path ∪ K₂ (partial)
└── tests/
```
