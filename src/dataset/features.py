import re
from fractions import Fraction
from functools import lru_cache
from itertools import combinations

import networkx as nx
import numpy as np # numpy node and edge features for a snapshot graph

GOAL = (24,)
NODE_COLS = ["visit_count", "depth", "has_depth", "n_numbers", "reach24",
             "is_start", "is_current", "is_goal", "is_subgoal_node"]
EDGE_COLS = ["op_add", "op_sub", "op_mul", "op_div", "op_other",
             "is_subgoal", "n_tries", "is_correct"]
_OP_RE = re.compile(r"^\s*-?[\d.]+\s*([+\-*/])")

def _frac(x):
    return Fraction(x) if isinstance(x, int) else Fraction(str(x))

@lru_cache(maxsize=None)
def _can_make_24(nums: tuple) -> bool:
    # can these nums all be used to make 24? 
    if len(nums) == 1:
        return nums[0] == 24
    for i, j in combinations(range(len(nums)), 2): # trying all combos
        a, b = nums[i], nums[j]
        rest = [nums[k] for k in range(len(nums)) if k not in (i, j)]
        # arith operations
        results = {a + b, a * b, a - b, b - a}
        if b != 0:
            results.add(a / b)
        if a != 0:
            results.add(b / a)
        for c in results:
            if _can_make_24(tuple(sorted(rest + [c]))):
                return True
    return False

def reachable_24(state) -> bool:
    # true if this state can make 24
    return _can_make_24(tuple(sorted(_frac(x) for x in state)))

def _start_node(G):
    for n, d in G.nodes(data=True):
        if d.get("visitation_timesteps") and d["visitation_timesteps"][0] == 0:
            return n
    return None

def featurize(G: nx.DiGraph, current_state):
    # dict(nodes, x, edge_index, edge_attr, node_cols, edge_cols) for a snapshot
    nodes = list(G.nodes)
    index = {n: i for i, n in enumerate(nodes)} # fix the order, gives each state a row index

    start = _start_node(G)
    current = tuple(sorted(current_state))

    # depth = fewest operations from the start state
    ops_only = nx.DiGraph()
    ops_only.add_nodes_from(G.nodes)
    ops_only.add_edges_from((u, v) for u, v, d in G.edges(data=True) if d.get("operation") != "subgoal")
    # edges where operation isn't subgoal ^ 

    # shortest path length from start
    depth = nx.single_source_shortest_path_length(ops_only, start) if start is not None else {}

    x = np.zeros((len(nodes), len(NODE_COLS)), dtype = np.float32)

    # for each node fill in all of this info
    for n, i in index.items():
        d = G.nodes[n]
        x[i] = [
            len(d.get("visitation_timesteps", [])), # visit_count
            depth.get(n, 0), # depth
            1.0 if n in depth else 0.0,
            len(n), # numbers remaining
            1.0 if reachable_24(n) else 0.0, 
            1.0 if n == start else 0.0, # three position flags
            1.0 if n == current else 0.0,
            1.0 if n == GOAL else 0.0,
            1.0 if d.get("initialized_as_subgoal", False) else 0.0, # whether created by subgoal
        ]

    edges = list(G.edges(data=True))
    # row 0 = edge source, row 1 = target
    edge_index = np.array([[index[u] for u, _, _ in edges], 
                           [index[v] for _, v, _ in edges]], dtype=np.int64).reshape(2, len(edges))
    # 8 numbers per edge
    edge_attr = np.zeros((len(edges), len(EDGE_COLS)), dtype=np.float32)

    for k, (_, _, d) in enumerate(edges):
        op = d.get("operation", "")
        is_sub = op == "subgoal"
        m = None if is_sub else _OP_RE.match(op)
        sym = m.group(1) if m else None
        edge_attr[k] = [
            sym == "+", sym == "-", sym == "*", sym == "/", # is it any of these operations
            (not is_sub) and sym is None, # other
            is_sub, # subgoal edge
            len(d.get("op_timesteps", [])), # how many times tried
            True if is_sub else bool(d.get("is_correct", True)), # is the arithmetic correct
        ]
        
    return dict(nodes=nodes, x=x, edge_index=edge_index, edge_attr=edge_attr,
                node_cols=NODE_COLS, edge_cols=EDGE_COLS)