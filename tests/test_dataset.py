# double checking leakage / integrity tests for cached dataset
import numpy as np
import pandas as pd

from src.dataset.cache import load_attempts, load_dataset
from src.dataset.features import EDGE_COLS, NODE_COLS
from src.dataset.folds import load_folds

SNAPS, META = load_dataset()
ATT = load_attempts()

def test_counts_match_the_proposal(): 
    # 4947 attempts, 27136 snapshots, 8277 solved-labelled for example
    assert META["n_attempts"] == 4947
    assert ATT.ran.all()
    assert int(ATT.correct.sum()) == 2647 and int((ATT.correct == 0).sum()) == 2300
    assert len(SNAPS) == 27136
    assert sum(s["event"] for s in SNAPS) == 8277

def test_no_reached_goal_feature():
    assert not any("reach" in c and "goal" in c for c in NODE_COLS + EDGE_COLS)
    assert "reached_goal" not in NODE_COLS

def test_no_operation_edge_into_the_goal_in_any_snapshot():
    # goal is row with is_goal = 1
    g, sub = NODE_COLS.index("is_goal"), EDGE_COLS.index("is_subgoal")
    for s in SNAPS:
        goal_rows = np.where(s["x"][:, g] == 1)[0]
        into_goal = np.isin(s["edge_index"][1], goal_rows)
        assert not (into_goal & (s["edge_attr"][:, sub] == 0)).any()
        assert s["x"][goal_rows, NODE_COLS.index("visit_count")].sum() == 0

def test_labels_are_consistent():
    for s in SNAPS:
        assert s["time_to_event"] >= (1 if s["event"] else 0)
        assert s["event"] in (0, 1)

def test_steps_are_contiguous_per_attempt():
    df = pd.DataFrame([(s["pid"], s["trial_index"], s["step"]) for s in SNAPS], columns=["pid", "t", "step"])
    for _, g in df.groupby(["pid", "t"]):
        assert list(g.step) == list(range(1, len(g) + 1))

def test_graph_never_shrinks_within_an_attempt():
    last = {}

    for s in SNAPS:
        key = (s["pid"], s["trial_index"])
        n_edges = s["edge_index"].shape[1]
        assert n_edges >= last.get(key, 0)
        last[key] = n_edges

def test_all_snapshots_of_a_group_share_a_fold():
    df = pd.DataFrame(SNAPS)[["pid", "puzzle", "puzzle_fold", "participant_fold"]]

    assert df.groupby("puzzle").puzzle_fold.nunique().max() == 1
    assert df.groupby("pid").participant_fold.nunique().max() == 1

def test_folds_file_matches_the_dataset():
    folds = load_folds()
    keys = set(zip(folds.pid, folds.trial_index))
    
    assert {(s["pid"], s["trial_index"]) for s in SNAPS} <= keys

def test_shapes_and_values_are_finite():
    for s in SNAPS:
        assert s["x"].shape[1] == len(NODE_COLS) and s["edge_attr"].shape[1] == len(EDGE_COLS)
        assert s["edge_attr"].shape[0] == s["edge_index"].shape[1]
        assert np.isfinite(s["x"]).all() and np.isfinite(s["edge_attr"]).all()

# python -m pytest tests/test_dataset.py