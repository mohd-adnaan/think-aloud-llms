# baseline 1: predicting only from the 4 starting numbers
# 2 solver based difficulty features per puzzle: 1. log_paths = log(1 + num distinct move squences from start that end at 24), 2. good_first = share of distinct first moves after which 24 still reachable
from fractions import Fraction
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.dataset.cache import load_dataset
from src.dataset.features import reachable_24
from src.dataset.folds import N_FOLDS, split_masks
from src.dataset.load import ROOT
from src.dataset.predictions import save_predictions, within_k_labels

OUT = ROOT / "data/predictions/puzzle_only.csv"

def _moves(state):
    """All distinct states one operation away (state = sorted tuple of Fractions)."""
    out = set()
    seen = set()
    for i, j in combinations(range(len(state)), 2):
        a, b = state[i], state[j]
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rest = [state[k] for k in range(len(state)) if k not in (i, j)]
        results = {a + b, a * b, a - b, b - a}
        if b != 0:
            results.add(a / b)
        if a != 0:
            results.add(b / a)
        for r in results:
            out.add(tuple(sorted(rest + [r])))
    return out

@lru_cache(maxsize=None)
def count_paths(state: tuple) -> int:
    """Number of distinct move sequences from `state` that end at the single number 24."""
    if len(state) == 1:
        return int(state[0] == 24)
    return sum(count_paths(s) for s in _moves(state))

def puzzle_features(puzzle: str) -> dict:
    state = tuple(sorted(Fraction(int(n)) for n in puzzle.split("-")))
    first = _moves(state)
    good = sum(reachable_24(s) for s in first) / max(len(first), 1)
    return dict(log_paths=np.log1p(count_paths(state)), good_first=good)

# logistic regression predicts 'solves within the next 3 steps', fits on the training folds only
def run():
    snaps, _ = load_dataset()
    df = pd.DataFrame([{k: s[k] for k in ("pid", "trial_index", "puzzle", "step", "event", "time_to_event",
                                            "puzzle_fold", "participant_fold")} for s in snaps])
    feats = pd.DataFrame({p: puzzle_features(p) for p in df.puzzle.unique()}).T
    X = feats.loc[df.puzzle].to_numpy(dtype=float)
    usable, y = within_k_labels(df.event, df.time_to_event)
    rows, report = [], []
    for scheme in ("puzzle", "participant"):
        for test_fold in range(N_FOLDS):
            train, _, test = split_masks(df, scheme, test_fold)
            tr = (train & usable).to_numpy()
            model = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
            te = test.to_numpy()
            p = model.predict_proba(X[te])[:, 1]
            part = df.loc[te, ["pid", "trial_index", "step"]].copy()
            part.insert(0, "model", "puzzle_only")
            part.insert(1, "scheme", scheme)
            part.insert(2, "test_fold", test_fold)
            part.insert(3, "seed", 0)
            part["risk"], part["p_within3"] = p, p
            rows.append(part)
            u = usable[te]
            report.append((scheme, test_fold, roc_auc_score(y[te][u], p[u]), int(u.sum())))
    pred = pd.concat(rows, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(pred, OUT)
    return pred, pd.DataFrame(report, columns=["scheme", "test_fold", "auroc_within3", "n_scored"]), feats

if __name__ == "__main__":
    pred, rep, feats = run()
    print(feats.describe().round(2).to_string())
    print(rep.round(3).to_string(index=False))
    print(rep.groupby("scheme").auroc_within3.agg(["mean", "std"]).round(3))
    print("wrote", OUT, len(pred), "rows")

# python -m src.baselines.puzzle_only