# trap state look up, predicts from state the person is in (numbers left), using how earlier people did in same state
# estimate P(solves within the next 3 steps), p = (k + m * p_parent) / (n + m)
import numpy as np
import pandas as pd

from src.dataset.cache import load_dataset
from src.dataset.features import reachable_24
from src.dataset.folds import N_FOLDS, split_masks
from src.dataset.load import ROOT
from src.dataset.predictions import save_predictions, within_k_labels
from src.dataset.quick_eval import score
from sklearn.metrics import roc_auc_score

OUT = ROOT / "data/predictions/trap_state.csv"
M_GRID = (1, 3, 10, 30, 100, 300)

def _shrunk(table: pd.DataFrame, parent: np.ndarray, m: float) -> np.ndarray:
    """table has columns n, k, indexed like the rows we predict for (NaN when unseen)."""
    n = table["n"].fillna(0).to_numpy()
    k = table["k"].fillna(0).to_numpy()
    return (k + m * parent) / (n + m)


def predict(train_df, test_df, m):
    """Return (p, exact_seen) for test_df using counts from train_df (both have state, coarse, y)."""
    p_all = train_df.y.mean()
    coarse = train_df.groupby("coarse").y.agg(n="size", k="sum")
    p_coarse = _shrunk(coarse.reindex(test_df.coarse), np.full(len(test_df), p_all), m)
    exact = train_df.groupby("state").y.agg(n="size", k="sum")
    t = exact.reindex(test_df.state)
    return _shrunk(t, p_coarse, m), t["n"].notna().to_numpy()


def run():
    snaps, _ = load_dataset()
    df = pd.DataFrame([{k: s[k] for k in ("pid", "trial_index", "step", "event", "time_to_event",
                                            "puzzle_fold", "participant_fold")} for s in snaps])
    states = [tuple(sorted(s["current_state"])) for s in snaps]
    reach = {st: reachable_24(st) for st in set(states)}
    df["state"] = ["-".join(f"{x:g}" for x in st) for st in states]
    df["coarse"] = [f"{len(st)}-{int(reach[st])}" for st in states]
    usable, y = within_k_labels(df.event, df.time_to_event)
    df["y"], df["usable"] = y, usable

    rows, cover = [], []
    for scheme in ("puzzle", "participant"):
        for fold in range(N_FOLDS):
            train, val, test = (mk.to_numpy() for mk in split_masks(df, scheme, fold))
            tr, va = df[train & usable], df[val & usable]
            best_m = max(M_GRID, key=lambda m: roc_auc_score(va.y, predict(tr, va, m)[0]))
            te = df[test]
            p, seen = predict(tr, te, best_m)
            part = te[["pid", "trial_index", "step"]].copy()
            part.insert(0, "model", "trap_state"); part.insert(1, "scheme", scheme)
            part.insert(2, "test_fold", fold); part.insert(3, "seed", 0)
            part["risk"], part["p_within3"] = p, p
            rows.append(part)
            cover.append((scheme, fold, best_m, seen.mean()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(pd.concat(rows, ignore_index=True), OUT)
    return pd.DataFrame(cover, columns=["scheme", "test_fold", "m", "exact_state_seen"])


if __name__ == "__main__":
    cover = run()
    print(cover.groupby("scheme").agg(m_chosen=("m", lambda s: list(s)), exact_state_seen=("exact_state_seen", "mean")).round(3))
    allpred = pd.concat([pd.read_csv(p) for p in sorted(OUT.parent.glob("*.csv"))], ignore_index=True)
    print(score(allpred).to_string(index=False))

# python -m src.baselines.trap_state