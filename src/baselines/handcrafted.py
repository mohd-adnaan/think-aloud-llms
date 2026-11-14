import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.duration.hazard_regression import PHReg

from src.baselines.puzzle_only import puzzle_features
from src.dataset.cache import load_dataset
from src.dataset.features import EDGE_COLS, NODE_COLS
from src.dataset.folds import N_FOLDS, split_masks
from src.dataset.load import ROOT
from src.dataset.predictions import HORIZON, save_predictions, within_k_labels
from src.dataset.quick_eval import score

OUT_DIR = ROOT / "data/predictions"
# near-duplicates (|corr| >= 0.97) that stop the unpenalised Cox fit from converging
COX_DROP = ["n_op_edges", "dead_states", "cur_numbers"]
N, E = NODE_COLS.index, EDGE_COLS.index


def snapshot_features(s) -> dict:
    x, ei, ea = s["x"], s["edge_index"], s["edge_attr"]
    has_depth = x[:, N("has_depth")] == 1
    ops_edge = ea[:, E("is_subgoal")] == 0
    cur = np.where(x[:, N("is_current")] == 1)[0]
    cur = cur[0] if len(cur) else None
    sources = ei[0][ops_edge]
    out_deg = np.bincount(sources, minlength=len(x)) if len(sources) else np.zeros(len(x))
    non_start = has_depth & (x[:, N("is_start")] == 0)
    visits = x[:, N("visit_count")]
    return dict(
        step=s["step"], n_backtracks=s["n_backtracks"], n_subgoals=s["n_subgoals"],
        n_states=int(has_depth.sum()), n_op_edges=int(ops_edge.sum()),
        max_depth=float(x[has_depth, N("depth")].max()) if has_depth.any() else 0.0,
        mean_branching=float(ops_edge.sum() / max((out_deg > 0).sum(), 1)),
        max_branching=float(out_deg.max()) if len(out_deg) else 0.0,
        revisits=float(np.clip(visits - 1, 0, None).sum()),
        live_states=int((x[non_start, N("reach24")] == 1).sum()),
        dead_states=int((x[non_start, N("reach24")] == 0).sum()),
        calc_errors=int((ea[ops_edge, E("is_correct")] == 0).sum()),
        cur_reach24=float(x[cur, N("reach24")]) if cur is not None else 0.0,
        cur_depth=float(x[cur, N("depth")]) if cur is not None else 0.0,
        cur_visits=float(visits[cur]) if cur is not None else 0.0,
        cur_numbers=float(x[cur, N("n_numbers")]) if cur is not None else 0.0,
    )


def build_table():
    snaps, _ = load_dataset()
    meta = pd.DataFrame([{k: s[k] for k in ("pid", "trial_index", "puzzle", "step", "event", "time_to_event",
                                              "puzzle_fold", "participant_fold")} for s in snaps])
    feats = pd.DataFrame([snapshot_features(s) for s in snaps])
    pz = pd.DataFrame({p: puzzle_features(p) for p in meta.puzzle.unique()}).T
    pz = pz.loc[meta.puzzle].reset_index(drop=True)
    return meta, feats, pz


def _tune_gbm(Xtr, ytr, Xva, yva):
    best, best_auc = None, -1
    for lr, depth, iters in itertools.product((0.05, 0.1), (2, 4), (100, 300)):
        m = HistGradientBoostingClassifier(learning_rate=lr, max_depth=depth, max_iter=iters,
                                           random_state=0).fit(Xtr, ytr)
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        if auc > best_auc:
            best, best_auc = dict(learning_rate=lr, max_depth=depth, max_iter=iters), auc
    return HistGradientBoostingClassifier(random_state=0, **best).fit(Xtr, ytr), best


def _cox(Xtr, t, e, Xte):
    sc = StandardScaler().fit(Xtr)
    res = PHReg(t, sc.transform(Xtr), status=e, ties="breslow").fit(disp=0)
    lhr = np.asarray(res.predict(exog=sc.transform(Xte), pred_type="lhr").predicted_values)
    times, cumhaz, _ = res.baseline_cumulative_hazard[0]
    # statsmodels reports the cumulative hazard just BEFORE the events at each time, so the hazard
    # through step HORIZON is the value at the first time after it
    i = np.searchsorted(times, HORIZON, side="right")
    h0 = cumhaz[i] if i < len(times) else cumhaz[-1]
    return lhr, 1 - np.exp(-h0 * np.exp(lhr))


# logistic regression, gradient boosting and Cox
# Features come from each snapshot's graph and counts (no step beyond the cut, no reached-goal flag). Models are fit on the training folds; gradient boosting is tuned on the validation fold only.
def run():
    meta, feats, pz = build_table()
    usable, y = within_k_labels(meta.event, meta.time_to_event)
    Xg = feats.to_numpy(float)
    Xgp = np.hstack([Xg, pz.to_numpy(float)])
    Xc = feats.drop(columns=COX_DROP).to_numpy(float)
    preds = {k: [] for k in ("logreg_graph", "gbm_graph", "gbm_graph_puzzle", "cox_graph")}
    chosen = []
    for scheme in ("puzzle", "participant"):
        for fold in range(N_FOLDS):
            train, val, test = (m.to_numpy() for m in split_masks(meta, scheme, fold))
            tr, va = train & usable, val & usable
            key = meta.loc[test, ["pid", "trial_index", "step"]].copy()

            def add(name, risk, p):
                part = key.copy()
                part.insert(0, "model", name); part.insert(1, "scheme", scheme)
                part.insert(2, "test_fold", fold); part.insert(3, "seed", 0)
                part["risk"], part["p_within3"] = risk, p
                preds[name].append(part)

            sc = StandardScaler().fit(Xg[tr])
            lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xg[tr]), y[tr])
            p = lr.predict_proba(sc.transform(Xg[test]))[:, 1]
            add("logreg_graph", p, p)

            for name, X in (("gbm_graph", Xg), ("gbm_graph_puzzle", Xgp)):
                m, params = _tune_gbm(X[tr], y[tr], X[va], y[va])
                p = m.predict_proba(X[test])[:, 1]
                add(name, p, p)
                chosen.append((name, scheme, fold, params))

            fit = train & (meta.time_to_event.to_numpy() > 0)
            lhr, p = _cox(Xc[fit], meta.time_to_event.to_numpy()[fit], meta.event.to_numpy()[fit], Xc[test])
            add("cox_graph", lhr, p)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, parts in preds.items():
        save_predictions(pd.concat(parts, ignore_index=True), OUT_DIR / f"{name}.csv")
    return chosen, list(feats.columns)


if __name__ == "__main__":
    chosen, cols = run()
    print("features:", cols)
    allpred = pd.concat([pd.read_csv(p) for p in sorted(OUT_DIR.glob("*.csv"))], ignore_index=True)
    print(score(allpred).to_string(index=False))

# python -m src.baselines.handcrafted