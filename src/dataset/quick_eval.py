# scorer, every model prediction can be compared the same way
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.dataset.cache import load_dataset
from src.dataset.predictions import within_k_labels


def harrell_c(t, event, risk) -> float:
    t, e, r = np.asarray(t, float), np.asarray(event, bool), np.asarray(risk, float)
    num = den = 0.0
    idx = np.where(e)[0]
    for start in range(0, len(idx), 400):
        ii = idx[start:start + 400]
        comparable = t[ii, None] < t[None, :]
        concordant = (r[ii, None] > r[None, :]) + 0.5 * (r[ii, None] == r[None, :])
        num += (comparable * concordant).sum()
        den += comparable.sum()
    return float(num / den)


def labels_table() -> pd.DataFrame:
    snaps, _ = load_dataset()
    return pd.DataFrame([{k: s[k] for k in ("pid", "trial_index", "step", "event", "time_to_event")}
                         for s in snaps])


def score(pred: pd.DataFrame, labels: pd.DataFrame = None) -> pd.DataFrame:
    labels = labels_table() if labels is None else labels
    m = pred.merge(labels, on=["pid", "trial_index", "step"], how="left", validate="many_to_one")
    assert m.event.notna().all(), "predictions refer to snapshots that do not exist"
    usable, y = within_k_labels(m.event, m.time_to_event)
    m["usable"], m["y"] = usable, y
    rows = []
    for (model, scheme, fold), g in m.groupby(["model", "scheme", "test_fold"]):
        u = g[g.usable]
        c = g[g.time_to_event > 0]
        rows.append(dict(model=model, scheme=scheme, test_fold=fold,
                         auroc_within3=roc_auc_score(u.y, u.p_within3),
                         c_index=harrell_c(c.time_to_event, c.event, c.risk)))
    per_fold = pd.DataFrame(rows)
    return (per_fold.groupby(["model", "scheme"])[["auroc_within3", "c_index"]].mean()
            .round(3).reset_index())


# per scheme, the mean over test folds of auroc_within3 AUROC for 'solves within the next 3 steps' (usable snapshots only), c_index Harrell's concordance of `risk` against time to solve (snapshots with a later step)