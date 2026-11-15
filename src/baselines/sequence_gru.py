"""
Per step the GRU sees 7 numbers: can the current state still reach 24, how many numbers are left,
the step number, new backtracks, new subgoals, new arithmetic errors, and whether the person is in a
state they have been in before. It predicts 'solves within the next 3 steps' at every step (causal).
Trained on the training folds, early-stopped on the validation fold, 3 seeds per split.
"""

# sequence only gru, read each attempt as sequence of moves w/o the graph
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.dataset.cache import load_dataset
from src.dataset.features import EDGE_COLS, NODE_COLS
from src.dataset.folds import N_FOLDS
from src.dataset.load import ROOT
from src.dataset.predictions import save_predictions, within_k_labels
from src.dataset.quick_eval import score

OUT = ROOT / "data/predictions/gru_sequence.csv"
SEEDS = (0, 1, 2)
# SEEDS = (0,) # for smoke test
N, E = NODE_COLS.index, EDGE_COLS.index
N_INPUT = 7


def build_sequences(snaps):
    """Group snapshots into attempts. Returns X (n_seq, L, 7), rows (n_seq, L) with the index of the
    snapshot at each position (-1 = padding), and the keys. Everything here is causal."""
    groups = {}
    for i, s in enumerate(snaps):
        groups.setdefault((s["pid"], s["trial_index"]), []).append(i)
    keys = list(groups)
    L = max(len(v) for v in groups.values())
    X = np.zeros((len(keys), L, N_INPUT), dtype=np.float32)
    rows = -np.ones((len(keys), L), dtype=np.int64)
    for q, key in enumerate(keys):
        idx = sorted(groups[key], key=lambda i: snaps[i]["step"])
        prev_b = prev_s = prev_c = 0
        for t, i in enumerate(idx):
            s = snaps[i]
            x, ea = s["x"], s["edge_attr"]
            cur = np.where(x[:, N("is_current")] == 1)[0]
            cur = cur[0] if len(cur) else None
            calc = int(((ea[:, E("is_correct")] == 0) & (ea[:, E("is_subgoal")] == 0)).sum())
            X[q, t] = [
                x[cur, N("reach24")] if cur is not None else 0.0,
                x[cur, N("n_numbers")] / 4 if cur is not None else 0.0,
                s["step"] / 10,
                s["n_backtracks"] - prev_b,
                s["n_subgoals"] - prev_s,
                max(calc - prev_c, 0),
                float(x[cur, N("visit_count")] > 1) if cur is not None else 0.0,
            ]
            prev_b, prev_s, prev_c = s["n_backtracks"], s["n_subgoals"], calc
            rows[q, t] = i
    return X, rows, keys


def _auroc(y, p, mask):
    return roc_auc_score(y[mask], p[mask])


def train_one(Xtr, ytr, mtr, Xva, yva, mva, seed, hidden=32, max_epochs=80, patience=10, lr=3e-3, batch=128):
# def train_one(Xtr, ytr, mtr, Xva, yva, mva, seed, hidden=32, max_epochs=5, patience=10, lr=3e-3, batch=128): # for smoke test
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(N_INPUT, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            return self.head(self.gru(x)[0]).squeeze(-1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    tx, ty, tm = (torch.tensor(a) for a in (Xtr, ytr.astype(np.float32), mtr.astype(np.float32)))
    best, best_state, bad = -1.0, None, 0
    for _ in range(max_epochs):
        net.train()
        order = rng.permutation(len(tx))
        for b in range(0, len(order), batch):
            ii = order[b:b + batch]
            loss = (lossf(net(tx[ii]), ty[ii]) * tm[ii]).sum() / tm[ii].sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        pv = predict(net, Xva)
        auc = _auroc(yva, pv, mva)
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net


def predict(net, X):
    import torch
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(X))).numpy()


def run():
    snaps, _ = load_dataset()
    X, rows, keys = build_sequences(snaps)
    valid = rows >= 0
    ev = np.array([s["event"] for s in snaps]); tte = np.array([s["time_to_event"] for s in snaps])
    usable_s, y_s = within_k_labels(ev, tte)
    ysq = np.where(valid, y_s[rows.clip(0)], 0)
    msk = valid & np.where(valid, usable_s[rows.clip(0)], False)
    fold = {s: np.array([snaps[r[0]][f"{s}_fold"] for r in rows]) for s in ("puzzle", "participant")}
    out = []
    for scheme in ("puzzle", "participant"):
        f = fold[scheme]
        for test_fold in range(N_FOLDS):
            te, va = f == test_fold, f == (test_fold + 1) % N_FOLDS
            tr = ~(te | va)
            for seed in SEEDS:
                net = train_one(X[tr], ysq[tr], msk[tr], X[va], ysq[va], msk[va], seed)
                p = predict(net, X[te])
                r = rows[te]
                ok = r >= 0
                part = pd.DataFrame({
                    "pid": [snaps[i]["pid"] for i in r[ok]], "trial_index": [snaps[i]["trial_index"] for i in r[ok]],
                    "step": [snaps[i]["step"] for i in r[ok]], "risk": p[ok], "p_within3": p[ok]})
                part.insert(0, "model", "gru_sequence"); part.insert(1, "scheme", scheme)
                part.insert(2, "test_fold", test_fold); part.insert(3, "seed", seed)
                out.append(part)
                print(scheme, test_fold, seed, "val-selected model done", flush=True)
    pred = pd.concat(out, ignore_index=True)
    save_predictions(pred, OUT)
    return pred


if __name__ == "__main__":
    run()
    allpred = pd.concat([pd.read_csv(p) for p in sorted(OUT.parent.glob("*.csv"))], ignore_index=True)
    print(score(allpred).to_string(index=False))

# python -m src.baselines.sequence_gru