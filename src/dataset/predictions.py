# shared format for every model so single evaluator scores them all
import numpy as np
import pandas as pd

REQUIRED = ["model", "scheme", "test_fold", "seed", "pid", "trial_index", "step", "risk", "p_within3"]
# model (name), scheme ("puzzle" or "participant"), test fold (0-4, the fold this row was predicted as part of test set), seed, pid, trial index, step, risk (higher = expected to solve sooner -> concordance index), p_within3 (predicted prob of solving within next 3 steps)

HORIZON = 3

def save_predictions(df: pd.DataFrame, path) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    assert not missing, f"missing columns: {missing}"
    assert df.p_within3.between(0, 1).all(), "p_within3 must be a probability"
    assert np.isfinite(df.risk).all(), "risk must be finite"
    df[REQUIRED].to_csv(path, index=False)

def load_predictions(path) -> pd.DataFrame:
    return pd.read_csv(path)

def within_k_labels(event, time_to_event, k: int = HORIZON):
    # solves within the next k steps 
    event = np.asarray(event)
    tte = np.asarray(time_to_event)
    usable = ~((event == 0) & (tte < k))
    y = ((event == 1) & (tte <= k)).astype(int)
    return usable, y 
    # censored snapshot with fewer than k observed steps left can't be labeled so marked unusable
# every model must be scored on the same usable rows