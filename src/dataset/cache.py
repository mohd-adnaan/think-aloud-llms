import pickle
import pandas as pd
from src.dataset.features import EDGE_COLS, NODE_COLS, featurize
from src.dataset.folds import load_folds
from src.dataset.load import DEFAULT_CODER, ROOT, load_trials
from src.dataset.snapshots import snapshots_for_attempt

CACHE_DIR = ROOT / "data/cache"
DATASET_PATH = CACHE_DIR / "dataset_v1.pkl"
ATTEMPTS_PATH = CACHE_DIR / "attempts_v1.csv"

def build(coder: str = DEFAULT_CODER):
    trials = load_trials(coder)
    folds = load_folds().set_index(["pid", "trial_index"])
    snapshots, attempts = [], []

    for r in trials.itertuples():
        a = snapshots_for_attempt(r.lm_code_translation, r.correct)
        if a is None: # if didn't run, leave out
            attempts.append(dict(pid=r.pid, trial_index=r.trial_index, ran=False))
            continue

        fo = folds.loc[(r.pid, r.trial_index)]
        for s in a.snapshots:
            f = featurize(s.graph, s.current_state) # snapshots w/ features, labels, fold column
            snapshots.append(dict(
                pid=r.pid, trial_index=r.trial_index, puzzle=fo["puzzle"],
                puzzle_fold=int(fo["puzzle_fold"]), participant_fold=int(fo["participant_fold"]),
                step=s.step, event=s.event, time_to_event=s.time_to_event,
                n_backtracks=s.n_backtracks, n_subgoals=s.n_subgoals,
                x=f["x"], edge_index=f["edge_index"], edge_attr=f["edge_attr"],
            ))

        # attempts for further analysis
        attempts.append(dict(
            pid=r.pid, trial_index=r.trial_index, ran=True, puzzle=fo["puzzle"],
            correct=r.correct, rt_s=r.rt_s, # rt_s for analysis only, never model input
            n_ops=a.n_ops, solve_step=a.solve_step, n_snapshots=len(a.snapshots),
            flags=";".join(a.flags),
        ))

    meta = dict(coder=coder, n_attempts=len(trials), n_snapshots=len(snapshots),
                node_cols=NODE_COLS, edge_cols=EDGE_COLS)
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "wb") as fh:
        pickle.dump(dict(meta=meta, snapshots=snapshots), fh, protocol=4)
    pd.DataFrame(attempts).to_csv(ATTEMPTS_PATH, index=False)

    return meta

def load_dataset():
    # (snapshots, meta), snapshot = dict
    with open(DATASET_PATH, "rb") as fh:
        d = pickle.load(fh)
    return d["snapshots"], d["meta"]

def load_attempts() -> pd.DataFrame:
    return pd.read_csv(ATTEMPTS_PATH)


if __name__ == "__main__":
    print(build()) # can be rebuilt b/c data deterministic
    # cached dataset = one record per snapshot, vs attempts = one row per attempt
    # python -m src.dataset.cache