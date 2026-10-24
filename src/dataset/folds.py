import json
import numpy as np
import pandas as pd
from src.dataset.load import ROOT, load_trials

N_FOLDS = 5
SEED = 0
FOLDS_PATH = ROOT / "data/splits/folds.csv"

def puzzle_id(choices: str) -> str:
    # '[5,12,5,12]' -> '5-5-12-12' (the same puzzle regardless of order) 
    return "-".join(str(n) for n in sorted(json.loads(choices)))

# deciding which fold each puzzle / participant goes into
def _assign(groups: pd.Series, n_folds: int, seed: int) -> dict:
    # each group one fold, balance num trials per fold
    sizes = groups.value_counts()
    rng = np.random.default_rng(seed)
    names = list(sizes.index)
    rng.shuffle(names) # tie breaking
    names.sort(key=lambda g: -sizes[g]) # largest groups first
    load = [0] * n_folds
    fold_of = {}

    for g in names:
        f = int(np.argmin(load)) # finding the fold w/ fewest trials, assign group
        fold_of[g] = f 
        load[f] += int(sizes[g])

    return fold_of

def make_folds(overwrite: bool = False) -> pd.DataFrame:
    if FOLDS_PATH.exists() and not overwrite:
        raise FileExistsError(f"{FOLDS_PATH} already exists; load it with load_folds() instead")
    
    df = load_trials()
    out = df[["pid", "trial_index"]].copy() # table and adds puzzle label below
    out["puzzle"] = df["choices"].map(puzzle_id)

    # 5 fold splits
    # outer test fold i uses fold (i + 1) % 5 as validation fold for tuning and remaining folds for training
    out["puzzle_fold"] = out["puzzle"].map(_assign(out["puzzle"], N_FOLDS, SEED)) # map each puzzle to its fold
    out["participant_fold"] = out["pid"].map(_assign(out["pid"], N_FOLDS, SEED))

    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(FOLDS_PATH, index=False)

    return out

def load_folds() -> pd.DataFrame:
    return pd.read_csv(FOLDS_PATH) # use this one

# folds table, scheme is either "puzzle" or "participant", and which fold is test set
def split_masks(folds: pd.DataFrame, scheme: str, test_fold: int):
    f = folds[f"{scheme}_fold"]
    val_fold = (test_fold + 1) % N_FOLDS 
    test = f == test_fold # true / false marking rows
    val = f == val_fold

    return ~(test | val), val, test # train is everything else aka rows in neither val nor test
    # grouped by puzzle (unseen puzzles) and by participant (unseen people)

if __name__ == "__main__":
    folds = make_folds()
    print("wrote", FOLDS_PATH, len(folds), "trials")
    # everyone loading the splits / folds file <- 4947 trials