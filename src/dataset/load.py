from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] 
EXPERIMENT = "full-experiment"
DEFAULT_CODER = "claude-3-5-sonnet-20241022"

def excluded_pids() -> list:
    # only keep participants with relevant transcripts
    trials = pd.read_csv(ROOT / f"data/processed/{EXPERIMENT}/{EXPERIMENT}-trials.csv")
    trials = trials.query("not practice")
    ratio = trials.groupby("pid")["relevant"].transform("mean")

    return list(trials.loc[ratio <= 0.5, "pid"].unique())

def load_trials(coder: str = DEFAULT_CODER) -> pd.DataFrame:
    # returning trials for a coder, with exclusions applied
    path = ROOT / f"data/coded/{EXPERIMENT}/{EXPERIMENT}_model-{coder}.csv"
    exclude = excluded_pids()

    df = pd.read_csv(path)
    df = (
        df.query("not practice")
        .query("choices != '[1,1,2,6]'")
        .assign(trial_index = lambda d: d.groupby("pid").cumcount() + 1)
        .query("relevant == 1")
        .query("pid not in @exclude")
        # trials that ran over 181 s because of lag count as unsolved
        .assign(correct = lambda d: d["correct"].where(d["rt_s"] <= 181, 0.0).astype(float))
        .assign(rt_s = lambda d: d["rt_s"].clip(upper = 180))
        .reset_index(drop=True)
    )
    return df

if __name__ == "__main__":
    d = load_trials()
    print("trials", len(d), "solved", int(d.correct.sum()),
          "unsolved", int((d.correct == 0).sum()), "participants", d.pid.nunique())
    # 4947 trials, 2647 solved, 2300 unsolved, 541 participants
    assert (len(d), int(d.correct.sum()), int((d.correct == 0).sum())) == (4947, 2647, 2300) # loading the specific trials 
    print("Counts match") # AssertionError if doesn't match
    # referenced from notebooks/MainAnalyses.ipynb
    # python -m src.dataset.load 