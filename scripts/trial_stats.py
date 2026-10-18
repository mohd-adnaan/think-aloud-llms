# running each trial's code through GraphBuilder, summarize step counts
import sys
import numpy as np
import pandas as pd

from src.dataset.load import load_trials
from src.preproc.reasoning_graph import GraphBuilder
from src.preproc.utils import run_code

coder = sys.argv[1] if len(sys.argv) > 1 else "claude-3-5-sonnet-20241022" # python scripts/trial_stats.py [coder] <- default claude
df = load_trials(coder)

rows = []
for r in df.itertuples():
    g = run_code(r.lm_code_translation, for_pretraining = False)
    if not isinstance(g, GraphBuilder):
        rows.append(dict(pid = r.pid, trial_index = r.trial_index, ok = False,
                         error = str(g).strip().splitlines()[-1][:120]))
        continue
    kinds = [a["type"] for a in g.actions]
    n_sub = kinds.count("set_subgoal")
    rows.append(dict(
        # columns for the resulting csv file: 
        pid = r.pid, trial_index = r.trial_index, ok = True, error = "",
        correct = r.correct, rt_s = r.rt_s,
        steps_A = kinds.count("explore_operation"), # explore_operation calls <- most likely go with this option
        steps_B = g.op_timestep - 1 - n_sub, # edges, subgoal edges excluded
        steps_B_with_subgoals = g.op_timestep - 1, # subgoal edges counted
        n_backtracks = kinds.count("move_to_node"),
        n_subgoals = n_sub,
    ))

out = pd.DataFrame(rows)
# out.to_csv("trial_stats.csv", index = False)
out.to_csv(f"trial_stats_{coder}.csv", index=False)

ok = out[out.ok]
# checking coder mistakes leading to failed runs: 
print(f"coder {coder}: {len(out)} trials, {(~out.ok).sum()} failed to run ({(~out.ok).mean():.1%})") 
if (~out.ok).any():
    print(out[~out.ok].error.value_counts().head(5).to_string())
for col in ["steps_A", "steps_B", "steps_B_with_subgoals"]:
    s = ok[col]
    print(f"{col}: total {int(s.sum())}, median {s.median():.0f}; "
          f"solved median {ok[ok.correct == 1][col].median():.0f}, "
          f"unsolved median {ok[ok.correct == 0][col].median():.0f}; "
          f"share <= 10: {(s <= 10).mean():.1%}")
print("move_to_node total:", int(ok.n_backtracks.sum()), "| subgoal actions total:", int(ok.n_subgoals.sum()))
# median 5 operations per attempt which matches A