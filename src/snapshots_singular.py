from src.dataset.load import load_trials
from src.dataset.snapshots import snapshots_for_attempt

df = load_trials()
r = df.iloc[0] # first trial
a = snapshots_for_attempt(r.lm_code_translation, r.correct)
for s in a.snapshots:
    print(s.step, s.graph.number_of_edges(), s.event, s.time_to_event) 
    # first trial has 2 snapshots so 2 rows
    # each line has number, number of edges (increments), solved vs timed out, how many steps left (counts down)
    # 2 rows so reached 24 on 3rd operation