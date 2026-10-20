from src.dataset.load import load_trials
from src.dataset.snapshots import snapshots_for_attempt

df = load_trials()
total = events = 0
for r in df.itertuples():
    a = snapshots_for_attempt(r.lm_code_translation, r.correct)
    total += len(a.snapshots)
    events += sum(s.event for s in a.snapshots)
print(total, events) # prints 27136 8277, so 8277 are labeled solved, the rest are censored