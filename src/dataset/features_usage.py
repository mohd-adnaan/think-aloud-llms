from src.dataset.load import load_trials
from src.dataset.snapshots import snapshots_for_attempt
from src.dataset.features import featurize

df = load_trials()
r = df.iloc[0] # pick a trial
a = snapshots_for_attempt(r.lm_code_translation, r.correct)

s = a.snapshots[1] # pick snapshot
f = featurize(s.graph, s.current_state) # compute features

print("step", s.step, "event", s.event, "time to event", s.time_to_event)  # label for this snapshot
print("node features:")
print(f["x"]) # node features
# visit count, depth from the start state, whether it has a depth, how many numbers are left, whether 24 is still reachable, is it the start, current state, goal, subgoal node
# each row is a different state / node
# row 0 = start with the 4 nums, row 1 = (24,) aka goal, for example row 3 sets is_current to 1

print("edges indices:")
print(f["edge_index"]) # source and target rows

print("edge attributions:")
print(f["edge_attr"]) # row per edge describing what the operation is, for example if the transition is 12+5 col 0 is set to 1
# which operation it is (+, −, ×, ÷, other), whether it’s a subgoal edge, how many times it was tried, whether the arithmetic was correct