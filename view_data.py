import pandas as pd
from src.dataset.cache import load_dataset, load_attempts

snaps, meta = load_dataset() # list of dicts + info dict
print(meta["n_snapshots"], len(snaps)) # 27136 27136

s = snaps[1] # look at 1 snapshot
print(list(s.keys()))
print(s["x"]) # node feature array
print(s["edge_index"], s["edge_attr"]) # edges and edge features

df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("x", "edge_index", "edge_attr")} for r in snaps])
print(df.head())
print(df.event.value_counts()) # expects 0: 18859, 1: 8277 <- censored vs solved
print(load_attempts().head()) # attempts table