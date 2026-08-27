import json
import numpy as np
from collections import defaultdict

data = []
with open("/root/AgenticRetrieve/CRPO/outputs/b1_deltas_full.jsonl") as f:
    for line in f:
        data.append(json.loads(line))

def pool_of(n):
    pools = []
    if 2 <= n <= 8: pools.append("narrow")
    if 1 <= n <= 15: pools.append("mid")
    if n >= 16: pools.append("tail")
    return pools

# collect delta by (pool, group). Pools are nested; a traj can belong to narrow(⊂mid) etc.
# We report each pool as a standalone slice (narrow, mid, tail, full) x group.
buckets = defaultdict(list)  # (poolname, group) -> list of deltas

for d in data:
    n = d["n_search"]
    membership = []
    if 2 <= n <= 8: membership.append("narrow")
    if 1 <= n <= 15: membership.append("mid")
    if n >= 16:      membership.append("tail")
    membership.append("full")  # everyone in full
    for r in d["rounds"]:
        g = r["group"]; dv = r["delta_prefix"]
        for pool in membership:
            buckets[(pool, g)].append(dv)

def stats(arr):
    a = np.array(arr)
    return {
        "n": len(a),
        "mean": a.mean(),
        "median": float(np.median(a)),
        "pos": 100.0*(a>0).mean(),
        "p10": float(np.percentile(a,10)),
        "p90": float(np.percentile(a,90)),
        "std": a.std(),
        "neg_strong": 100.0*(a < -0.5).mean(),   # fraction strongly negative
        "pos_strong": 100.0*(a >  0.5).mean(),   # fraction strongly positive
    }

pools = ["narrow", "mid", "tail", "full"]
groups = ["kept", "enabler", "dropped_other"]

print("=== POOL x GROUP cross table ===")
hdr = f"{'pool':7} {'group':14} {'n':>8} {'mean':>8} {'median':>8} {'pos%':>6} {'p10':>8} {'p90':>8} {'std':>7} {'<-0.5%':>7} {'>0.5%':>7}"
print(hdr)
print("-"*len(hdr))
for pool in pools:
    for g in groups:
        s = stats(buckets[(pool,g)])
        print(f"{pool:7} {g:14} {s['n']:>8,} {s['mean']:>+8.3f} {s['median']:>+8.3f} {s['pos']:>5.1f} {s['p10']:>+8.3f} {s['p90']:>+8.3f} {s['std']:>7.2f} {s['neg_strong']:>6.1f} {s['pos_strong']:>6.1f}")
    print()

# Focus: full pool enabler anomaly — mean up, median/pos down vs dropped
print("=== ENABLER anomaly deep-dive (full pool) ===")
en = np.array(buckets[("full","enabler")])
dr = np.array(buckets[("full","dropped_other")])
for name, a in [("enabler", en), ("dropped_other", dr)]:
    print(f"{name}: n={len(a):,} mean={a.mean():+.3f} median={np.median(a):+.3f} "
          f"pos%={100*(a>0).mean():.1f} std={a.std():.2f}")
    for q in [1,5,10,25,50,75,90,95,99]:
        print(f"    p{q:<2} = {np.percentile(a,q):+.3f}", end="")
    print()
    # tail mass
    print(f"    share |Δ|>2: {100*(np.abs(a)>2).mean():.2f}%   Δ>2: {100*(a>2).mean():.2f}%   Δ<-2: {100*(a<-2).mean():.2f}%")
    print(f"    contribution of top-5% to mean: ", end="")
    thr = np.percentile(a,95)
    top = a[a>=thr]
    print(f"top5% mean={top.mean():+.2f}, they are {len(top)} rounds, "
          f"remove them -> mean of rest={a[a<thr].mean():+.3f}")
