import json, numpy as np
data=[]
with open("/root/AgenticRetrieve/CRPO/outputs/b1_deltas_full.jsonl") as f:
    for line in f: data.append(json.loads(line))
g={"kept":[],"enabler":[],"dropped_other":[]}
for d in data:
    for r in d["rounds"]:
        g[r["group"]].append(r["delta_prefix"])
for k in g: g[k]=np.array(g[k])

edges=np.arange(-4,8.0001,0.5)
labels=[f"{edges[i]:.1f}" for i in range(len(edges)-1)]
def hist(a):
    ac=np.clip(a,-3.999,7.999)
    c,_=np.histogram(ac,bins=edges)
    return (100.0*c/len(a)).round(3).tolist()

print("=== group summary (full pool) ===")
for k,a in g.items():
    print(f"{k:14} n={len(a):>7,} mean={a.mean():+.3f} median={np.median(a):+.3f} "
          f"pos%={100*(a>0).mean():.1f} std={a.std():.2f}")

print("\n=== kept: false-positive-rate under Δ̃ criterion ===")
kp=g["kept"]
for thr,name in [(0,"Δ<0"),(-0.1,"Δ<-0.1"),(-0.5,"Δ<-0.5"),(-1.0,"Δ<-1.0"),(-2.0,"Δ<-2.0")]:
    print(f"  {name:8}: {100*(kp<thr).mean():5.2f}%")

print("\n=== positive-side shape: among Δ>0 rounds ===")
for k,a in g.items():
    pos=a[a>0]
    print(f"{k:14} P(Δ>0)={100*(a>0).mean():5.1f}%  |  among Δ>0: "
          f"mean={pos.mean():+.3f} median={np.median(pos):+.3f} "
          f"p50={np.percentile(pos,50):+.3f} p90={np.percentile(pos,90):+.3f} p99={np.percentile(pos,99):+.3f}")

print("\n=== kept vs enabler positive-side percentiles ===")
for k in ["kept","enabler","dropped_other"]:
    pos=g[k][g[k]>0]
    qs=[np.percentile(pos,q) for q in [10,25,50,75,90,95,99]]
    print(f"{k:14} " + " ".join(f"p{q}={v:+.3f}" for q,v in zip([10,25,50,75,90,95,99],qs)))

print("\n=== HISTJSON ===")
print(json.dumps({
  "labels":labels,
  "kept":hist(g["kept"]),
  "enabler":hist(g["enabler"]),
  "dropped":hist(g["dropped_other"]),
}, ensure_ascii=False))
