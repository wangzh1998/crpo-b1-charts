import json, numpy as np
data=[]
with open("/root/AgenticRetrieve/CRPO/outputs/b1_deltas_full.jsonl") as f:
    for line in f: data.append(json.loads(line))

def membership(n):
    m=[]
    if 2<=n<=8: m.append("narrow")
    if 1<=n<=15: m.append("mid")
    if n>=16: m.append("tail")
    m.append("full")
    return m

pools={p:{"kept":[],"enabler":[],"dropped_other":[]} for p in ["narrow","mid","tail","full"]}
for d in data:
    ms=membership(d["n_search"])
    for r in d["rounds"]:
        g=r["group"]; dv=r["delta_prefix"]
        for p in ms: pools[p][g].append(dv)
for p in pools:
    for g in pools[p]: pools[p][g]=np.array(pools[p][g])

edges=np.arange(-4,8.0001,0.5)
labels=[f"{edges[i]:.1f}" for i in range(len(edges)-1)]
def hist(a):
    if len(a)==0: return [0]*(len(edges)-1)
    ac=np.clip(a,-3.999,7.999)
    c,_=np.histogram(ac,bins=edges)
    return (100.0*c/len(a)).round(3).tolist()

def pct(a,qs): return [round(float(np.percentile(a,q)),3) for q in qs]

OUT={"labels":labels,"pools":{}}
for p in pools:
    kp=pools[p]["kept"]; en=pools[p]["enabler"]; dr=pools[p]["dropped_other"]
    # contribution: top5% analysis
    def contrib(a):
        thr=np.percentile(a,95); top=a[a>=thr]; rest=a[a<thr]
        return round(float(top.mean()),3), round(float(rest.mean()),3), round(float(a.mean()),3)
    en_c=contrib(en); dr_c=contrib(dr)
    def possidep(a):
        pos=a[a>0]
        return pct(pos,[10,25,50,75,90,95,99]) if len(pos)>0 else [0]*7
    OUT["pools"][p]={
      "hist":{"kept":hist(kp),"enabler":hist(en),"dropped":hist(dr)},
      "summary":{g:{"n":int(len(a)),"mean":round(float(a.mean()),3),
                    "median":round(float(np.median(a)),3),
                    "pos":round(float(100*(a>0).mean()),1),
                    "std":round(float(a.std()),2)}
                 for g,a in [("kept",kp),("enabler",en),("dropped_other",dr)]},
      "keptfp":[round(float(100*(kp<t).mean()),2) for t in [0,-0.1,-0.5,-1.0,-2.0]],
      "posside":{"kept":possidep(kp),"enabler":possidep(en),"dropped":possidep(dr)},
      "percentile":{"enabler":pct(en,[1,5,10,25,50,75,90,95,99]),
                    "dropped":pct(dr,[1,5,10,25,50,75,90,95,99])},
      "contribution":[en_c[0],en_c[1],en_c[2],dr_c[0],dr_c[1],dr_c[2]],
      "tailratio":{"enabler":[round(float(100*(en>0).mean()),1),round(float(100*(np.abs(en)>2).mean()),2),
                              round(float(100*(en>2).mean()),2),round(float(100*(en<-2).mean()),2)],
                   "dropped":[round(float(100*(dr>0).mean()),1),round(float(100*(np.abs(dr)>2).mean()),2),
                              round(float(100*(dr>2).mean()),2),round(float(100*(dr<-2).mean()),2)]},
    }
print(json.dumps(OUT,ensure_ascii=False))
