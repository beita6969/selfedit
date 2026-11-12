"""Leave-one-entry-out arms (coordinator 11-11): ood_wo_<k> (prio -7) then lrn_tipE (only where the tip's own E-path read is missing in the
cache) + lrn_wo_<k> (prio -6). Samples 0-1, E path; rows = where entry/group k is routed."""
import glob, json, os, subprocess, sys, time
V4 = "/data/fd/mingdazhang/selfedit/v4"; O = "/workspace/fudan/mingdazhang/selfedit-v4/ood"; X = f"{V4}/scratch/ext"
G = {"E1": "1", "E2": "2", "E5": "5", "E6": "6", "E9": "9", "E11": "11", "H": "3,4,7,8,10"}
def build(role, side):
    out = {}
    for k, ix in G.items():
        p = f"{X}/wo_{side}_{k}.jsonl"
        e = subprocess.run(["python3", f"{O}/tipview_wo.py", "t007c", role, "0,1", p, ix, f"{X}/wo_{side}_{k}_rows.json"], capture_output=True, text=True, cwd=V4)
        print(side, k, e.stderr.strip().splitlines()[-2:], flush=True); assert e.returncode == 0, e.stderr[-800:]
        out[k] = [json.loads(l) for l in open(p)]
    return out
def put(name, L):
    if not L: print(name, "empty, skipped"); return
    assert len({q["key"] for q in L}) == len(L) and all("edit" not in q and "ops" not in q for q in L)
    p = f"{V4}/q/in/{name}.jsonl"; open(p + ".tmp", "w").write("".join(json.dumps(x) + "\n" for x in L)); os.replace(p + ".tmp", p)
    open(f"{X}/submit_times.jsonl", "a").write(json.dumps(dict(batch=name, t_submit=time.time(), n=len(L))) + "\n")
    print(name, len(L), flush=True); time.sleep(1.1)
tot = 0
OD = build("ood_v2", "ood")
for k, L in OD.items():
    L = [dict(q, key=f"owo{k}|" + q["key"], engine="E", priority=-7) for q in L]; put(f"ood_wo_{k}", L); tot += len(L)
LD = build("show,monitor,judge", "lrn")
need = {}
for L in LD.values():
    for q in L: need[q["tip_key"]] = (q["row"], q["sample"])
have = {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if l.startswith(b'{"key": "x|'):
            k = l[9:l.index(b'"', 9)].decode()
            if k in need:
                x = json.loads(l)
                if not x.get("error") and x.get("content") is not None and x.get("engine_type") != "W": have[k] = os.path.basename(f)
miss = sorted(set(need) - set(have))
import collections
print(f"learn tip reference: {len(need)} (row, sample) needed, {len(have)} in cache (E path, tip's own keys) from {dict(collections.Counter(have.values()).most_common(6))}; missing {len(miss)}", flush=True)
json.dump(dict(have=have, miss=miss), open(f"{X}/lrn_tip_ref.json", "w"))
if miss:
    tv = f"{X}/tip_lrn_all.jsonl"
    subprocess.run(["python3", "/workspace/fudan/mingdazhang/selfedit-v4/engine_v4/tipview.py", "t007c", "show,monitor,judge", "0,1", tv], cwd=V4, capture_output=True)
    T = {q["key"]: q for q in map(json.loads, open(tv))}
    L = [dict(T[k], key="ltip|" + k, engine="E", priority=-6) for k in miss]; put("lrn_tipE", L); tot += len(L)
for k, L in LD.items():
    L = [dict(q, key=f"lwo{k}|" + q["key"], engine="E", priority=-6) for q in L]; put(f"lrn_wo_{k}", L); tot += len(L)
print("TOTAL new reads", tot)
