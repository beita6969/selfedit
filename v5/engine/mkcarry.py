"""Hand carry-in reads (coordinator 22:5x): for each candidate, its recorded step-1 requests (sample 0) copied unchanged with
sample s in (1, 2, 3) and key ...|0 -> ...|s (= the loop's read_req(entries, row, s)). Matching: setting candidates by their settings,
text candidates by their procedure text inside the request prefix. usage: python mkcarry.py [go]"""
import json, glob, sys, os, collections
V4 = "/data/fd/mingdazhang/selfedit/v4"; R = f"{V4}/runs/main_1110"
CANDS = ["g003c31", "g004c12", "g003c00", "g003c16", "g004c11"]
PRIO = int(os.environ.get("PRIO", "6"))
role = {json.loads(l)["id"]: json.loads(l)["v4_role"] for l in open(f"{V4}/rows/learn.jsonl")}
led = {}
for l in open(f"{R}/ledger.jsonl"):
    d = json.loads(l)
    if d.get("id") in CANDS and "payload" in d:
        led[d["id"]] = d
import hashlib
STOCK = {"min_image_tokens": None, "max_image_tokens": 16384, "temperature": 1.0}
def sig(prefix, req_st):
    st = {k: (STOCK[k] if v is None else v) for k, v in req_st.items()}
    nz = {k: v for k, v in sorted(st.items()) if v != STOCK.get(k)}
    return hashlib.sha256(json.dumps([prefix or None, nz], sort_keys=True).encode()).hexdigest()[:16]
par = {}
for l in open(glob.glob(f"{V4}/q/in/g004_parentbg_*.jsonl")[0]):
    q = json.loads(l)
    if q["sample"] == 4:
        par[q["row"]] = q
bad = sum(sig(q.get("prefix"), q["settings"]) != q["key"].split("|")[1] for q in par.values())
print(f"t002 parent sig check: {len(par)} rows, mismatches {bad}")
go = len(sys.argv) > 1 and sys.argv[1] == "go"
for c in CANDS:
    d = led[c]; p = d["payload"]; g = c[1:4]
    reqs = []
    for f in glob.glob(f"{V4}/q/in/g{g}_rung1_*.jsonl"):
        for l in open(f):
            q = json.loads(l)
            if q.get("sample") != 0:
                continue
            if d["kind"] == "setting":
                st = {k: v for k, v in (q.get("settings") or {}).items() if v is not None}
                want = dict(p["settings"]); want.setdefault("min_image_tokens", 4096)
                if st == want and not q.get("ops"):
                    reqs.append(q)
            else:
                if p.get("procedure") and q.get("prefix") and p["procedure"] in q["prefix"]:
                    reqs.append(q)
    if d["kind"] == "setting":
        rec = {q["row"]: q for q in reqs}
        reqs = []
        for r, q0 in par.items():
            if role[r] != "judge":
                continue
            st = dict(q0["settings"]); st.update(p["settings"])
            q = dict(key=f"x|{sig(q0.get('prefix'), st)}|{r}|0", mode="read", engine="E", priority=1, row=r, sample=0, prefix=q0.get("prefix"), settings=st)
            if r in rec:
                assert rec[r]["key"] == q["key"] and rec[r].get("prefix") == q["prefix"], (rec[r]["key"], q["key"])
            reqs.append(q)
        print(f"  {c}: built {len(reqs)} from the t002 parent view; {len(rec)} agree exactly with the recorded step-1 requests")
    rows = collections.Counter(q["row"] for q in reqs)
    dup = sum(v > 1 for v in rows.values())
    rl = collections.Counter(role[q["row"]] for q in reqs)
    ops = sum(bool(q.get("ops")) for q in reqs)
    print(f"{c}: kind {d['kind']} routed {d['routed']} | matched {len(reqs)} reqs, {len(rows)} rows, dup rows {dup}, roles {dict(rl)}, with ops {ops}, sigs {len({q['key'].split('|')[1] for q in reqs})}")
    if go and reqs and not dup and len(rows) == int(d["routed"]):
        out = []
        for q in reqs:
            if role[q["row"]] != "judge":
                continue
            for s in (1, 2, 3):
                q2 = dict(q); q2["sample"] = s; q2["priority"] = PRIO + min(CANDS.index(c), 2)
                assert q["key"].endswith("|0"); q2["key"] = q["key"][:-2] + f"|{s}"
                out.append(q2)
        b = f"carry_{c}"
        open(f"{V4}/q/in/.{b}.tmp", "w").write("".join(json.dumps(x) + "\n" for x in out))
        os.replace(f"{V4}/q/in/.{b}.tmp", f"{V4}/q/in/{b}.jsonl")
        print(f"  submitted {b}: {len(out)} reads")
