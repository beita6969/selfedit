"""Weight-cell table vs a concurrent control: python3 cell_report.py NAME CELL_BATCHES [--ref REF_BATCHES] [--tip8 scratch/ext/tip_signal8.jsonl] [--gate 1500]
Primary: cell - ref (W-null), same row and sample index, row-clustered se. Secondary: cell - the tip's cached row mean (all cached samples of
the tip view, up to 8). Also vs the cached same-index sample. Reads cut at max_tokens are counted separately (wrong by construction).
Classes: gated_tip = tip's cached same-index read > gate tokens; gated_ref = the W-null read > gate tokens."""
import argparse, collections, glob, json, math, os, statistics
V4 = "/data/fd/mingdazhang/selfedit/v4"
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("cells"); ap.add_argument("--ref", default="")
ap.add_argument("--tip8", default=f"{V4}/scratch/ext/tip_signal8.jsonl"); ap.add_argument("--gate", type=int, default=1500); ap.add_argument("--fired", default=f"{V4}/scratch/ext/tip_fired.json")
a = ap.parse_args()
def load(bs):
    req, res = {}, {}
    for b in bs.split(","):
        if not b:
            continue
        for l in open(f"{V4}/q/in/{b}.jsonl" if os.path.exists(f"{V4}/q/in/{b}.jsonl") else f"{V4}/q/hold/{b}.jsonl"):
            q = json.loads(l); req[(q["row"], q["sample"])] = q
        try:
            for l in open(f"{V4}/q/out/{b}.jsonl"):
                x = json.loads(l); res[x["key"]] = x
        except FileNotFoundError:
            pass
    return {rs: res[q["key"]] for rs, q in req.items() if q["key"] in res and not res[q["key"]].get("error")}, req
C, creq = load(a.cells); Rf, _ = load(a.ref) if a.ref else ({}, {})
t8 = [json.loads(l) for l in open(a.tip8)]
need = {q["key"] for q in t8}; tip = {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if l.startswith(b'{"key": "x|') or l.startswith(b'{"key": "p000|'):
            k = l[9:l.index(b'"', 9)].decode()
            if k in need:
                x = json.loads(l)
                if not x.get("error") and x.get("content") is not None:
                    tip[k] = x
T = {}
for q in t8:
    if q["key"] in tip:
        T[(q["row"], q["sample"])] = tip[q["key"]]
m8 = collections.defaultdict(list)
for (r, s), x in T.items():
    m8[r].append(float(bool(x.get("ok"))))
P = []
for (r, s), x in C.items():
    t = T.get((r, s)); f = Rf.get((r, s))
    P.append(dict(row=r, s=s, bench=r.split("/")[0], ok=float(bool(x.get("ok"))), tok=int(x.get("tokens") or 0), cut=x.get("finish") == "length",
                  tok_ok=None if t is None else float(bool(t.get("ok"))), ttok=None if t is None else int(t.get("tokens") or 0),
                  ref_ok=None if f is None else float(bool(f.get("ok"))), rtok=None if f is None else int(f.get("tokens") or 0),
                  rcut=None if f is None else f.get("finish") == "length", m8=sum(m8[r]) / len(m8[r]) if m8[r] else None))
def dstat(ps, key):
    byr = collections.defaultdict(list)
    for p in ps:
        if p[key] is not None:
            byr[p["row"]].append(p["ok"] - p[key])
    if len(byr) < 2:
        return "n/a"
    d = [sum(v) / len(v) for v in byr.values()]; m = sum(d) / len(d); se = statistics.pstdev(d) / math.sqrt(len(d) - 1)
    pv = math.erfc(abs(m / se) / math.sqrt(2)) if se > 0 else 1.0
    return f"{m*100:+.2f}±{se*100:.2f} (p {pv:.3f}, {len(byr)} rows)"
def line(ps, label):
    if not ps:
        return
    pr = [p for p in ps if p["ref_ok"] is not None]
    wr = sum(1 for p in pr if p["ok"] > p["ref_ok"]); rw = sum(1 for p in pr if p["ok"] < p["ref_ok"])
    mult_ref = sum(p["tok"] for p in pr) / max(1, sum(p["rtok"] for p in pr)) if pr else float("nan")
    pt = [p for p in ps if p["ttok"] is not None]
    mult_tip = sum(p["tok"] for p in pt) / max(1, sum(p["ttok"] for p in pt))
    print(f"  {label:20s} reads {len(ps):4d}  acc {sum(p['ok'] for p in ps)/len(ps):.3f}"
          + (f" ref {sum(p['ref_ok'] for p in pr)/len(pr):.3f}" if pr else "")
          + f" | vs W-null {dstat(ps, 'ref_ok')} w->r {wr} r->w {rw} tok x{mult_ref:.2f}"
          + f" | vs tip mean8 {dstat(ps, 'm8')} | vs tip same-index {dstat(ps, 'tok_ok')} tok x{mult_tip:.2f}"
          + f" | cut {sum(p['cut'] for p in ps)} (ref {sum(1 for p in pr if p['rcut'])})")
print(f"{a.name}: cell reads {len(C)}/{len(creq)}; ref reads {len(Rf)}; paired with ref {sum(1 for p in P if p['ref_ok'] is not None)}")
line(P, "all")
line([p for p in P if p["ttok"] is not None and p["ttok"] > a.gate], "gated(tip read)")
line([p for p in P if p["ttok"] is not None and p["ttok"] <= a.gate], "other(tip read)")
if a.ref:
    line([p for p in P if p["rtok"] is not None and p["rtok"] > a.gate], "gated(W-null read)")
    line([p for p in P if p["rtok"] is not None and p["rtok"] <= a.gate], "other(W-null read)")
for bn in sorted({p["bench"] for p in P}):
    line([p for p in P if p["bench"] == bn], f"{bn}")
# length profile vs the control (FL cells, coordinator 06:05): per-read token ratio cell / W-null, reads above 16k tokens, cuts by benchmark,
# accuracy on reads the cell made longer than 2x the control vs the others (cell and control accuracy on the same reads)
pr = [p for p in P if p["ref_ok"] is not None and p["rtok"]]
if pr:
    rat = sorted(p["tok"] / p["rtok"] for p in pr)
    q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]
    print(f"  length profile vs W-null: token ratio per read median x{q(rat, .5):.2f} p90 x{q(rat, .9):.2f}; reads > 16k tokens {sum(p['tok'] > 16000 for p in P)/len(P):.1%} "
          f"(W-null {sum(p['rtok'] > 16000 for p in pr)/len(pr):.1%}); cuts at max_tokens {sum(p['cut'] for p in P)} (lego {sum(p['cut'] for p in P if p['bench'] == 'lego')}, "
          f"other {sum(p['cut'] for p in P if p['bench'] != 'lego')}); W-null cuts {sum(1 for p in pr if p['rcut'])} (lego {sum(1 for p in pr if p['rcut'] and p['bench'] == 'lego')})")
    for lab, sub in (("longer than 2x W-null", [p for p in pr if p["tok"] > 2 * p["rtok"]]), ("within 2x W-null", [p for p in pr if p["tok"] <= 2 * p["rtok"]])):
        if sub:
            print(f"  {lab:22s} reads {len(sub):4d} ({len(sub)/len(pr):.0%})  acc cell {sum(p['ok'] for p in sub)/len(sub):.3f} vs W-null {sum(p['ref_ok'] for p in sub)/len(sub):.3f}"
                  f"  d {dstat(sub, 'ref_ok')}  cut {sum(p['cut'] for p in sub)}")

# split by the tip's router conditions: rows where entry i fires (its procedure prefix / settings / ops are in the row's view) vs not
import os
if a.fired and os.path.exists(a.fired):
    TF = json.load(open(a.fired))
    print("  by tip entry (condition fires on the row):")
    for e in TF["entries"]:
        if e["condition"] in (None, "always"):
            continue
        on = [p for p in P if e["i"] in TF["rows"].get(p["row"], [])]
        if on:
            line(on, f"e{e['i']} on")
    line([p for p in P if not [i for i in TF["rows"].get(p["row"], []) if TF["entries"][i]["condition"] not in (None, "always")]], "no entry fires")
    print("  entries: " + " | ".join(f"e{e['i']}: {(e['condition'] or '')[:70]}" for e in TF["entries"]))
