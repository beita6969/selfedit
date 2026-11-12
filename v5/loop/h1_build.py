"""AM23 hand bundle (coordinator 11-10 23:25): request file for the synthetic bundle candidate H1 = the member entries applied together on
the current tip's content, the way the tip applies several entries (each on the rows its own router decision holds for; entry order =
candidate id). Reads: the union of the members' routed JUDGE rows, samples 4-7 (new on every row: on a row where one member holds alone the
bundle's view equals that member's view, whose samples 0-3 are its own selection data). Read-only on the run: it works on a copy of
state.json and writes only into OUT. Members: the given cids plus (--auto-gen G) every routed (not global), op-free, edit-free proposer
candidate of generation G with d > 0 on its read rows (final numbers, else judge).
usage: h1_build.py RUN_DIR OUT_DIR CID[,CID...] [--auto-gen G] [--name bundle_H1] [--prio 2] [--s0 4] [--m 4] [--tip t005r]
OUT_DIR/requests.jsonl  the read requests (engine_api §3 read lines; the engine agent submits them as one batch)
OUT_DIR/table.json      per union row: H1 keys, the members holding, the parent (tip) sample keys; members with their numbers
OUT_DIR/h1_cand.json    the synthetic candidate record (entry.parts, members) for the "add" annotation if H1 is accepted"""
import collections, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen  # noqa: E402

av = sys.argv[1:]


def opt(k, d):
    return av[av.index(k) + 1] if k in av else d


RUN, OUT, CIDS = av[0], av[1], [x for x in av[2].split(",") if x]
AUTO = int(opt("--auto-gen", 0))
NAME, PRIO, S0, M, TIPID = opt("--name", "bundle_H1"), int(opt("--prio", 2)), int(opt("--s0", 4)), int(opt("--m", 4)), opt("--tip", "t005r")
os.makedirs(OUT, exist_ok=True)
cp = os.path.join(OUT, "runcopy")
os.makedirs(cp, exist_ok=True)
shutil.copy(os.path.join(RUN, "state.json"), os.path.join(cp, "state.json"))
R = gen.Run(gen.args(["--run", cp]))
st = R.state
tips = {t["id"]: t for t in st["tips"] + [st["tip"]]}
assert TIPID in tips, f"{TIPID} not a tip of the run"
base = tips[TIPID]["entries"]
key = lambda es: json.dumps(es, sort_keys=True)
allc = {c["cid"]: c for G in st["gens"].values() for c in G.get("cands", [])}


def num(c):
    j = c.get("final") or c.get("judge") or {}
    return j.get("D", j.get("d")), j.get("p"), j.get("n_rows") or j.get("n")


def se_of(c):
    j = c.get("final") or c.get("judge") or {}
    return j.get("se_used", j.get("se"))


def eligible(c, why_fixed=False):
    e = c.get("entry") or {}
    bad = []
    if c["author"] == "system" or e.get("parts"):
        bad.append("system / stack")
    if e.get("cond") == "always" or R.is_global(c):
        bad.append("global")
    if e.get("ops"):
        bad.append("ops")
    if e.get("edit"):
        bad.append("weight edit")
    if c.get("merged") or c.get("accepted"):
        bad.append("accepted / merged")
    if c.get("withdrawn"):
        bad.append("withdrawn")
    if key((tips.get(c.get("parent")) or {}).get("entries")) != key(base):
        bad.append(f"read against {c.get('parent')}, not {TIPID}'s content")
    return bad


members, notes = [], []
for i in CIDS:
    c = allc.get(i)
    assert c is not None, f"{i} not in state"
    bad = eligible(c)
    assert not bad, f"{i} cannot be bundled: {bad}"
    members.append(c)
if AUTO:
    for c in st["gens"][str(AUTO)].get("cands", []):
        if not c.get("rung") or not c.get("entry") or c["cid"] in CIDS:
            continue
        d, p, n = num(c)
        bad = eligible(c)
        if bad:
            notes.append((c["cid"], c["author"], "out: " + ", ".join(bad), d, p, n))
            continue
        if d is None or d <= 0:
            notes.append((c["cid"], c["author"], "out: d <= 0" if d is not None else "out: no numbers", d, p, n))
            continue
        members.append(c)
        notes.append((c["cid"], c["author"], "in", d, p, n))
members.sort(key=lambda c: c["cid"])
parts = [dict(c["entry"]) for c in members]
H = dict(condition=f"bundle H1 of {len(parts)} routed entries", cond="bundle", procedure=None, settings={}, ops=[], parts=parts, author="system")
R.state["tip"] = dict(tips[TIPID])                           # the reads are against TIPID's content (the copy only)
R.load_routes(base + parts, R.judge_rows)
miss = [(c["cid"], r) for c in members for r in R.judge_rows if (c["entry"]["cond"], r) not in R.routes]
union = [r for r in R.judge_rows if R.fires(H, r)]
holders = {r: [c["cid"] for c in members if R.fires(c["entry"], r)] for r in union}
reqs, table, exist = [], [], 0
for r in union:
    ks = []
    for s in range(S0, S0 + M):
        q, sg = R.read_req(base + [H], r, s, PRIO)
        if R.io.get(q["key"]) is not None or q["key"] in R.io.pending:
            exist += 1
        reqs.append(q)
        ks.append(q["key"])
    _, _, psg = R.view(base, r)
    table.append(dict(row=r, bench=R.rows[r]["bench"], pic=R.rows[r].get("pic") or r, members=holders[r], keys=ks,
                      parent_keys=[R.read_key(psg, r, s) for s in range(gen.P_FULL)],
                      parent_have=sum(1 for s in range(gen.P_FULL) if R.io.get(R.read_key(psg, r, s)) is not None)))
with open(os.path.join(OUT, "requests.jsonl"), "w") as f:
    for q in reqs:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
mem = [dict(cid=c["cid"], author=c["author"], gen=c.get("gen"), routed=sum(1 for r in R.judge_rows if R.fires(c["entry"], r)),
            d=num(c)[0], se=se_of(c), p=num(c)[1], n_rows=num(c)[2], cancelled=bool(c.get("reads_cancelled")),
            condition=c["entry"].get("condition")) for c in members]
json.dump(dict(name=NAME, tip=TIPID, samples=list(range(S0, S0 + M)), priority=PRIO, members=mem, union_rows=len(union),
               rows_2plus=sum(1 for r in union if len(holders[r]) >= 2), reads=len(reqs), rows=table), open(os.path.join(OUT, "table.json"), "w"), indent=1)
json.dump(dict(cid="H1", author="system", kind="bundle", parent=TIPID, entry=H, members=[c["cid"] for c in members],
               routed=len(union), share=len(union) / max(len(R.judge_rows), 1)), open(os.path.join(OUT, "h1_cand.json"), "w"), indent=1)
hist = collections.Counter(len(v) for v in holders.values())
print(f"H1 on {TIPID}: {len(members)} members {[c['cid'] for c in members]}")
for m_ in mem:
    f4 = lambda x: "-" if x is None else f"{x:+.4f}"
    print(f"  {m_['cid']} {m_['author']:3s} routed {m_['routed']:3d}  rows read {m_['n_rows']}  d {f4(m_['d'])}  se {f4(m_['se'])}  p {f4(m_['p'])}"
          f"{'  (reads cancelled)' if m_['cancelled'] else ''} | {(m_['condition'] or '')[:80]}")
if AUTO:
    print(f"generation {AUTO} candidates considered:")
    for x in notes:
        d, p, n = x[3], x[4], x[5]
        print(f"  {x[0]} {x[1]:3s} {x[2]}" + (f"  d {d:+.4f} p {p:.4f} rows {n}" if d is not None and p is not None else ""))
print(f"union {len(union)} JUDGE rows; members holding per row: {dict(sorted(hist.items()))}; rows with >= 2 members: "
      f"{sum(1 for r in union if len(holders[r]) >= 2)}")
print(f"{len(reqs)} reads (samples {S0}-{S0 + M - 1}, priority {PRIO}); already done or pending: {exist}; routes missing: {len(miss)}; "
      f"rows without the full parent cache: {sum(1 for t in table if t['parent_have'] < gen.P_FULL)}")
print(f"wrote {OUT}/requests.jsonl, table.json, h1_cand.json")
