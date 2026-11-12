"""B1 checks (coordinator 11-11, Track B): the signal export.
1. v4io: the location index holds signal rows only; full_result reads the whole line (content longer than KEEP) while the in-memory copy is
   cut; a line with U+2028 in its content is one line; offsets stay right across polls with a partial last line.
2. A dry generation (fake engine, request files written as the real IO does): traces.json of the tip after the parent step and after the
   report: rows = show + monitor exactly, NO judge row, the format of the hand export (fields, samples by str(index), sample fields),
   settings / prefix = the tip's read request; traces_meta.json complete with P_FULL samples per row.
3. No judge row can appear: trace_row raises on a judge row; with a judge row forced into the row set the export fails (logged, the loop
   goes on) and the files on disk keep no judge row; every traces_cand line is on a show / monitor row and never a tip key.
4. traces_cand.jsonl: the candidates' show reads and the lab reads against the tip, tagged (batch, gen, cid for per-candidate batches,
   lab turn), each key once; a restart (new Run on the same run dir) appends nothing again.
5. Tip change: write_tip writes the new tip's export at once; rows whose route for a new conditional entry is unknown are left out and the
   export says not complete; once routed, the next tick completes it.
usage: python3 t_signal.py"""
import collections, json, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
os.chdir(HERE)
import gen  # noqa: E402
import signal_export as SX  # noqa: E402
import t_versions as T  # noqa: E402
import v4io  # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("ok   " if cond else "FAIL ") + name, flush=True)
    ok, fail = ok + bool(cond), fail + (not cond)


gen.time.sleep = lambda s: None
LOG = []
gen.log = lambda m: LOG.append(m)

# ---------------------------------------------------------------------------------------------------------------- 1. v4io location index
d = tempfile.mkdtemp(prefix="t_signal_io_")
io = v4io.IO(d, log=lambda *a: None, loc_rows={"rs"})
long_c, long_r = "c" * 5000 + " tail", "r" * 4000
f = os.path.join(d, "q", "out", "b1.jsonl")
with open(f, "w") as fh:
    fh.write(json.dumps(dict(key="x|a|rs|0", mode="read", row="rs", sample=0, content=long_c, reasoning=long_r, ok=True), ensure_ascii=False) + "\n")
    fh.write(json.dumps(dict(key="x|a|rj|0", mode="read", row="rj", sample=0, content="j", ok=True)) + "\n")
    fh.write(json.dumps(dict(key="x|a|rs|1", mode="read", row="rs", sample=1, content="partial"))[:20])
io.poll()
check("v4io: the U+2028 line is one result; the partial last line waits", "x|a|rs|0" in io.res and "x|a|rs|1" not in io.res and "x|a|rj|0" in io.res)
check("v4io: location index holds the signal row only; in memory the content is cut to KEEP", set(io.loc) == {"x|a|rs|0"}
      and len(io.res["x|a|rs|0"]["content"]) == v4io.KEEP)
with open(f, "a") as fh:
    fh.write(json.dumps(dict(key="x|a|rs|1", mode="read", row="rs", sample=1, content="partial"))[20:] + "\n")
    fh.write(json.dumps(dict(key="x|a|rs|2", mode="read", row="rs", sample=2, content="z" * 3000, ok=False)) + "\n")
io.poll()


class _S(SX.SignalMixin):
    pass


s_ = _S()
s_.io = io
check("v4io: after the partial line completes, offsets are right (whole lines read back for every indexed key)",
      s_.full_result("x|a|rs|0")["content"] == long_c and s_.full_result("x|a|rs|0")["reasoning"] == long_r
      and s_.full_result("x|a|rs|1")["content"] == "partial" and len(s_.full_result("x|a|rs|2")["content"]) == 3000
      and s_.full_result("x|a|rj|0") is None)


# ---------------------------------------------------------------------------------------------------------------- 1b. finish "abort" (11-11 01:45)
d2 = tempfile.mkdtemp(prefix="t_signal_abort_")
io2 = v4io.IO(d2, log=lambda *a: None, loc_rows={"rs"})
io2.submit("g001_parent", [dict(key="x|a|rs|0", mode="read", row="rs", sample=0, prefix=None, settings={}),
                            dict(key="x|a|rs|1", mode="read", row="rs", sample=1, prefix=None, settings={})])
o2 = os.path.join(d2, "q", "out", "b.jsonl")


def put(*rs):
    with open(o2, "a") as fh:
        for r in rs:
            fh.write(json.dumps(r) + "\n")


put(dict(key="x|a|rs|0", mode="read", row="rs", sample=0, content="A", ok=True, finish="stop"),
    dict(key="x|a|rs|1", mode="read", row="rs", sample=1, content="", ok=False, finish="abort"))
io2.poll()
rr = sorted(f for f in os.listdir(os.path.join(d2, "q", "in")) if "_rr_" in f)
check(f"abort: not a result (not in res / err), counted, pending again and re-sent as g001_parent_rr ({rr})",
      "x|a|rs|1" not in io2.res and "x|a|rs|1" not in io2.err and io2.aborted["x|a|rs|1"] == 1 and len(rr) == 1 and rr[0].startswith("g001_parent_rr_")
      and [json.loads(l)["key"] for l in open(os.path.join(d2, "q", "in", rr[0]))] == ["x|a|rs|1"] and "x|a|rs|0" in io2.res)
put(dict(key="x|a|rs|1", mode="read", row="rs", sample=1, finish="abort"))
io2.poll()
put(dict(key="x|a|rs|1", mode="read", row="rs", sample=1, finish="abort"))
io2.poll()
rr = sorted(f for f in os.listdir(os.path.join(d2, "q", "in")) if "_rr_" in f)
check(f"abort: re-sent at most {v4io.MAX_RR} times by the process that submitted it ({len(rr)} re-sends after 3 aborts)", len(rr) == v4io.MAX_RR)
put(dict(key="x|a|rs|1", mode="read", row="rs", sample=1, content="B", ok=True, finish="stop"),
    dict(key="x|a|rs|0", mode="read", row="rs", sample=0, finish="abort"))
io2.poll()
check("abort: the good line wins; an abort line after a good one changes nothing", io2.res["x|a|rs|1"]["content"] == "B" and io2.res["x|a|rs|0"]["content"] == "A")
put(dict(key="x|a|rs|9", mode="read", row="rs", sample=9, finish="abort"))
with open(os.path.join(d2, "q", "in", "g000_old_1.jsonl"), "w") as fh:
    fh.write(json.dumps(dict(key="x|a|rs|9", mode="read", row="rs", sample=9)) + "\n")
io3 = v4io.IO(d2, log=lambda *a: None)
check("abort: a new process (generation start) sees the aborted key as not pending, so the step that needs it re-sends it",
      "x|a|rs|9" not in io3.pending and "x|a|rs|9" not in io3.res and io3.submit("g002_parent", [dict(key="x|a|rs|9", mode="read", row="rs", sample=9)]) == 1)

# ---------------------------------------------------------------------------------------------------------------- 2. dry generation
class FileIO(T.InstantIO):
    """InstantIO that also writes each batch's new requests to <root>/q/in/<name>_<n>.jsonl, as the real IO does."""

    def __init__(self, root, chat_fn):
        super().__init__(root, chat_fn)
        self.root = root
        os.makedirs(os.path.join(root, "q", "in"), exist_ok=True)

    def submit(self, name, reqs):
        new = [q for q in dict((q["key"], q) for q in reqs).values() if q["key"] not in self.res]
        n = super().submit(name, reqs)
        if new:
            p = os.path.join(self.root, "q", "in", f"{name}_{len(self.batches)}.jsonl")
            with open(p, "w") as fh:
                for q in new:
                    fh.write(json.dumps(q, ensure_ascii=False) + "\n")
        return n

    def cancel_prefix(self, name):
        pass


root = tempfile.mkdtemp(prefix="t_signal_dry_")
R = T.make(root, cls=gen.Run, slots=6, extra=("--no-versions", "--no-am16", "--no-w", "--no-ecomp"))
R.io = FileIO(root, T.chat_default)
T.run_to(R, "propose")
tid = R.state["tip"]["id"]
sd = os.path.join(R.rd, "signal", tid)
tr = json.load(open(os.path.join(sd, "traces.json")))
meta = json.load(open(os.path.join(sd, "traces_meta.json")))
show, mon, judge = set(R.role["show"]), set(R.role["monitor"]), set(R.judge_rows)
check(f"after the parent step: traces.json of {tid} with every show + monitor row ({len(tr)} = {len(show)} + {len(mon)}), no judge row",
      set(tr) == show | mon and not (set(tr) & judge) and judge)
r0 = sorted(show)[0]
x = tr[r0]
check("format of the hand export: bench, role, kind, answer, prompt, image_paths, prefix, settings, samples{str(i): fields}",
      {"bench", "role", "kind", "answer", "prompt", "image_paths", "prefix", "settings", "samples"} <= set(x)
      and set(x["samples"]) == {str(s) for s in range(gen.P_FULL)} and all(set(v) == set(SX.SAMPLE_FIELDS) for v in x["samples"].values()))
q, sg = R.read_req(R.state["tip"]["entries"], r0, 0, 5)
check("prefix / settings = the tip's read request; samples = the cached results of the tip's keys",
      x["prefix"] == q.get("prefix") and x["settings"] == q["settings"]
      and all(x["samples"][str(s)]["content"] == R.io.get(R.read_key(sg, r0, s)).get("content") for s in range(gen.P_FULL)))
check(f"traces_meta: complete, roles {meta['roles']}, {meta['samples']} samples, P_FULL per row", meta["complete"] and meta["rows"] == len(tr)
      and meta["per_row"] == dict(min=gen.P_FULL, max=gen.P_FULL) and meta["roles"] == dict(show=len(show), monitor=len(mon)))
check("state records the export (samples, complete)", R.state["signal"][tid]["complete"] and R.state["signal"][tid]["samples"] == meta["samples"])
check("generation records the tip it started on", R.state["gens"]["1"]["tip_at_start"] == tid)
T.run_to(R, "l1")

# ---------------------------------------------------------------------------------------------------------------- 3. no judge row
try:
    R.trace_row(sorted(judge)[0], None, {})
    raised = False
except ValueError:
    raised = True
check("trace_row raises on a judge row", raised)
orig = R.signal_rows
R.signal_rows = lambda: orig() + [sorted(judge)[0]]
n_before = len(LOG)
R.signal_tick("forced judge row", force=True)
R.signal_rows = orig
check("a judge row forced into the row set: the export fails, logged, the loop goes on",
      any("signal export (forced judge row) failed: ValueError" in m for m in LOG[n_before:]))
cur = R.state["tip"]["id"]
for t_ in os.listdir(os.path.join(R.rd, "signal")):
    tr_ = json.load(open(os.path.join(R.rd, "signal", t_, "traces.json")))
    cl = [json.loads(l) for l in open(os.path.join(R.rd, "signal", t_, "traces_cand.jsonl"))] \
        if os.path.exists(os.path.join(R.rd, "signal", t_, "traces_cand.jsonl")) else []
    check(f"files of {t_}: no judge row in traces.json ({len(tr_)} rows) nor in traces_cand.jsonl ({len(cl)} lines)",
          not (set(tr_) & judge) and not any(c["row"] in judge for c in cl) and all(c["role"] in SX.SIGNAL_ROLES for c in cl))

# ---------------------------------------------------------------------------------------------------------------- 4. traces_cand
cl = [json.loads(l) for l in open(os.path.join(sd, "traces_cand.jsonl"))]
tipk = R._tip_keys(R._tip_of(tid))
want = set()
for fn in os.listdir(os.path.join(root, "q", "in")):
    nm = fn[:-6].rsplit("_", 1)[0]
    if nm.startswith("g001_") and any(t in nm for t in ("_lab", "_show", "_rung", "_carry", "_lcomp")):
        for l in open(os.path.join(root, "q", "in", fn)):
            qq = json.loads(l)
            if qq.get("mode") == "read" and qq["row"] in show | mon and qq["key"] not in tipk and R.io.get(qq["key"]) is not None:
                want.add(qq["key"])
got = [c["key"] for c in cl]
bt = collections.Counter(c["batch"].split("_", 1)[1] for c in cl)
check(f"traces_cand: every candidate / lab read on a show row against {tid}, each once ({len(got)} lines; batches {dict(bt)})",
      set(got) == want and len(got) == len(set(got)) and want)
check("traces_cand: no tip key, every line tagged (batch, gen, tip), lab reads carry their turn, per-candidate batches their cid",
      not (set(got) & tipk) and all(c["gen"] == 1 and c["tip"] == tid and c["batch"] for c in cl)
      and all(c["lab_turn"] is not None for c in cl if "_labread" in c["batch"])
      and all(c["cid"] for c in cl if "_rung1_" in c["batch"]))
shw = [c for c in cl if c["batch"].endswith("_show")]
okc = all(c["cid"] and R.read_req(R._tip_of(tid)["entries"] + [next(x for x in R.state["gens"]["1"]["cands"] if x["cid"] == c["cid"])["entry"]],
                                  c["row"], c["sample"], 1)[0]["key"] == c["key"] for c in shw if len(c["cids"]) == 1)
check(f"traces_cand: SHOW-batch reads resolved to their candidate by view ({sum(1 for c in shw if c['cid'])} of {len(shw)}; "
      f"{sum(1 for c in shw if len(c['cids']) > 1)} shared by several)", okc and sum(1 for c in shw if c["cids"]) == len(shw) and shw)
rs_ = sorted(show)[3]
lq = R.read_req(R._tip_of(tid)["entries"] + [dict(condition="always", cond="always", procedure="Lab draft.", settings={}, ops=[])], rs_, 1, 1)[0]
lq2 = dict(lq, key=lq["key"].replace(f"|{rs_}|", f"|{sorted(judge)[0]}|"), row=sorted(judge)[0])
with open(os.path.join(root, "q", "in", "g001_labread2_99999.jsonl"), "w") as fh:
    fh.write(json.dumps(lq) + "\n" + json.dumps(lq2) + "\n")
R.io.res[lq["key"]] = dict(key=lq["key"], mode="read", row=rs_, sample=1, content="lab says A", reasoning="r", tokens=5, ok=True, pred="A", finish="stop")
R.io.res[lq2["key"]] = dict(key=lq2["key"], mode="read", row=lq2["row"], sample=1, content="judge row", ok=True)
R.export_traces_cand("lab", tid)
cl2 = [json.loads(l) for l in open(os.path.join(sd, "traces_cand.jsonl"))][len(cl):]
check(f"a lab read on a show row: appended with lab_turn 2, its own prefix, the one sample; its twin on a JUDGE row is not ({len(cl2)} new)",
      len(cl2) == 1 and cl2[0]["lab_turn"] == 2 and cl2[0]["prefix"] == lq["prefix"] and "Lab draft." in cl2[0]["prefix"]
      and cl2[0]["samples"] == {"1": dict(content="lab says A", reasoning="r", tokens=5, prompt_tokens=None, image_tokens=None, ok=True, pred="A",
                                          finish="stop")} and cl2[0]["row"] == rs_)
cl = cl + cl2
c0 = next(c for c in cl if "_labread" in c["batch"] or "_show" in c["batch"])
qq = next(json.loads(l) for fn in os.listdir(os.path.join(root, "q", "in")) for l in open(os.path.join(root, "q", "in", fn))
          if json.loads(l)["key"] == c0["key"])
check("traces_cand: prefix / settings / ops are the request's own, the one sample is the result",
      c0["prefix"] == qq.get("prefix") and c0["settings"] == qq.get("settings") and c0.get("ops") == (qq.get("ops") or None) or not qq.get("ops")
      and list(c0["samples"]) == [str(qq["sample"])])
R2 = T.make(root, cls=gen.Run, slots=6, extra=("--no-versions", "--no-am16", "--no-w", "--no-ecomp"))
R2.io = R.io
R2.g = 1
R2.load_routes(R2._tip_of(tid)["entries"], R2.signal_rows())
n2 = R2.export_traces_cand("restart", tid)
check("restart: the keys file stops a second append", n2 == 0 and len(open(os.path.join(sd, "traces_cand.jsonl")).readlines()) == len(cl))

# ---------------------------------------------------------------------------------------------------------------- 5. tip change
t = R.state["tip"]
R.state["tips"].append(dict(t))
e_new = dict(condition="Does the question show a test condition?", cond="Does the question show a test condition?", procedure="Look twice.",
             settings={}, ops=[], cid="zz1", author="P0")
R.state["tip"] = dict(t, id="t9test", entries=t["entries"] + [e_new], parent=t["id"], born=2)
mrow = sorted(mon)[0]
for r in R.signal_rows():
    if r != mrow:
        R.routes[(e_new["cond"], r)] = (r == sorted(show)[1])
R.routes.pop((e_new["cond"], mrow), None)
R.write_tip()
m9 = json.load(open(os.path.join(R.rd, "signal", "t9test", "traces_meta.json")))
t9 = json.load(open(os.path.join(R.rd, "signal", "t9test", "traces.json")))
check(f"tip change: write_tip writes the new tip's export at once; the unrouted monitor row is left out, not complete ({m9['unrouted']}, "
      f"{m9['complete']})", m9["tip"] == "t9test" and m9["unrouted"] == 1 and not m9["complete"] and mrow not in t9 and len(t9) == len(show | mon) - 1)
r1 = sorted(show)[1]
t1x = json.load(open(os.path.join(R.rd, "signal", t["id"], "traces.json")))
same = [r for r in t9 if r != r1]
check(f"on the row where the new entry holds, the new view (prefix with its procedure) and no samples yet; on the other {len(same)} rows the "
      f"view and samples of {t['id']} carry over", "Look twice." in (t9[r1]["prefix"] or "") and t9[r1]["samples"] == {}
      and all(t9[r]["prefix"] == t1x[r]["prefix"] and t9[r]["samples"] == t1x[r]["samples"] for r in same))
R.routes[(e_new["cond"], mrow)] = False
R.signal_tick("routed", force=True)
m9b = json.load(open(os.path.join(R.rd, "signal", "t9test", "traces_meta.json")))
check("once routed, the next tick writes the row (still not complete: the new view has no samples on its row)",
      m9b["unrouted"] == 0 and m9b["rows"] == len(show | mon) and not m9b["complete"])
check("--no-signal: no tick", (lambda: (setattr(R.a, "no_signal", True), R.signal_tick("off", force=True), setattr(R.a, "no_signal", False)))() is not None
      and not any("signal export (off)" in m for m in LOG))

# ---------------------------------------------------------------------------------------------------------------- 6. a new process before its parent step
# live 11-11 02:13: gen 8's first tick (a new gen.py process, no routes loaded yet, nothing recorded in state for t007c) wrote 0 rows over the
# hand export. The export now loads the tip's routes itself, and never replaces a tip's export by one with fewer samples.
t0x = R.state["tip"]                                      # t9test: one conditional entry; its routes as results in the IO, as live
for r in R.signal_rows():
    R.io.res[R.route_key(e_new["cond"], r)] = dict(key=R.route_key(e_new["cond"], r), mode="route", p=0.9 if R.routes.get((e_new["cond"], r)) else 0.1)
t = dict(t, id=t0x["id"])
f0 = os.path.join(R.rd, "signal", t["id"], "traces.json")
m0 = json.load(open(os.path.join(R.rd, "signal", t["id"], "traces_meta.json")))
R3 = T.make(root, cls=gen.Run, slots=6, extra=("--no-versions", "--no-am16", "--no-w", "--no-ecomp"))
R3.io, R3.g = R.io, 2
R3.state["tip"] = dict(t0x)
R3.state["signal"] = {}
R3.routes.clear()
nu0 = sum(not R3.route_known(t0x["entries"], r) for r in R3.signal_rows())
R3.export_traces("tick")
m3 = json.load(open(os.path.join(R.rd, "signal", t["id"], "traces_meta.json")))
check(f"new process, no routes loaded, nothing in state: the export loads the tip's routes, all rows written ({m3['rows']}, {m3['samples']} samples, "
      f"unrouted {m3['unrouted']}; {nu0} rows unrouted before the load)", nu0 > 0 and m3["rows"] == len(show | mon) and m3["unrouted"] == 0 and m3["samples"] == m0["samples"])
b0 = open(f0, "rb").read()
o3 = R3.signal_rows
R3.signal_rows = lambda: o3()[:3]
R3.state["signal"] = {}
R3.export_traces("tick")
R3.signal_rows = o3
check("an export with fewer samples than the file already written for the tip: file kept, logged once",
      open(f0, "rb").read() == b0 and sum("already written for" in m and "file kept" in m for m in LOG) == 1)

print(f"\n{ok} ok, {fail} failed")
sys.exit(1 if fail else 0)
