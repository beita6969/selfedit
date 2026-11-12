"""Tests of fake_dispatch.py. Run: python3 t_fake_dispatch.py -> ALL GREEN. Works in a fresh temp root (left in place, nothing deleted)."""
import collections
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fake_dispatch as F  # noqa: E402
import proposer as P  # noqa: E402

ROWS = os.path.join(HERE, "..", "data", "learn.jsonl")
KNOBS = json.load(open(os.path.join(HERE, "knobs.json")))
FAKE = [sys.executable, os.path.join(HERE, "fake_dispatch.py")]
PLAIN_DIGEST = "566efee1e4e7100df339fe901bbc97c6772a0c23adc28a8b4c4d2408278ac4a1"   # stock reads of the pre-ops fake (11-10)


def check(name, cond):
    if not cond:
        raise AssertionError(name)
    print("ok  ", name)


def submit(root, batch, reqs):
    d = os.path.join(root, "q", "in")
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, batch + ".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in reqs:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, os.path.join(d, batch + ".jsonl"))


def run(root, *extra):
    subprocess.run(FAKE + ["--root", root, "--rows", ROWS, "--rate", "0", "--quiet", "--once", *extra], check=True)


def out(root, batch):
    f = os.path.join(root, "q", "out", batch + ".jsonl")
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


def done(root, batch):
    f = os.path.join(root, "q", "out", batch + ".done")
    return json.load(open(f)) if os.path.exists(f) else None


def main():
    rows = P.load_rows(ROWS)
    ids = sorted(rows)
    root = tempfile.mkdtemp(prefix="fake_v4_", dir=os.environ.get("TMPDIR"))
    print("     root:", root)

    # ---------------------------------------------------------------- a mixed batch
    reqs = []
    for i, rid in enumerate(ids[:60]):
        for s in range(2):
            reqs.append(dict(key=f"g001|parent|{rid}|{s}", mode="read", engine="E", priority=2, row=rid, sample=s, prefix=None, settings={}))
        reqs.append(dict(key=f"g001|c001|judge|{rid}|0", mode="read", engine="E", priority=1, row=rid, sample=0,
                         prefix="Read every label first.", settings={"max_image_tokens": 8192, "min_image_tokens": None, "temperature": None}))
        reqs.append(dict(key=f"g001|route|c001|{rid}", mode="route", engine="E", priority=0, row=rid, condition="The image contains a chart."))
    reqs += [dict(key=f"g001|chat|P0|{k}", mode="chat", engine="P", priority=0, messages=[{"role": "user", "content": "x"}]) for k in range(20)]
    reqs.append(dict(key="g001|bad|row", mode="read", engine="E", priority=1, row="nobench/000", sample=0))
    submit(root, "g001_mixed", reqs)
    with open(os.path.join(root, "q", "in", "g001_partial.jsonl.tmp"), "w") as f:        # a .tmp file is never read
        f.write(json.dumps(dict(key="tmp|1", mode="read", row=ids[0], sample=0)) + "\n")
    run(root)
    res = out(root, "g001_mixed")
    by = {r["key"]: r for r in res}
    d = done(root, "g001_mixed")
    check("one result line per request, .done counts match",
          len(res) == len(reqs) == len(by) and d["total"] == len(reqs) and d["finished"] == len(reqs) and d["cancelled"] == 0 and d["errors"] == 1)
    check(".tmp batch never read", not out(root, "g001_partial") and "tmp|1" not in by)
    prio = {r["key"]: r["priority"] for r in reqs}
    order = [prio[r["key"]] for r in res]
    check("priority order: lower number first (line order)", order == sorted(order))
    rd = [r for r in res if r["mode"] == "read" and "error" not in r]
    need = {"key", "mode", "row", "sample", "content", "reasoning", "finish", "tokens", "prompt_tokens", "image_tokens", "ok", "loop",
            "grade_error", "server", "t_submit", "t_done", "seed", "settings", "prefix", "bench", "id", "kind", "answer", "options", "category", "pred"}
    check("read: §4 fields + seed, settings, prefix, bench, pred", all(need <= set(r) for r in rd))
    check("read: pred = the boxed answer on every read, None on 'length'; no ops / op_trace without ops (as the engine)",
          all(r["pred"] == (r["content"][len("Answer: \\boxed{"):-1] if r["finish"] == "stop" else None) for r in rd)
          and not any("ops" in r or "op_trace" in r for r in rd))
    check("read: seed = crc32('101|0|<row>') % (2**31-1024) + sample", all(r["seed"] == F.crn_seed(r["row"], r["sample"]) for r in rd))
    check("read: tokens 500-6000 on stop, length -> ok False and empty content",
          all((500 <= r["tokens"] <= 6000) if r["finish"] == "stop" else (not r["ok"] and r["content"] == "") for r in rd))
    check("read: content 'Answer: \\boxed{...}', gold when ok",
          all(r["content"] == f"Answer: \\boxed{{{r['answer']}}}" for r in rd if r["ok"])
          and all(r["content"].startswith("Answer: \\boxed{") for r in rd if r["finish"] == "stop"))
    check("read: unknown row -> error line (unread), no ok field", "error" in by["g001|bad|row"] and "ok" not in by["g001|bad|row"])
    rt = [r for r in res if r["mode"] == "route"]
    check("route: fields, deterministic yes, cache written",
          all({"key", "mode", "row", "cond", "yes", "content", "cached"} <= set(r) and r["yes"] == F.route_yes(r["row"], "The image contains a chart.")
              and r["cached"] is False for r in rt)
          and len(open(os.path.join(root, "route_cache.jsonl")).readlines()) == 60)
    ch = [r for r in res if r["mode"] == "chat"]
    check("chat: content, reasoning, finish, tokens", all({"content", "reasoning", "finish", "tokens"} <= set(r) for r in ch))

    # ---------------------------------------------------------------- deterministic outcome model
    stock = {}
    check("effect: 0 for null prefix + stock settings (explicit stock values too)",
          F.effect(None, {}, KNOBS) == 0 and F.effect("", {"temperature": 1.0, "max_image_tokens": 16384, "min_image_tokens": None}, KNOBS) == 0)
    effs = [F.effect(f"proc {k}", {"temperature": 0.8} if k % 2 else {}, KNOBS) for k in range(2000)]
    check("effect: in [-0.06, +0.08], spread over the range", min(effs) >= -0.06 and max(effs) <= 0.08 and min(effs) < -0.05 and max(effs) > 0.07)
    check("--effect-scale 0: every effect is 0", all(F.effect(f"proc {k}", {}, KNOBS, scale=0.0) == 0 for k in range(100)))
    bases = [F.base_p(r) for r in ids]
    check("base: per row in [0.2, 0.9]", min(bases) >= 0.2 and max(bases) <= 0.9 and min(bases) < 0.22 and max(bases) > 0.88)
    qs = [F.route_q(f"cond {k}") for k in range(500)]
    check("route q(condition) in [0.1, 0.6]", min(qs) >= 0.1 and max(qs) <= 0.6)
    cnd = "The question asks to count objects in the image."
    yr = sum(F.route_yes(r, cnd) for r in ids) / len(ids)
    check(f"route: yes share over 768 rows ({yr:.3f}) ~ q ({F.route_q(cnd):.3f})", abs(yr - F.route_q(cnd)) < 0.06)
    # stock accuracy over many samples ~ mean base; a fresh root gives identical outcomes
    root2 = tempfile.mkdtemp(prefix="fake_v4_", dir=os.environ.get("TMPDIR"))
    big = [dict(key=f"p000|parent|{rid}|{s}", mode="read", row=rid, sample=s, prefix=None, settings={}) for rid in ids for s in range(4)]
    submit(root2, "p000_parent_0", big)
    run(root2)
    o2 = out(root2, "p000_parent_0")
    acc = sum(r["ok"] for r in o2) / len(o2)
    mb = sum(F.base_p(r) for r in ids) / len(ids) * (1 - F.P_LENGTH)
    check(f"stock accuracy {acc:.3f} ~ mean base x (1 - 1 % length) {mb:.3f}", abs(acc - mb) < 0.02)
    lr = sum(r["finish"] == "length" for r in o2) / len(o2)
    check(f"length share {lr:.4f} ~ 0.01", 0.003 < lr < 0.02)
    root3 = tempfile.mkdtemp(prefix="fake_v4_", dir=os.environ.get("TMPDIR"))
    submit(root3, "p000_parent_0", big[:400])
    run(root3)
    strip = lambda r: {k: v for k, v in r.items() if k not in ("t_submit", "t_done")}
    check("outcomes are deterministic (same requests, fresh root, same results)",
          [strip(r) for r in out(root3, "p000_parent_0")] == [strip(r) for r in o2[:400]])
    dig = hashlib.sha256(json.dumps([[r["key"], r["ok"], r["finish"], r["tokens"], r["content"], r["seed"], r["prompt_tokens"]] for r in o2]).encode()).hexdigest()
    check("stock reads are those of the fake before harness ops (frozen digest of 3072 reads)", dig == PLAIN_DIGEST)
    # a positive-effect prefix raises accuracy, a null world does not
    pre = next(f"procedure {k}" for k in range(1000) if F.effect(f"procedure {k}", {}, KNOBS) > 0.07)
    cand = [dict(key=f"c|{rid}|{s}", mode="read", row=rid, sample=s, prefix=pre, settings={}) for rid in ids for s in range(4)]
    for scale, name in (("1", "eff"), ("0", "null")):
        rr = tempfile.mkdtemp(prefix="fake_v4_", dir=os.environ.get("TMPDIR"))
        submit(rr, "cand", cand)
        run(rr, "--effect-scale", scale)
        a = sum(r["ok"] for r in out(rr, "cand")) / len(cand)
        if name == "eff":
            check(f"effect {F.effect(pre, {}, KNOBS):+.3f}: accuracy {a:.3f} vs stock {acc:.3f}", 0.04 < a - acc < 0.11)
        else:
            check(f"--effect-scale 0: accuracy {a:.3f} vs stock {acc:.3f} (draws differ, p equal)", abs(a - acc) < 0.03)

    # ---------------------------------------------------------------- harness ops on read (engine_api.md §7)
    nimg = {rid: len(rows[rid].get("image_paths") or []) for rid in ids}
    small = [rid for rid in ids if nimg[rid] <= 4]
    VAR = {"zoom": [{"op": "zoom"}], "tiles2": [{"op": "tiles", "n": 2}], "tiles3": [{"op": "tiles", "n": 3}], "check": [{"op": "check"}],
           "all": [{"op": "check"}, {"op": "tiles", "n": 2}, {"op": "zoom"}]}            # given out of order on purpose
    ro = tempfile.mkdtemp(prefix="fake_v4_ops_", dir=os.environ.get("TMPDIR"))
    oreq = [dict(key=f"o|{v}|{rid}|{s}", mode="read", row=rid, sample=s, prefix=None, settings={}, ops=ops)
            for v, ops in VAR.items() for rid in small for s in range(4)]
    oreq += [dict(key=f"o|plain|{rid}|{s}", mode="read", row=rid, sample=s, prefix=None, settings={}) for rid in small for s in range(4)]
    big_rows = [rid for rid in ids if nimg[rid] >= 5]
    four = [rid for rid in ids if nimg[rid] == 4]
    eight = [rid for rid in ids if nimg[rid] == 8]
    oreq += [dict(key=f"o|skip3|{rid}|0", mode="read", row=rid, sample=0, ops=[{"op": "tiles", "n": 3}]) for rid in big_rows]
    oreq += [dict(key=f"o|t3four|{rid}|0", mode="read", row=rid, sample=0, ops=[{"op": "tiles", "n": 3}]) for rid in four]
    oreq += [dict(key=f"o|zt3four|{rid}|0", mode="read", row=rid, sample=0, ops=[{"op": "zoom"}, {"op": "tiles", "n": 3}]) for rid in four]
    oreq += [dict(key=f"o|zt2eight|{rid}|0", mode="read", row=rid, sample=0, ops=[{"op": "zoom"}, {"op": "tiles", "n": 2}]) for rid in eight]
    oreq.append(dict(key="o|bad", mode="read", row=ids[0], sample=0, ops=[{"op": "blur"}]))
    submit(ro, "ops", oreq)
    run(ro)
    ob = {r["key"]: r for r in out(ro, "ops")}
    do = done(ro, "ops")
    check(f"ops batch: one line per request ({len(oreq)}), the unknown op is an error line, op_skipped lines count as finished",
          len(ob) == len(oreq) and do["finished"] == len(oreq) and do["errors"] == 1 and "error" in ob["o|bad"] and "ok" not in ob["o|bad"])
    rs = lambda v: [ob[f"o|{v}|{rid}|{s}"] for rid in small for s in range(4)]
    check("ops: echoed as given; op_trace passes in engine order (zoom, read, check); tokens = the sum over passes",
          all(r["ops"] == VAR[v] and [t["op"] for t in r["op_trace"]] == [o for o in ("zoom", "read", "check") if o == "read" or o in v or v == "all"]
              and r["tokens"] == sum(t["tokens"] for t in r["op_trace"]) for v in VAR for r in rs(v)))
    exact = True
    for v, ops in VAR.items():
        for r in rs(v):
            ek = F.effect_key(None, {}, KNOBS)
            _, length, _, base = F.draw(r["seed"], ek, ops, F.READ_MAX)
            f = (1.15 if v in ("zoom", "all") else 1.0) * (1.4 if v in ("tiles2", "all") else 1.6 if v == "tiles3" else 1.0)
            R = F.READ_MAX if length else min(F.READ_MAX, round(base * f))
            want = R + (48 if v in ("zoom", "all") else 0) + (round(0.8 * R) if v in ("check", "all") else 0)
            exact &= r["tokens"] == want
    check("ops tokens: read pass x 1.15 (zoom) x (1 + 0.2 n) (tiles), + 48 for the zoom pass, + 0.8 x read for check (exact on every read)", exact)
    mean = lambda v: sum(r["tokens"] for r in rs(v) if r["finish"] == "stop") / sum(r["finish"] == "stop" for r in rs(v))
    m0 = mean("plain")
    ratios = {v: round(mean(v) / m0, 3) for v in VAR}
    check(f"ops tokens vs plain reads (mean ratio): {ratios} ~ zoom 1.15 + 48 / mean, tiles 1.4 / 1.6, check 1.8",
          abs(ratios["zoom"] - (1.15 + 48 / m0)) < 0.04 and abs(ratios["tiles2"] - 1.4) < 0.05 and abs(ratios["tiles3"] - 1.6) < 0.05
          and abs(ratios["check"] - 1.8) < 0.06)
    acc = lambda v: sum(r["ok"] for r in rs(v)) / len(rs(v))
    gains = {v: round(acc(v) - acc("plain"), 3) for v in VAR}
    check(f"ops effect on accuracy at --op-effect 1: {gains} (means zoom .02, tiles .01 / .015, check .03; all = .06)",
          -0.02 < gains["tiles2"] < 0.045 and 0.0 < gains["check"] < 0.065 and 0.025 < gains["all"] < 0.1)
    zt = [t for r in rs("zoom") for t in r["op_trace"] if t["op"] == "zoom"]
    check(f"zoom pass: 48 tokens, a box [1, x0, y0, x1, y1] or zoom_fail ({sum(t['zoom_fail'] for t in zt)} of {len(zt)} failed)",
          all(t["tokens"] == 48 and (t["box"] is None) == t["zoom_fail"] and (t["zoom_fail"] or (len(t["box"]) == 5 and 0 <= t["box"][1] < t["box"][3] <= 1)) for t in zt)
          and 0 < sum(t["zoom_fail"] for t in zt) < 0.12 * len(zt))
    ck = [(r, r["op_trace"][-1]) for r in rs("check")]
    up = [(r, t) for r, t in ck if F.op_effect({"op": "check"}, r["row"]) >= 0 and r["finish"] == "stop"]
    check(f"check pass: proposed / ok_before / pred_before; where check helps ({len(up)} reads) a right answer stays right",
          all(t["op"] == "check" and t["pred_before"] == (t["proposed"] if r["finish"] == "stop" else None) for r, t in ck)
          and all(r["ok"] for r, t in up if t["ok_before"]) and any(r["ok"] and not t["ok_before"] for r, t in up))
    t3 = [ob[f"o|t3four|{rid}|0"] for rid in four]
    check(f"tiles: read pass reports n x n x images tiles and their prompt tokens; 4 images x 9 tiles + 4 = 40 is not skipped ({len(t3)} rows)",
          all(r["op_trace"][0]["tiles"] == 36 and "op_skipped" not in r and r["prompt_tokens"] == r["op_trace"][0]["prompt_tokens"]
              == ob[f"o|plain|{r['row']}|0"]["prompt_tokens"] + sum(9 * max(256, t) for t in r["image_tokens"]) for r in t3))
    sk = [ob[f"o|skip3|{rid}|0"] for rid in big_rows] + [ob[f"o|zt3four|{rid}|0"] for rid in four] + [ob[f"o|zt2eight|{rid}|0"] for rid in eight]
    check(f"op_skipped: > 40 images after zoom / tiles ({len(sk)} rows: 5+ images with tiles 3, 4 with zoom + tiles 3, 8 with zoom + tiles 2): "
          "no content, no ok, ops echoed, not an error",
          sk and all(r["op_skipped"].startswith("image_limit: ") and "content" not in r and "ok" not in r and "error" not in r and r["ops"] for r in sk))
    rz = tempfile.mkdtemp(prefix="fake_v4_ops0_", dir=os.environ.get("TMPDIR"))
    submit(rz, "ops", [q for q in oreq if q["key"].startswith(("o|check|", "o|plain|"))])
    run(rz, "--op-effect", "0")
    oz = {r["key"]: r for r in out(rz, "ops")}
    a0 = sum(oz[f"o|check|{rid}|{s}"]["ok"] for rid in small for s in range(4)) / (4 * len(small))
    check(f"--op-effect 0: check changes tokens and draws, not p ({a0 - acc('plain'):+.3f} vs plain)",
          abs(a0 - acc("plain")) < 0.03 and all(oz[f"o|check|{rid}|0"]["tokens"] == ob[f"o|check|{rid}|0"]["tokens"] for rid in small))

    # ---------------------------------------------------------------- chat answers vs proposer.parse / validate
    kinds = collections.Counter()
    tip = {"settings": {}, "entries": []}
    for k in range(1000):
        c, _, _ = F.canned_chat(f"g001|chat|P0|{k}", KNOBS)
        cand, err = P.parse(c)
        if err:
            kinds["broken"] += 1
            continue
        assert P.validate(cand, tip, KNOBS, rows) == [], (c, P.validate(cand, tip, KNOBS, rows))
        kinds["always" if cand["condition"] == "always" else "worded"] += 1
    check(f"chat: every parsable answer passes validate vs stock tip; split {dict(kinds)} ~ 40/50/10",
          abs(kinds["always"] - 400) < 60 and abs(kinds["worded"] - 500) < 60 and abs(kinds["broken"] - 100) < 40)
    check("chat: every canned condition x procedure is valid",
          all(P.validate(dict(why="", condition=c, procedure=p, settings={}), tip, KNOBS, rows) == [] for c in F.CONDS for p in F.PROCS))
    check("chat: deterministic per key", F.canned_chat("abc", KNOBS) == F.canned_chat("abc", KNOBS) and by["g001|chat|P0|3"]["content"] == F.canned_chat("g001|chat|P0|3", KNOBS)[0])
    wo = collections.Counter()
    for k in range(1000):
        c0, c3 = F.canned_chat(f"k{k}", KNOBS)[0], F.canned_chat(f"k{k}", KNOBS, 0.3)[0]
        a0_, a3 = P.parse(c0)[0], P.parse(c3)[0]
        if a0_ is None:
            assert a3 is None
            continue
        assert P.validate(a3, tip, KNOBS, rows) == [] and {k_: v for k_, v in a3.items() if k_ != "ops"} == {k_: v for k_, v in a0_.items() if k_ != "ops"}
        wo["ops" if a3["ops"] else "none"] += 1
    check(f"--chat-ops 0.3: that share of the parsable answers also carries ops, valid, the rest unchanged ({dict(wo)})",
          abs(wo["ops"] / (wo["ops"] + wo["none"]) - 0.3) < 0.05)

    # ---------------------------------------------------------------- idempotence across restarts
    before = {b: open(os.path.join(root, "q", "out", b + ".jsonl")).read() for b in ("g001_mixed",)}
    d_before = open(os.path.join(root, "q", "out", "g001_mixed.done")).read()
    run(root)
    after = open(os.path.join(root, "q", "out", "g001_mixed.jsonl")).read()
    # the error line is retried on restart (errors are unread, like the real dispatcher); nothing else is re-run
    new = after[len(before["g001_mixed"]):].splitlines()
    check("rerun: no key re-run except the error line; .done untouched",
          after.startswith(before["g001_mixed"]) and [json.loads(l)["key"] for l in new] == ["g001|bad|row"]
          and open(os.path.join(root, "q", "out", "g001_mixed.done")).read() == d_before)
    # same keys in a new batch: all already done -> .done at once, no lines; route cache answers new keys of known (row, cond)
    submit(root, "g002_repeat", reqs[:50])
    submit(root, "g002_route", [dict(key=f"g002|route|{rid}", mode="route", row=rid, condition="The image contains a chart.") for rid in ids[:10]])
    run(root)
    dr = done(root, "g002_repeat")
    check("repeat batch: keys done elsewhere are not re-run, .done written",
          not out(root, "g002_repeat") and dr["finished"] == 50 and dr["total"] == 50)
    check("route cache answers a new key for a known (row, condition)", all(r["cached"] for r in out(root, "g002_route")))
    # crash mid-batch (--max-results) then resume: no duplicate keys
    submit(root, "g003_resume", [dict(key=f"g003|{rid}|0", mode="read", row=rid, sample=0) for rid in ids[:100]])
    run(root, "--max-results", "37")
    n1, d1 = len(out(root, "g003_resume")), done(root, "g003_resume")
    run(root)
    o3 = out(root, "g003_resume")
    check(f"resume after a stop at {n1} results: 100 lines, no duplicate key, .done once complete",
          0 < n1 < 100 and d1 is None and len(o3) == 100 and len({r["key"] for r in o3}) == 100 and done(root, "g003_resume")["finished"] == 100)
    # duplicate key inside two batches submitted together: answered once, both batches done
    submit(root, "g004_a", [dict(key=f"g004|{rid}", mode="read", row=rid, sample=0) for rid in ids[:5]])
    time.sleep(0.01)
    submit(root, "g004_b", [dict(key=f"g004|{rid}", mode="read", row=rid, sample=0) for rid in ids[:5]] + [dict(key="g004|extra", mode="read", row=ids[9], sample=0)])
    run(root)
    check("same keys in two new batches: answered once (first batch), both .done",
          len(out(root, "g004_a")) == 5 and [r["key"] for r in out(root, "g004_b")] == ["g004|extra"] and done(root, "g004_b")["finished"] == 6)

    # ---------------------------------------------------------------- cancel
    submit(root, "g005_cancel", [dict(key=f"g005|{rid}", mode="read", row=rid, sample=0) for rid in ids[:40]])
    run(root, "--max-results", "10")                                  # some started and finished (the retried error line counts too)
    n5 = len(out(root, "g005_cancel"))
    open(os.path.join(root, "q", "cancel", "g005_cancel"), "w").close()
    run(root)
    d5 = done(root, "g005_cancel")
    check(f"cancel: finished requests kept, the rest cancelled ({d5['finished']} + {d5['cancelled']} = 40)",
          0 < n5 < 40 and len(out(root, "g005_cancel")) == n5 and d5["finished"] == n5 and d5["cancelled"] == 40 - n5)
    submit(root, "g006_cancel", [dict(key=f"g006|{rid}", mode="read", row=rid, sample=0) for rid in ids[:5]])
    submit(root, "g006_keep", [dict(key=f"g006|{rid}", mode="read", row=rid, sample=0) for rid in ids[:5]])
    open(os.path.join(root, "q", "cancel", "g006_cancel"), "w").close()
    run(root)
    check("cancel of a batch whose keys another batch also wants: they run for that batch",
          done(root, "g006_cancel")["cancelled"] == 5 and len(out(root, "g006_keep")) == 5 and done(root, "g006_keep")["finished"] == 5)

    # ---------------------------------------------------------------- polling mode (no --once), rate-limited
    root4 = tempfile.mkdtemp(prefix="fake_v4_", dir=os.environ.get("TMPDIR"))
    pr = subprocess.Popen(FAKE + ["--root", root4, "--rows", ROWS, "--rate", "100", "--jitter", "0.5", "--quiet"])
    try:
        time.sleep(0.3)
        t0 = time.time()
        submit(root4, "live", [dict(key=f"live|{rid}", mode="read", row=rid, sample=0) for rid in ids[:100]])
        while done(root4, "live") is None and time.time() - t0 < 10:
            time.sleep(0.1)
        dt = time.time() - t0
    finally:
        pr.terminate()
        pr.wait()
    check(f"polling mode: a batch dropped in later is picked up and done; 100 results at --rate 100 took {dt:.2f} s",
          done(root4, "live") is not None and len(out(root4, "live")) == 100 and 0.8 < dt < 3.5)
    print("ALL GREEN")


if __name__ == "__main__":
    main()
