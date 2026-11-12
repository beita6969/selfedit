"""v4 PILOT generation (rehearsal, coordinator 11-10). Results only in V4/pilot/; never fed to the official run, its ledger or any prompt.
Steps: a propose (32 cards, loop/proposer.py as is) -> b route (non-always entries, all 768 learn rows) -> c reads on routed SHOW+MONITOR rows
(never JUDGE) -> d arm-B one-factor cells on MONITOR -> e table. All through the dispatcher at priority 6-8 (Builder batches go first).
usage: python pilot.py [step ...]   (default: all; each step skips work whose batch already finished)"""
import collections, glob, json, math, os, sys, time
sys.path[:0] = ["/workspace/fudan/mingdazhang/selfedit-v4/engine_v4/pilot_deps", os.path.dirname(os.path.abspath(__file__))]
import proposer as PR

V4 = "/data/fd/mingdazhang/selfedit/v4"
PD = f"{V4}/pilot"
os.makedirs(PD, exist_ok=True)
QIN, QOUT = f"{V4}/q/in", f"{V4}/q/out"
NCARDS, SEED = 32, "pilot1110"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
rows = PR.load_rows(f"{V4}/rows/learn.jsonl")
knobs = json.load(open("/workspace/fudan/mingdazhang/selfedit-v4/engine_v4/pilot_deps/knobs.json"))
role = {i: r["v4_role"] for i, r in rows.items()}
bench = {i: r.get("bench") or i.split("/")[0] for i, r in rows.items()}
clus = {i: (r.get("pic") or (r.get("img_sha256") or [i])[0]) for i, r in rows.items()}
SM = sorted(i for i in rows if role[i] in ("show", "monitor"))
MON = sorted(i for i in rows if role[i] == "monitor")


def results(pattern):
    out = []
    for f in glob.glob(f"{QOUT}/{pattern}.jsonl"):
        out += [json.loads(l) for l in open(f)]
    return out


def parent():
    P = collections.defaultdict(list)
    for r in results("p000_parent_*"):
        if "error" not in r:
            P[r["row"]].append(r)
    return P


def submit(name, reqs, wait=True):
    f = f"{QIN}/{name}.jsonl"
    if not os.path.exists(f):
        with open(f + ".tmp", "w") as fh:
            for r in reqs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(f + ".tmp", f)
        json.dump(dict(t=time.time(), n=len(reqs)), open(f"{PD}/{name}.submitted", "w"))
        log(f"submitted {name}: {len(reqs)}")
    while wait and not os.path.exists(f"{QOUT}/{name}.done"):
        time.sleep(10)
    t0 = json.load(open(f"{PD}/{name}.submitted"))["t"] if os.path.exists(f"{PD}/{name}.submitted") else None
    t1 = json.load(open(f"{QOUT}/{name}.done"))["t"] if os.path.exists(f"{QOUT}/{name}.done") else None
    return {r["key"]: r for r in results(name) if "error" not in r}, (t1 - t0 if t0 and t1 else None)


def step_propose():
    P = parent()
    reqs, meta = [], []
    for k in range(NCARDS):
        ids = PR.cards(rows, P, k, SEED)
        system, user = PR.card_prompt(rows, P, ids, None, knobs)
        reqs.append(dict(key=f"pilot|prop|{k}", mode="chat", engine="E", priority=6,
                         messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]))
        meta.append(dict(k=k, card=ids, chars=len(system) + len(user)))
    if not os.path.exists(f"{PD}/card_prompt_0.txt"):
        open(f"{PD}/card_prompt_0.txt", "w").write(reqs[0]["messages"][0]["content"] + "\n\n=====USER=====\n\n" + reqs[0]["messages"][1]["content"])
    res, wall = submit("pilot_prop", reqs)
    E = []
    for m in meta:
        r = res.get(f"pilot|prop|{m['k']}")
        e = dict(k=m["k"], card=m["card"], prompt_chars=m["chars"])
        if not r:
            e.update(valid=False, problems=["no result"])
        else:
            cand, err = PR.parse(r["content"])
            probs = [err] if err else PR.validate(cand, None, knobs, rows)
            e.update(cand=cand, valid=not probs, problems=probs, finish=r["finish"], tokens=r["tokens"], content_tail=r["content"][-1500:])
        E.append(e)
    json.dump(dict(wall_s=wall, entries=E), open(f"{PD}/entries.json", "w"), indent=1, ensure_ascii=False)
    with open(f"{PD}/entries.txt", "w") as f:
        for e in E:
            c = e.get("cand") or {}
            kind = ("text+settings" if c.get("procedure") and c.get("settings") else "text" if c.get("procedure") else "settings" if c.get("settings") else "-")
            f.write(f"=== entry {e['k']}  valid={e['valid']}  kind={kind}  finish={e.get('finish')}  tokens={e.get('tokens')}\n")
            if e["problems"]:
                f.write(f"problems: {e['problems']}\n")
            f.write(f"condition: {c.get('condition')}\nprocedure: {c.get('procedure')}\nsettings: {json.dumps(c.get('settings'))}\nwhy: {c.get('why')}\n")
            if not c:
                f.write(f"raw tail: {e.get('content_tail', '')[-800:]}\n")
            f.write("\n")
    v = [e for e in E if e["valid"]]
    kinds = collections.Counter(("text+settings" if e["cand"]["procedure"] and e["cand"]["settings"] else "text" if e["cand"]["procedure"] else "settings") for e in v)
    log(f"propose: wall {wall:.0f}s, valid {len(v)}/{len(E)}, kinds {dict(kinds)}, always {sum(PR._is_always(e['cand']['condition']) for e in v)}")


def entries():
    return [e for e in json.load(open(f"{PD}/entries.json"))["entries"] if e["valid"]]


def step_route():
    reqs = []
    for e in entries():
        if PR._is_always(e["cand"]["condition"]):
            continue
        for i in sorted(rows):
            reqs.append(dict(key=f"pilot|route|{e['k']}|{i}", mode="route", engine="E", priority=6, row=i, condition=e["cand"]["condition"]))
    res, wall = submit("pilot_route", reqs)
    json.dump(dict(wall_s=wall, n=len(reqs)), open(f"{PD}/route_wall.json", "w"))
    log(f"route: {len(reqs)} calls, wall {wall}")


def routed():
    R = {}
    rr = {r["key"]: r for r in results("pilot_route") if "error" not in r}
    for e in entries():
        if PR._is_always(e["cand"]["condition"]):
            R[e["k"]] = {i: True for i in rows}
        else:
            R[e["k"]] = {i: rr[f"pilot|route|{e['k']}|{i}"]["yes"] for i in rows if f"pilot|route|{e['k']}|{i}" in rr}
    return R, rr


def settings_of(c):
    s = c.get("settings") or {}
    return {"min_image_tokens": s.get("min_image_tokens"), "max_image_tokens": s.get("max_image_tokens"), "temperature": s.get("temperature")}


def step_read():
    R, _ = routed()
    reqs = []
    for e in entries():
        for i in SM:
            if R[e["k"]].get(i):
                reqs.append(dict(key=f"pilot|read|{e['k']}|{i}|0", mode="read", engine="E", priority=7, row=i, sample=0,
                                 prefix=e["cand"].get("procedure"), settings=settings_of(e["cand"])))
    assert not any(role[q["row"]] == "judge" for q in reqs)
    res, wall = submit("pilot_read", reqs)
    log(f"read: {len(reqs)} reads, wall {wall}")


def cells():
    out = []
    for k, d in knobs["settings"].items():
        for v in d["values"]:
            if not PR._same(v, d["stock"]):
                out.append((f"{k}={v}", {k: v}))
    return out


def step_armb(wait=True):
    reqs = []
    for name, s in cells():
        st = {"min_image_tokens": None, "max_image_tokens": None, "temperature": None, **s}
        for i in MON:
            reqs.append(dict(key=f"pilot|armb|{name}|{i}|0", mode="read", engine="E", priority=8, row=i, sample=0, prefix=None, settings=st))
    res, wall = submit("pilot_armb", reqs, wait=wait)
    log(f"armb: {len(reqs)} reads, wall {wall}")


def paired(P, got):
    """got: {row: result}. d_i = ok - mean(parent ok); cluster-robust se by picture."""
    d, tok, ptok, up, down = {}, [], [], 0, 0
    for i, r in got.items():
        ps = [s for s in P.get(i, []) if s.get("ok") is not None]
        if not ps:
            continue
        pm = sum(bool(s["ok"]) for s in ps) / len(ps)
        d[i] = float(bool(r["ok"])) - pm
        tok.append(r["tokens"]); ptok.append(sum(s["tokens"] for s in ps) / len(ps))
        pr = PR._is_right(ps)
        up += bool(r["ok"]) and not pr
        down += (not r["ok"]) and pr
    n = len(d)
    if not n:
        return dict(n=0)
    m = sum(d.values()) / n
    C = collections.defaultdict(float)
    for i, x in d.items():
        C[clus[i]] += x - m
    G = len(C)
    se = math.sqrt(sum(v * v for v in C.values()) * (G / (G - 1) if G > 1 else 1)) / n
    return dict(n=n, clusters=G, mean=m, se=se, tok=sum(tok) / n, ptok=sum(ptok) / n, up=up, down=down)


def step_table():
    P = parent()
    R, rr = routed()
    E = entries()
    reads = {r["key"]: r for r in results("pilot_read") if "error" not in r}
    lines, recs = [], []
    hdr = "| entry | kind | routed all | show | judge | monitor | rows read | mean d (pp) | se (pp) | whole-set d (pp) | tok/sample vs parent | up / down |"
    lines += [hdr, "|" + "---|" * 12]
    for e in E:
        k, c = e["k"], e["cand"]
        rk = R[k]
        share = lambda ids: (sum(bool(rk.get(i)) for i in ids) / max(1, sum(i in rk for i in ids)))
        by = {ro: share([i for i in rows if role[i] == ro]) for ro in ("show", "judge", "monitor")}
        all_ = share(list(rows))
        got = {i: reads[f"pilot|read|{k}|{i}|0"] for i in SM if f"pilot|read|{k}|{i}|0" in reads}
        s = paired(P, got)
        kind = "text+settings" if c["procedure"] and c["settings"] else "text" if c["procedure"] else "settings"
        rec = dict(k=k, kind=kind, condition=c["condition"], routed_all=all_, routed=by, per_bench={b: share([i for i in rows if bench[i] == b]) for b in sorted(set(bench.values()))}, **s)
        recs.append(rec)
        if s["n"]:
            lines.append(f"| {k} | {kind} | {all_:.2f} | {by['show']:.2f} | {by['judge']:.2f} | {by['monitor']:.2f} | {s['n']} | {100 * s['mean']:+.1f} | {100 * s['se']:.1f} | {100 * all_ * s['mean']:+.2f} | {s['tok']:.0f} vs {s['ptok']:.0f} | {s['up']} / {s['down']} |")
        else:
            lines.append(f"| {k} | {kind} | {all_:.2f} | {by['show']:.2f} | {by['judge']:.2f} | {by['monitor']:.2f} | 0 | | | | | |")
    # router cost and quality
    rl = [r for r in rr.values() if not r.get("cached")]
    bad = [r for r in rr.values() if not r["content"].strip().lower().lstrip(" *\"'").startswith(("yes", "no"))]
    lat = sorted(r["t_done"] - r["t_submit"] for r in rl)
    rcost = dict(calls=len(rr), uncached=len(rl), malformed=len(bad), malformed_examples=[r["content"] for r in bad[:10]],
                 latency_p50=lat[len(lat) // 2] if lat else None, latency_p95=lat[int(.95 * len(lat))] if lat else None,
                 gen_tokens_mean=(sum(r.get("tokens") or 0 for r in rl) / len(rl)) if rl else None,
                 prompt_tokens_mean=(sum(r.get("prompt_tokens") or 0 for r in rl) / len(rl)) if rl else None,
                 wall=json.load(open(f"{PD}/route_wall.json")) if os.path.exists(f"{PD}/route_wall.json") else None)
    # arm B
    ab = {r["key"]: r for r in results("pilot_armb") if "error" not in r}
    blines = ["| cell | rows read | mean d (pp) | se (pp) | tok/sample vs parent | up / down |", "|---|---|---|---|---|---|"]
    brecs = []
    for name, s_ in cells():
        got = {i: ab[f"pilot|armb|{name}|{i}|0"] for i in MON if f"pilot|armb|{name}|{i}|0" in ab}
        s = paired(P, got)
        brecs.append(dict(cell=name, **s))
        if s["n"]:
            blines.append(f"| {name} | {s['n']} | {100 * s['mean']:+.1f} | {100 * s['se']:.1f} | {s['tok']:.0f} vs {s['ptok']:.0f} | {s['up']} / {s['down']} |")
    # transcripts: two best and two worst entries by mean d (n >= 5), 3 SHOW rows each (flipped rows first)
    ok = sorted([r for r in recs if r.get("n", 0) >= 5], key=lambda r: r["mean"])
    picks = ok[-2:][::-1] + ok[:2]
    tx = []
    for r in picks:
        k = r["k"]
        sh = [i for i in SM if role[i] == "show" and f"pilot|read|{k}|{i}|0" in reads]
        sh.sort(key=lambda i: -abs(float(bool(reads[f"pilot|read|{k}|{i}|0"]["ok"])) - sum(bool(s["ok"]) for s in P[i]) / max(1, len(P[i]))))
        tx.append(f"### entry {k} (mean d {100 * r['mean']:+.1f} pp on {r['n']} rows)\ncondition: {r['condition']}\n")
        for i in sh[:3]:
            e = reads[f"pilot|read|{k}|{i}|0"]
            p0 = sorted(P[i], key=lambda s: s["sample"])[0]
            tx.append(f"- row {i} (gold {rows[i]['answer']!r}); parent ok {[bool(s['ok']) for s in sorted(P[i], key=lambda s: s['sample'])]}\n"
                      f"  - parent s0 ({p0['tokens']} tok) final: {p0['content'][-400:]!r}\n  - with entry ok={e['ok']} ({e['tokens']} tok) final: {e['content'][-400:]!r}\n")
    json.dump(dict(entries=recs, router=rcost, armb=brecs), open(f"{PD}/table.json", "w"), indent=1)
    ej = json.load(open(f"{PD}/entries.json"))
    allE = ej["entries"]
    v = [e for e in allE if e["valid"]]
    kinds = collections.Counter(r["kind"] for r in recs)
    md = [f"# v4 pilot generation (rehearsal) 11-10\n\nNOT part of the official run. Stock parent (4 samples, medium), 32 cards from loop/proposer.py, seed {SEED}.\n",
          f"## a. Proposals\nvalid {len(v)}/{len(allE)}; kinds {dict(kinds)}; batch wall {ej['wall_s']:.0f} s; mean proposer tokens {sum(e.get('tokens') or 0 for e in allE) / len(allE):.0f}; "
          f"problems of invalid: {[e['problems'] for e in allE if not e['valid']]}\nAll entries verbatim: {PD}/entries.txt\n",
          f"## b. Router\n{json.dumps(rcost, indent=1)}\nPer-bench routed share per entry: table.json.\n",
          "## c/e. Entries on SHOW+MONITOR (never JUDGE), 1 sample vs parent mean of 4; d in pp, se clustered by picture; whole-set d = routed share (all learn) x d\n", *lines, "",
          "## d. Arm B one-factor cells on MONITOR (144 rows)\n", *blines, "", "## Transcripts (2 best, 2 worst)\n", *tx]
    open(f"{PD}/pilot_report.md", "w").write("\n".join(md))
    log("table written", f"{PD}/pilot_report.md")


STEPS = dict(propose=step_propose, route=step_route, read=step_read, armb=step_armb, table=step_table)
if __name__ == "__main__":
    want = sys.argv[1:] or list(STEPS)
    if "armb" in want:          # arm B does not depend on the proposals: queue it right away (lowest priority), collect later
        step_armb(wait=False)
    for s in want:
        log("step", s)
        if s == "propose":
            P = parent()
            show = [i for i in rows if role[i] == "show"]
            while sum(1 for i in show if len(P.get(i, [])) >= 4) < len(show):
                log(f"waiting for the parent cache on SHOW rows: {sum(1 for i in show if len(P.get(i, [])) >= 4)}/{len(show)}")
                time.sleep(60)
                P = parent()
        STEPS[s]()
