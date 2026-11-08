"""One lineage: find -> vary -> select -> write -> verify -> next generation. Runs without a card on the node and hands GPU jobs to the queue (q.txt), so the jobs share the cards with everything else.
Each generation:
 scan     categories of the current model from its own traces; those below chance first, then by points lost.
 vary     the model reads its failures on half 0 and writes notes; blind notes as control.
 select   on half 0 only. Half 1 is untouched by all choices below.
 write    best read note and best blind note of each category into the last layer as gated units.
 verify   edited model on half 0 + 1, no note in the prompt. A run that goes wrong is stopped early and said so.
 pass on  edits with a clear gain on half 1 are joined and the whole pool rerun; its traces start the next generation.
Writes in RUN/g<N>: scan.json, table.jsonl (parent note gain and written gain per rule), demos/, gen.json; RUN/lineage.txt is the running record.
"""
import argparse, glob, json, os, random, re, shutil, subprocess, sys, time, zlib, collections

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--traces", required=True)     # generation 0 traces over the pool
ap.add_argument("--gens", type=int, default=2)
ap.add_argument("--top", type=int, default=8)
ap.add_argument("--jobs", type=int, default=2)
ap.add_argument("--proposals", type=int, default=24)
ap.add_argument("--tries", type=int, default=6)
ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--confirm", type=int, default=8)
ap.add_argument("--trunc-stop", type=float, default=.4)
a = ap.parse_args()
SITE = os.environ.get("EVO_SITE", "node")
if SITE == "vesta":       # Vesta jobs may only start through `gpu bg`, so this script leaves its lines in R/evoq.txt and the Mac starts them (vesta/vesta_bridge.py)
    N = "/data/fd/mingdazhang/selfedit"; R = f"{N}/rsi/mm"
    MODEL = f"{N}/models/Qwen3.8-27B-p8box"
    POOL = os.environ["EVO_POOL"]
    LIB = f"{N}/probe/library"
    logs_of = lambda name: glob.glob(f"/data/fd/mingdazhang/gpu-logs/selfedit-{name}.log")
else:
    N = "/home/bedicloud/sharestore2/zxc/selfedit"; R = f"{N}/rsi/mm"
    MODEL = "/dev/shm/selfedit/models/Qwen3.8-27B-p8box"
    POOL = f"{R}/pool.jsonl"
    LIB = f"{N}/xf/low/qwen38/library"
    logs_of = lambda name: glob.glob(f"{N}/logs/selfedit-{name}-*")
TAIL = f"--mm --model {MODEL} --pool {POOL} --dirlib {LIB} --crn 1 --seed 101 --tp 1 --max-model-len 36864 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --effort medium --budget 8192 --mb 128 --max-num-seqs 64 --gpu-util 0.85"
ENV = f" ;; SE_MODEL={MODEL}" if SITE == "node" else ""
pool = {p["id"]: p for p in map(json.loads, open(POOL))}
half = lambda i: zlib.crc32(i.encode()) % 2
os.makedirs(a.run, exist_ok=True)


def say(s):
    line = time.strftime("%H:%M:%S ") + s
    print(line, flush=True)
    open(f"{a.run}/lineage.txt", "a").write(line + "\n")


def submit(name, cmd):
    """Head of the queue, once. Names stay in q.done for ever, so a rerun of this script finds its finished jobs by their files."""
    if SITE == "vesta":
        q = open(f"{R}/evoq.txt").read() if os.path.exists(f"{R}/evoq.txt") else ""
        if not any(l.startswith(name + "\t") for l in q.split("\n")):
            open(f"{R}/evoq.txt", "a").write(f"{name}\t{cmd}\n")
        return
    done = set(open(f"{R}/q.done").read().split()) if os.path.exists(f"{R}/q.done") else set()
    q = open(f"{R}/q.txt").read()
    if name in done or any(l.startswith(name + "\t") for l in q.split("\n")):
        return
    open(f"{R}/q.txt.new", "w").write(f"{name}\t-\t{cmd}\n" + q)
    os.replace(f"{R}/q.txt.new", f"{R}/q.txt")


def wait(paths, what, every=60, watch=None):
    while True:
        miss = [p for p in paths if not os.path.exists(p)]
        if not miss:
            return
        if watch:
            watch()
        time.sleep(every)


def load(f):
    d = collections.defaultdict(list)
    for fn in glob.glob(f):
        for l in open(fn):
            r = json.loads(l)
            if "id" in r:
                d[r["id"]].append(r)
    return d


def rate(rs):
    return sum(bool(r["ok"]) and r["finish"] != "length" for r in rs) / max(1, len(rs))


def gain(after, before, ids=None, on=1):
    """Gain on one half with a bootstrap over problems; two-option categories are averaged over the right answer so a lean to one option does not count."""
    ids = sorted(i for i in (ids or after) if i in after and i in before and half(i) == on and (on == 0 or len(after[i]) == len(before[i])))
    if not ids:
        return None
    d = {i: rate(after[i]) - rate(before[i]) for i in ids}
    two = all(len(pool[i].get("options") or []) == 2 for i in ids) and len({str(pool[i]["answer"]).strip("()") for i in ids}) == 2
    by = collections.defaultdict(list)
    for i in ids:
        by[str(pool[i]["answer"]).strip("()") if two else "all"].append(i)

    def m(groups):
        return sum(sum(d[i] for i in g) / len(g) for g in groups) / len(groups)
    rng = random.Random(0)
    bs = sorted(m([rng.choices(g, k=len(g)) for g in by.values()]) for _ in range(2000))
    return dict(gain=m(list(by.values())), lo=bs[50], hi=bs[1949], n=len(ids), two=two, by={k: len(v) for k, v in by.items()})


def scan(tr):
    cats = collections.defaultdict(list)
    for i, rs in tr.items():
        if i in pool and pool[i].get("image_paths") and half(i) == 0:      # half 1 plays no part in any choice
            cats[pool[i]["category"]].append(i)
    rows = []
    for c, ids in cats.items():
        r = sum(rate(tr[i]) for i in ids) / len(ids)
        opt = [len(pool[i].get("options") or []) for i in ids]
        chance = sum(1 / o if o else 0 for o in opt) / len(opt)
        wrong0 = sum(rate(tr[i]) < 1 for i in ids if half(i) == 0)
        rows.append(dict(cat=c, n=len(ids), rate=r, chance=chance, below=chance > 0 and r < chance, lost=(1 - r) * len(ids), wrong0=wrong0, half1=sum(1 for p in pool.values() if p["category"] == c and p.get("image_paths") and half(p["id"]) == 1)))
    ok = [x for x in rows if x["wrong0"] >= 4 and x["half1"] >= 5]
    ok.sort(key=lambda x: (not x["below"], -x["lost"]))
    return rows, ok[:a.top]


def demos(out, rs, note):
    w = [r for r in rs if not r["ok"]][:4] + [r for r in rs if r["ok"]][:2]
    with open(out, "w") as f:
        for r in w:
            f.write(f"=== {r['id']} gold {r['answer']} ok {r['ok']} tokens {r['tokens']} finish {r['finish']}\n{(r.get('reasoning') or '')[:1800]}\n...\n{(r.get('reasoning') or '')[-500:]}\nANSWER {(r.get('content') or '')[-150:]}\n\n")


engine, traces = None, a.traces
for g in range(a.gens):
    G = f"{a.run}/g{g}"
    os.makedirs(f"{G}/demos", exist_ok=True)
    tr = load(traces)
    rows, top = scan(tr)
    json.dump(rows, open(f"{G}/scan.json", "w"), indent=0)
    say(f"g{g} scan: {len(tr)} problems, mean {sum(rate(v) for v in tr.values()) / len(tr):.3f}; chosen: " + "; ".join(f"{x['cat']} {x['rate']:.2f}{' BELOW' if x['below'] else ''}" for x in top))
    # vary and select
    parts = [top[j::a.jobs] for j in range(a.jobs)]
    flags = []
    for j, part in enumerate(parts):
        if not part:
            continue
        O = f"{G}/s{j}"
        cats = ",".join(x["cat"].replace(" ", "~") for x in part)
        en = f" --engine {engine}" if engine else ""
        submit(f"{os.path.basename(a.run)}_g{g}_s{j}", f"python -u rsi/mm_selfrule.py {O} --traces {traces} --cats {cats} --proposals {a.proposals} --tries {a.tries} --confirm {a.confirm}{en} --samples {a.samples} {TAIL}{ENV}")
        flags.append(O)

    def finished(O):
        return os.path.exists(f"{O}/report.txt") and open(f"{O}/report.txt").read().rstrip().endswith("done")
    while not all(finished(O) for O in flags):
        for j, O in enumerate(flags):
            for lf in logs_of(f"{os.path.basename(a.run)}_g{g}_s{j}"):
                if not finished(O) and "Traceback" in open(lf, errors="ignore").read()[-20000:]:
                    say(f"g{g} s{j} failed, see {lf}")
                    sys.exit(1)
        time.sleep(120)
    best = [dict(json.loads(l), dir=O) for O in flags if os.path.exists(f"{O}/best.jsonl") for l in open(f"{O}/best.jsonl")]
    say(f"g{g} select: {len(best)} rules from {len(flags)} jobs")
    # write and verify
    W = f"{G}/write"
    os.makedirs(W, exist_ok=True)
    arms = []
    skip = 0
    for n_rule, b in enumerate(best):
        tag = f"g{g}_{n_rule}_{b['tag'][:40]}_{b['arm']}"
        f = f"{W}/{tag}.tsv"
        open(f, "w").write(f"{tag}\t{b['note'].replace(chr(10), ' ').replace(chr(9), ' ')}\n")
        en = f" --engine {engine}" if engine else ""
        submit(f"{os.path.basename(a.run)}_w_{tag}", f"python -u rsi/mm_gate.py --model {MODEL} --traces {traces} --others {traces} --pool {POOL} --notes {f} --out {W}/gt --arms {W} --cat {b['cat'].replace(' ', '~')} --skip-units {skip}{en}{ENV}")
        skip += 400           # a note is at most about 60 words; mm_gate refuses more than this many tokens
        arms.append((b, tag))
    table = {}

    def tick():
        for b, tag in arms:
            arm, fin = f"{W}/gt_{tag}.json", f"{W}/gt_{tag}.{b['tag']}.jsonl"
            stop = f"{W}/gt_{tag}.stopped"
            pp = f"{W}/gt_{tag}.{b['tag']}.partial"
            if not os.path.exists(arm) or os.path.exists(fin) or os.path.exists(stop):
                continue
            if not os.path.exists(f"{W}/{b['tag']}.problems.jsonl"):
                ids = [p for p in pool.values() if p["category"] == b["cat"] and p.get("image_paths")]
                with open(f"{W}/{b['tag']}.problems.jsonl", "w") as o:
                    for p in ids:
                        o.write(json.dumps(p, ensure_ascii=False) + "\n")
            full = json.load(open(arm))
            en = (json.load(open(engine))[0]["engine"] if engine else []) + full[0]["engine"]
            if engine and not full[0].get("joined"):
                full[0]["engine"], full[0]["joined"] = en, True
                json.dump(full, open(arm, "w"))
            submit(f"{os.path.basename(a.run)}_v_{tag}", f"python -u rsi/probe_base.py {W} --probes {arm} --sets {b['tag']}:{W}/{b['tag']}.problems.jsonl --curve 2048,4096,8192 --samples {a.confirm} {TAIL}{ENV}")
            if os.path.exists(pp):
                rs = [json.loads(l) for l in open(pp)]
                rs = [r for r in rs if "id" in r]
                tr_ = sum(r["finish"] == "length" for r in rs) / max(1, len(rs))
                if len(rs) >= 48 and tr_ > a.trunc_stop:
                    if SITE == "vesta":
                        open(f"{R}/evostop.txt", "a").write(f"{os.path.basename(a.run)}_v_{tag}\n")
                    else:
                        subprocess.run(["pkill", "-f", f"gt_{tag}[.]json"])
                    open(stop, "w").write(f"trunc {tr_:.2f} of {len(rs)}\n")
                    say(f"g{g} {tag}: stopped early, truncation {tr_:.2f} of {len(rs)} records")
    while True:
        tick()
        pending = [tag for b, tag in arms if not os.path.exists(f"{W}/gt_{tag}.{b['tag']}.jsonl") and not os.path.exists(f"{W}/gt_{tag}.stopped")]
        for b, tag in arms:
            log = logs_of(f"{os.path.basename(a.run)}_w_{tag}")
            if log and "Traceback" in open(log[0], errors="ignore").read() and not os.path.exists(f"{W}/gt_{tag}.json") and not os.path.exists(f"{W}/gt_{tag}.stopped"):
                open(f"{W}/gt_{tag}.stopped", "w").write("write failed\n")
                say(f"g{g} {tag}: write failed, see {log[0]}")
        if not pending:
            break
        time.sleep(120)
    keep, rows_all = {}, []
    with open(f"{G}/table.jsonl", "w") as t:
        for b, tag in arms:
            base = load(f"{b['dir']}/{b['tag']}.base1.jsonl")
            base0 = load(f"{b['dir']}/{b['tag']}.base0.jsonl")
            note = load(f"{b['dir']}/{b['tag']}.{b['arm']}.note1.jsonl")
            fin = f"{W}/gt_{tag}.{b['tag']}.jsonl"
            pp = fin if os.path.exists(fin) else f"{W}/gt_{tag}.{b['tag']}.partial"
            if not os.path.exists(pp):
                say(f"g{g} {tag}: no records")
                continue
            child = load(pp)
            rs = [r for v in child.values() for r in v]
            start = b["note"][:25]
            row = dict(tag=tag, cat=b["cat"], arm=b["arm"], note=b["note"], fit=b["fit"], parent=gain(note, base), child=gain(child, base), child0=gain(child, base0, on=0), recite=sum((r.get("reasoning") or "").startswith(start) for r in rs) / max(1, len(rs)), trunc=sum(r["finish"] == "length" for r in rs) / max(1, len(rs)), tokens=sum(r["tokens"] for r in rs) / max(1, len(rs)), stopped=os.path.exists(f"{W}/gt_{tag}.stopped"))
            t.write(json.dumps(row, ensure_ascii=False) + "\n")
            demos(f"{G}/demos/{tag}.txt", rs, b["note"])
            c, p = row["child"], row["parent"]
            say(f"g{g} {tag}: parent {p and round(p['gain'], 3)} child {c and round(c['gain'], 3)} [{c and round(c['lo'], 3)},{c and round(c['hi'], 3)}] recite {row['recite']:.2f} trunc {row['trunc']:.2f} tok {row['tokens']:.0f}")
            rows_all.append((b, tag, row))
    # pass on: half 0 names one rule per category, by how well it works once written (a note that helps in the prompt need not help written, 11-06); half 1 only passes or fails that one
    for cat in {b["cat"] for b, _, _ in rows_all}:
        b, tag, row = max((x for x in rows_all if x[0]["cat"] == cat), key=lambda x: (x[2]["child0"] or {"gain": -9})["gain"])
        c = row["child"]
        if c and c["lo"] > 0 and not row["stopped"]:
            keep[cat] = (tag, c["gain"])
    en0 = json.load(open(engine))[0]["engine"] if engine else []
    en = list(en0)
    for cat, (tag, _) in keep.items():
        en += json.load(open(f"{W}/gt_{tag}.json"))[0]["engine"][len(en0):]      # each arm file is the generation's edits, then its own
    say(f"g{g} pass on: {sorted(keep)}" if keep else f"g{g} pass on: nothing cleared half 1")
    json.dump(dict(keep={k: v[0] for k, v in keep.items()}), open(f"{G}/gen.json", "w"))
    if not keep:
        break
    nxt = f"{G}/next.json"
    json.dump([{"label": f"g{g + 1}", "effort": "medium", "engine": en}], open(nxt, "w"))
    submit(f"{os.path.basename(a.run)}_whole_g{g + 1}", f"python -u rsi/probe_base.py {G} --probes {nxt} --sets learn:{POOL} --curve 2048,4096,8192 --samples 2 {TAIL.replace('--crn 1 --seed 101', '--seed 31')}{ENV}")
    wait([f"{G}/g{g + 1}.learn.jsonl"], "whole pool")
    engine, traces = nxt, f"{G}/g{g + 1}.learn.jsonl"
    new = load(traces)
    say(f"g{g} to g{g + 1}: whole pool {sum(rate(v) for v in tr.values()) / len(tr):.3f} -> {sum(rate(v) for v in new.values()) / len(new):.3f}")
say("lineage done")
