"""E1 (internals agent 11-12): counting procedure g002c26 as weight units, W generation reads on the 324 SIGNAL rows, floor-only base.
q CELL PRIO [S0 S1]   queue W reads of edits/<CELL>.json (op surgeon), settings = floor (min_image_tokens 4096), no prefix; keys E1|CELL|row|s
t CELL...             table vs the cached floor-only row mean (x|2988ed164be18b6a, s0-7): (a) 75 routed rows by bench (+ TEXT = cached t005r
                      reads, x|d5a30f27c6fe44c1 = floor + g002c26 prefix), (b) the 14 routed rows no fit has seen, (c) 249 unrouted rows with the
                      18 held-out ones apart; accuracy, thinking tokens, not-fit / error counts."""
import glob, json, math, os, sys, collections
V4 = "/data/fd/mingdazhang/selfedit/v4"; W = "/data/fd/mingdazhang/selfedit/int/work"; ED = f"{W}/cells_proc/edits"
ROWS = {r["id"]: r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl")) if r.get("v4_role") in ("show", "monitor")}
assert len(ROWS) == 324
FLOOR = {"min_image_tokens": 4096, "max_image_tokens": None, "temperature": None}
ROUTED = [l.strip() for l in open(f"{V4}/scratch/e1_routed_signal_rows.txt") if l.strip()]; assert len(ROUTED) == 75
TEST = {l.split()[0] for l in open(f"{W}/proc_test_items.txt") if l.strip()}
UHELD = {l.split()[0] for l in open(f"{W}/proc_unrouted_items.txt") if l.strip()}
w = sys.argv[1]
VW = "/data/fd/mingdazhang/selfedit/v4wv"; STOCK = {"min_image_tokens": None, "max_image_tokens": None, "temperature": None}
if w == "qvt":                                 # stacked base, NO edit, the g002c26 (t005r) prefix on the 75 routed rows, STOCK requests
    prio, s0, s1 = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    pre = json.load(open(f"{V4}/runs/main_1110/tips/t005r.json"))["entries"][1]["procedure"]
    Q = [dict(key=f"E1V|TEXT|{r}|{s}", mode="read", engine="W", priority=prio, row=r, sample=s, prefix=pre, settings=STOCK, edit=None)
         for s in range(s0, s1 + 1) for r in ROUTED]
    b = f"e1v_TEXT_s{s0}{s1}"; p = f"{VW}/q/in/{b}.jsonl"; assert not os.path.exists(p), p
    open(p + ".tmp", "w").write("".join(json.dumps(q) + "\n" for q in Q)); os.replace(p + ".tmp", p); print("queued (v4wv)", b, len(Q), "prio", prio)
    sys.exit()
if w == "qv":                                  # edit-on-edit stack: base = vp2m512 (W worker --model vp2m512, queue root v4wv), STOCK requests
    cell, prio, s0, s1 = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    edit = None if cell == "NULL" else {"op": "surgeon", "file": f"{ED}/{cell}.json"}
    if edit: assert os.path.exists(edit["file"]), edit
    Q = [dict(key=f"E1V|{cell}|{r}|{s}", mode="read", engine="W", priority=prio, row=r, sample=s, prefix=None, settings=STOCK, edit=edit)
         for s in range(s0, s1 + 1) for r in (ROUTED if "--routed" in sys.argv else sorted(ROWS))]
    b = f"e1v_{cell.replace('+', 'p').replace('.', '_')}_s{s0}{s1}" + ("_routed" if "--routed" in sys.argv else ""); p = f"{VW}/q/in/{b}.jsonl"; assert not os.path.exists(p), p
    open(p + ".tmp", "w").write("".join(json.dumps(q) + "\n" for q in Q)); os.replace(p + ".tmp", p); print("queued (v4wv)", b, len(Q), "prio", prio)
    sys.exit()
def qwrite(b, Q):
    p = f"{V4}/q/in/{b}.jsonl"; assert not os.path.exists(p), p
    open(p + ".tmp", "w").write("".join(json.dumps(q) + "\n" for q in Q)); os.replace(p + ".tmp", p); print("queued", b, len(Q), "prio", Q[0]["priority"])
def bname(cell): return cell.replace('+', 'p').replace('.', '_')
if w == "round":                               # iterated compile round (internals agent 11-12): floor base, no prefix, samples 0-1
    cell = sys.argv[2]; f = f"{ED}/{cell}.json"; nf = f"{ED}/RAND__{cell}.json"; assert os.path.exists(f), f
    mk = lambda c, rows, prio: [dict(key=f"E1|{c}|{r}|{s}", mode="read", engine="W", priority=prio, row=r, sample=s, prefix=None, settings=FLOOR,
                                     edit={"op": "surgeon", "file": f"{ED}/{c}.json"}) for s in (0, 1) for r in rows]
    qwrite(f"e1r_{bname(cell)}_loop", mk(cell, ROUTED, -9))                                   # 1. loop reads: 75 routed x s0-1 = 150
    qwrite(f"e1r_{bname(cell)}_rest", mk(cell, [r for r in sorted(ROWS) if r not in ROUTED], -8))   # 2. the other 249 x s0-1
    if os.path.exists(nf): qwrite(f"e1r_{bname('RAND__' + cell)}_s01", mk("RAND__" + cell, sorted(ROWS), -7))   # 3. its RAND__ null
    else: print("no null file", nf)
    sys.exit()
if w == "loop":                                # loop reads only (75 routed x s0-1) at a chosen priority: curve point, no rest, no null
    cell, prio = sys.argv[2], int(sys.argv[3]); assert os.path.exists(f"{ED}/{cell}.json"), cell
    qwrite(f"e1r_{bname(cell)}_loop", [dict(key=f"E1|{cell}|{r}|{s}", mode="read", engine="W", priority=prio, row=r, sample=s, prefix=None,
                                             settings=FLOOR, edit={"op": "surgeon", "file": f"{ED}/{cell}.json"}) for s in (0, 1) for r in ROUTED])
    sys.exit()
if w == "export":                              # e13 line format, FULL reasoning + content, of E1|cell keys on the given rows/samples
    cell, which = sys.argv[2], sys.argv[3]       # which = loop (75 routed x s0-1) | all (324 x s0-1)
    rows = ROUTED if which == "loop" else sorted(ROWS)
    want = {f"E1|{cell}|{r}|{s}": (r, s) for r in rows for s in (0, 1)}; got = {}
    for f in glob.glob(f"{V4}/q/out/*.jsonl"):
        for l in open(f, "rb"):
            if not l.startswith(b'{"key": "E1|'): continue
            k = l[9:l.index(b'"', 9)].decode()
            if k in want and k not in got:
                x = json.loads(l)
                if not x.get("error") and x.get("content") is not None: got[k] = x
    import gzip, hashlib
    out = f"{V4}/scratch/v5_exports/e1loop_{bname(cell)}_{which}_s01.jsonl.gz"
    with gzip.open(out, "wt") as fo:
        for k in sorted(got, key=lambda k: want[k]):
            x = got[k]; r, s = want[k]
            fo.write(json.dumps(dict(arm=cell, key=k, row=r, s=s, role=ROWS[r]["v4_role"], ok=bool(x.get("ok")), pred=x.get("pred"), tokens=x.get("tokens"),
                                     prompt_tokens=x.get("prompt_tokens"), image_tokens=x.get("image_tokens"), finish=x.get("finish"), seed=x.get("seed"),
                                     edit_sha=x.get("edit_sha"), reasoning=x.get("reasoning"), content=x.get("content")), ensure_ascii=False) + "\n")
    print(f"FILE {out} lines {len(got)} of {len(want)} sha256 {hashlib.sha256(open(out, 'rb').read()).hexdigest()}")
    sys.exit()
if w == "qt":                                  # cell applied + a procedure text sent as prefix (floor base): qt CELL TAG PREFIXFILE PRIO
    cell, tag, pf, prio = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]); pre = open(pf).read().strip()
    rows = [r for r in ROUTED if ROWS[r]["bench"] == "vlmsbiased"]
    Q = [dict(key=f"E1T|{tag}|{cell}|{r}|{s}", mode="read", engine="W", priority=prio, row=r, sample=s, prefix=pre, settings=FLOOR,
              edit={"op": "surgeon", "file": f"{ED}/{cell}.json"}) for s in (0, 1) for r in rows]
    qwrite(f"e1t_{tag}_{bname(cell)}_vlmsb_s01", Q); sys.exit()
if w == "q":
    cell, prio = sys.argv[2], int(sys.argv[3]); s0, s1 = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 3)
    f = f"{ED}/{cell}.json"; assert os.path.exists(f), f
    edit = {"op": "surgeon", "file": f}
    Q = [dict(key=f"E1|{cell}|{r}|{s}", mode="read", engine="W", priority=prio, row=r, sample=s, prefix=None, settings=FLOOR, edit=edit)
         for s in range(s0, s1 + 1) for r in sorted(ROWS)]
    b = f"e1_{cell.replace('+', 'p').replace('.', '_')}_s{s0}{s1}"; p = f"{V4}/q/in/{b}.jsonl"; assert not os.path.exists(p), p
    open(p + ".tmp", "w").write("".join(json.dumps(q) + "\n" for q in Q)); os.replace(p + ".tmp", p); print("queued", b, len(Q), "prio", prio)
    sys.exit()
if w == "ts":                                  # served artefact (e.g. w2) with stock requests vs the cached VP2M512 E reads (e11) as "neither"
    arm = sys.argv[2]; S1 = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    want = {}
    for r in ROWS:
        for s in range(4): want[f"vp2m512|stock|{r}|{s}"] = ("neither", r)
        for s in range(S1 + 1): want[f"{arm}|stock|{r}|{s}"] = (arm, r)
    D = collections.defaultdict(lambda: collections.defaultdict(dict)); err = collections.Counter()
    for f in glob.glob(f"{V4}/q/out/vp_*.jsonl"):
        for l in open(f, "rb"):
            if not l.startswith(b'{"key": "'): continue
            k = l[9:l.index(b'"', 9)].decode()
            if k in want:
                x = json.loads(l); a, r = want[k]
                if x.get("error"): err[a] += 1; continue
                if x.get("content") is None or x.get("finish") == "abort": continue
                D[a][r].setdefault(k, x)
    def mean(v): return sum(v) / len(v) if v else float("nan")
    def acc(a, r): return mean([float(bool(x.get("ok"))) for x in D[a][r].values()])
    def tok(a, r, f="tokens"): return mean([x.get(f) or 0 for x in D[a][r].values()])
    def line(lbl, rr):
        rr = [r for r in rr if D[arm][r] and D["neither"][r]]
        if not rr: print(f"    {lbl:20s} no reads"); return None
        d = [acc(arm, r) - acc("neither", r) for r in rr]; m = mean(d)
        se = math.sqrt(sum((y - m) ** 2 for y in d) / (len(d) - 1) / len(d)) if len(d) > 1 else float("nan")
        print(f"    {lbl:20s} rows {len(rr):3d} | {arm} {100*mean([acc(arm, r) for r in rr]):5.1f} vs neither (VP2M512 E, s0-3) {100*mean([acc('neither', r) for r in rr]):5.1f}"
              f" | d {100*m:+5.1f}+-{100*se:4.1f} | gen {tok(arm, rr[0]) and mean([tok(arm, r) for r in rr]):6.0f} vs {mean([tok('neither', r) for r in rr]):6.0f}"
              f" | prompt {mean([tok(arm, r, 'prompt_tokens') for r in rr]):6.0f}")
        return m
    UNR = [r for r in ROWS if r not in ROUTED]
    print(f"reads: {arm} {sum(len(v) for v in D[arm].values())}, neither {sum(len(v) for v in D['neither'].values())}; errors {dict(err)}")
    print("== regions"); line("routed 75", ROUTED); line("  seen 61", [r for r in ROUTED if r not in TEST]); line("  unseen 14", [r for r in ROUTED if r in TEST])
    line("vlmsbiased routed", [r for r in ROUTED if ROWS[r]["bench"] == "vlmsbiased"]); line("unrouted 249", UNR); line("ALL 324", list(ROWS))
    print("== per benchmark (all rows)"); mac = []
    for b in sorted({ROWS[r]["bench"] for r in ROWS}):
        m = line(b, [r for r in ROWS if ROWS[r]["bench"] == b]); mac.append(m)
    print(f"    MACRO d {100*mean([m for m in mac if m is not None]):+5.2f} pp over {len(mac)} benchmarks")
    sys.exit()
if w == "tv":                                  # stacked table: base vp2m512; refs = cached VP2M512 E reads (e11, s0-3) and the W-null on this base
    SMAX = int(sys.argv[sys.argv.index("--smax") + 1]) if "--smax" in sys.argv else 1      # --smax 3: use s0-3 where read (routed s2-3 batches)
    cells = [c for i, c in enumerate(sys.argv[2:], 2) if not c.startswith("--") and sys.argv[i - 1] not in ("--smax", "--pair")]
    want = {}
    for r in ROWS:
        for s in range(4): want[f"vp2m512|stock|{r}|{s}"] = ("E_vp2m512", r)
        for s in range(SMAX + 1):
            want[f"E1V|NULL|{r}|{s}"] = ("W-null", r)
            if r in ROUTED: want[f"E1V|TEXT|{r}|{s}"] = ("text", r)
            for c in cells: want[f"E1V|{c}|{r}|{s}"] = (c, r)
    D = collections.defaultdict(lambda: collections.defaultdict(dict)); err = collections.Counter()
    for f in glob.glob(f"{V4}/q/out/vp_vp2m512*.jsonl") + glob.glob(f"{VW}/q/out/*.jsonl"):
        for l in open(f, "rb"):
            if not l.startswith(b'{"key": "'): continue
            k = l[9:l.index(b'"', 9)].decode()
            if k in want:
                x = json.loads(l); a, r = want[k]
                if x.get("error"): err[a] += 1; continue
                if x.get("content") is None or x.get("finish") == "abort": continue
                D[a][r].setdefault(k, x)
    def mean(v): return sum(v) / len(v) if v else float("nan")
    def acc(a, r): return mean([float(bool(x.get("ok"))) for x in D[a][r].values()])
    def tok(a, r): return mean([x.get("tokens") or 0 for x in D[a][r].values()])
    UNR = [r for r in ROWS if r not in ROUTED]
    REG = [("routed 75", ROUTED), ("  seen 61", [r for r in ROUTED if r not in TEST]), ("  unseen 14", [r for r in ROUTED if r in TEST]),
           ("vlmsbiased routed", [r for r in ROUTED if ROWS[r]["bench"] == "vlmsbiased"])] \
        + [(f"  routed {b}", [r for r in ROUTED if ROWS[r]["bench"] == b]) for b in sorted({ROWS[r]["bench"] for r in ROUTED} - {"vlmsbiased"})] \
        + [("unrouted 249", UNR)]
    print(f"stacked base vp2m512 | samples s0-{SMAX} | reads:", {a: sum(len(v) for v in D[a].values()) for a in ["E_vp2m512", "W-null", "text"] + cells}, "errors:", dict(err))
    for name, rows in REG:
        print(f"== {name}")
        for arm in ["E_vp2m512", "W-null"] + (["text"] if "unrouted" not in name else []) + cells:
            rr = [r for r in rows if D[arm][r] and D["E_vp2m512"][r]]
            if not rr: print(f"    {arm:24s} no reads"); continue
            def dse(ref):
                q = [r for r in rr if D[ref][r]]
                if not q: return float("nan"), float("nan"), 0
                d = [acc(arm, r) - acc(ref, r) for r in q]; m = mean(d)
                return m, (math.sqrt(sum((y - m) ** 2 for y in d) / (len(d) - 1) / len(d)) if len(d) > 1 else float("nan")), len(q)
            (de, se_e, _), (dn, se_n, nn) = dse("E_vp2m512"), dse("W-null")
            print(f"    {arm:24s} rows {len(rr):3d} reads {sum(len(D[arm][r]) for r in rr):4d} | acc {100*mean([acc(arm, r) for r in rr]):5.1f} | d vs E {100*de:+5.1f}+-{100*se_e:4.1f}"
                  f" | d vs W-null {100*dn:+5.1f}+-{100*se_n:4.1f} ({nn} rows) | gen tok {mean([tok(arm, r) for r in rr]):6.0f}")
    if "--pair" in sys.argv:                   # coordinator 08:38: cell A vs cell B on the same rows (paired by row), e.g. candidate vs PR3a64tE
        A, B = sys.argv[sys.argv.index("--pair") + 1].split(",")
        print(f"== paired {A} - {B}")
        for name, rows in REG:
            q = [r for r in rows if D[A][r] and D[B][r]]
            if len(q) < 2: print(f"    {name:20s} n/a"); continue
            d = [acc(A, r) - acc(B, r) for r in q]; m = mean(d); se = math.sqrt(sum((y - m) ** 2 for y in d) / (len(d) - 1) / len(d))
            print(f"    {name:20s} rows {len(q):3d} | d {100*m:+5.1f}+-{100*se:4.1f} ({m/se if se else float('nan'):+.1f} se)")
    if "--utok" in sys.argv:                   # unrouted accuracy and generated tokens (mean, median) by benchmark, every arm
        import statistics
        print("== unrouted by benchmark: acc | gen tok mean / median")
        for b in sorted({ROWS[r]["bench"] for r in UNR}):
            rb = [r for r in UNR if ROWS[r]["bench"] == b]
            for arm in ["E_vp2m512", "W-null"] + cells:
                rr = [r for r in rb if D[arm][r]]
                if not rr: continue
                t = [x.get("tokens") or 0 for r in rr for x in D[arm][r].values()]
                print(f"    {b:12s} {arm:28s} rows {len(rr):3d} | acc {100*mean([acc(arm, r) for r in rr]):5.1f} | tok {mean(t):6.0f} / {statistics.median(t):6.0f}")
    sys.exit()
cells = sys.argv[2:]
want = {}
for r in ROWS:
    for s in range(8):
        want[f"x|2988ed164be18b6a|{r}|{s}"] = ("FLOOR", r); want[f"WNULL|6d9695b0f6f0de8c|{r}|{s}"] = ("WNULL", r)
        if r in ROUTED: want[f"x|d5a30f27c6fe44c1|{r}|{s}"] = ("TEXT", r)
        for c in cells: want[f"E1|{c}|{r}|{s}"] = (c, r)
D = collections.defaultdict(lambda: collections.defaultdict(dict)); err = collections.Counter()
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if not l.startswith(b'{"key": "'): continue
        k = l[9:l.index(b'"', 9)].decode()
        if k in want:
            x = json.loads(l); a, r = want[k]
            if x.get("error"): err[a] += 1; continue
            if x.get("content") is None or x.get("finish") == "abort": continue
            if a == "WNULL" and x.get("prefix"): continue          # W-null of the tip: only prefix-free reads are a clean "neither"
            D[a][r].setdefault(k, x)
def mean(v): return sum(v) / len(v) if v else float("nan")
def acc(a, r): return mean([float(bool(x.get("ok"))) for x in D[a][r].values()])
def tok(a, r): return mean([x.get("tokens") or 0 for x in D[a][r].values()])
UNR = [r for r in ROWS if r not in ROUTED]
REG = [("routed 75", ROUTED), ("  seen 61", [r for r in ROUTED if r not in TEST]), ("  unseen 14", [r for r in ROUTED if r in TEST]),
       ("vlmsbiased routed", [r for r in ROUTED if ROWS[r]["bench"] == "vlmsbiased"]), ("unrouted 249", UNR),
       ("  unrouted fit 231", [r for r in UNR if r not in UHELD]), ("  unrouted held-out 18", [r for r in UNR if r in UHELD])]
def pred(arm, reg):
    if arm in ("FLOOR", "WNULL", "TEXT"): return ""
    if reg.startswith(("unrouted", "  unrouted")): return "pred +-1"
    if arm.startswith(("RAND__", "ROT__")): return "pred +-1.5"
    return "pred +3.4..+5.0" if arm.startswith("PRa64") else "pred +2.5..+3.8"
print("reads:", {a: sum(len(v) for v in D[a].values()) for a in ["FLOOR", "WNULL", "TEXT"] + cells}, "not-fit / errors:", dict(err))
for name, rows in REG:
    print(f"== {name}")
    for arm in ["FLOOR", "WNULL"] + cells + (["TEXT"] if "unrouted" not in name else []):
        rr = [r for r in rows if D[arm][r] and D["FLOOR"][r]]
        if not rr: print(f"    {arm:24s} no reads"); continue
        d = [acc(arm, r) - acc("FLOOR", r) for r in rr]; m = mean(d)
        se = math.sqrt(sum((y - m) ** 2 for y in d) / (len(d) - 1) / len(d)) if len(d) > 1 else float("nan")
        print(f"    {arm:24s} rows {len(rr):3d} reads {sum(len(D[arm][r]) for r in rr):4d} | acc {100*mean([acc(arm, r) for r in rr]):5.1f} | d vs floor {100*m:+5.1f}+-{100*se:4.1f}"
              f" | gen tok {mean([tok(arm, r) for r in rr]):6.0f} | {pred(arm, name)}")
