"""Held-out REPORT read of the weights system (coordinator 11-12 04:27, after the ACCEPT of VP2M512): VP2M512 weights, STOCK requests
(no image settings, effort medium), id_test_768 + ood128_v2, sample-major (all rows of sample s, both sets, before sample s+1).
Request = dispatch._build read mode exactly: common.content_for(prompt, image_paths), SAMPLING, max_tokens 32768, CRN seed
crc32("101|0|row") % (2^31-1024) + s (the Venus held-out convention, same dispatcher code). Gold of babyvision/6557 = "=" (as on Venus).
Keys hv|vp2m512|row|s; results in /data/fd/mingdazhang/selfedit/v5/heldout/reads/hv_vp2m512.jsonl only (no learn-row cache is touched).
usage: hv_read.py run S0 S1 [--per N]     resumable (skips keys already in the results file)
       hv_read.py table S0 S1              macro vs the cached stock mean (compact Venus file)
       hv_read.py export NAME              compact file (fields of heldout_base_reads_compact) under v5/heldout/reads/"""
import asyncio, glob, gzip, hashlib, json, os, sys, time, zlib, collections, statistics
sys.path[:0] = ["/workspace/fudan/mingdazhang/selfedit-v4/engine_v4"]
from common import SAMPLING, EFFORT, content_for, grade
H = "/data/fd/mingdazhang/selfedit/v5/heldout"; OUT = f"{H}/reads"; os.makedirs(OUT, exist_ok=True); MODEL = os.environ.get("HV_MODEL", "vp2m512"); RES = f"{OUT}/hv_{MODEL}.jsonl"
SERVERS = [f"127.0.0.1:{p}" for p in os.environ.get("HV_PORTS", "8200,8201,8202").split(",")]
ROWS, SET = {}, {}
for name, f in (("id", "id_test_768.jsonl"), ("ood", "ood128_v2.jsonl")):
    for l in open(f"{H}/{f}"):
        r = json.loads(l)
        if r["id"] == "babyvision/6557": r["answer"] = "="
        ROWS[r["id"]] = r; SET[r["id"]] = name
assert sum(v == "id" for v in SET.values()) == 768 and sum(v == "ood" for v in SET.values()) == 768, collections.Counter(SET.values())
def seed(row, s): return zlib.crc32(f"101|0|{row}".encode()) % (2 ** 31 - 1024) + int(s)
def done_keys():
    k = set()
    if os.path.exists(RES):
        for l in open(RES):
            try: x = json.loads(l)
            except Exception: continue
            if not x.get("error"): k.add(x["key"])
    return k
async def one(sess, srv, row, s, fo):
    r = ROWS[row]; content, nt = content_for(r["prompt"], r.get("image_paths") or [])
    body = dict(messages=[{"role": "user", "content": content}], max_tokens=32768, model=MODEL, seed=seed(row, s),
                chat_template_kwargs={"reasoning_effort": EFFORT, "enable_thinking": True}, cache_salt=os.urandom(16).hex(), **SAMPLING)
    rec = dict(key=f"hv|{MODEL}|{row}|{s}", set=SET[row], bench=r["bench"], row=row, sample=s, system=MODEL, server=srv, seed=body["seed"], image_tokens=nt)
    t0 = time.time()
    for k in range(4):
        try:
            async with sess.post(f"http://{srv}/v1/chat/completions", json=body) as resp:
                txt = await resp.text()
                if resp.status != 200: raise RuntimeError(f"HTTP {resp.status}: {txt[:300]}")
                o = json.loads(txt); break
        except Exception as e:
            err = repr(e)[:300]
            if "HTTP 400" in err: rec.update(error=err); fo.write(json.dumps(rec) + "\n"); fo.flush(); return
            await asyncio.sleep(5 * (k + 1))
    else:
        rec.update(error=err); fo.write(json.dumps(rec) + "\n"); fo.flush(); return
    c = o["choices"][0]; m = c["message"]
    cont = (m.get("content") or "").strip(); reas = (m.get("reasoning_content") or m.get("reasoning") or "").strip()
    toks, ptoks, fin = o["usage"]["completion_tokens"], o["usage"]["prompt_tokens"], c.get("finish_reason")
    ok, loop, gerr, pred = grade(r, cont, reas, fin, toks, s)
    rec.update(ok=ok, pred=pred, loop=loop, grade_error=gerr, tokens=toks, prompt_tokens=ptoks, finish=fin, content=cont, reasoning=reas,
               answer=r["answer"], t=round(time.time() - t0, 1))
    fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
async def run(s0, s1, per):
    import aiohttp
    have = done_keys()
    todo = [(row, s) for s in range(s0, s1 + 1) for row in sorted(ROWS, key=lambda x: (SET[x], x)) if f"hv|{MODEL}|{row}|{s}" not in have]
    if os.environ.get("HV_LONGEST_FIRST"):      # coordinator 09:57: longest rows first (mean generated tokens of this model's earlier samples) -> short tail
        tk = collections.defaultdict(list)
        for x in load(0, 99): tk[x["row"]].append(x.get("tokens") or 0)
        todo.sort(key=lambda t: (t[1], -(sum(tk[t[0]]) / len(tk[t[0]]) if tk[t[0]] else 1e9)))
    print(f"{time.strftime('%H:%M:%S')} todo {len(todo)} (done {len(have)})", flush=True)
    q = asyncio.Queue()
    for t in todo: q.put_nowait(t)
    n = [0]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200)) as sess:
        with open(RES, "a") as fo:
            async def worker(srv):
                while not q.empty():
                    row, s = q.get_nowait(); await one(sess, srv, row, s, fo); n[0] += 1
                    if n[0] % 200 == 0: print(f"{time.strftime('%H:%M:%S')} {n[0]}/{len(todo)} (now sample {s})", flush=True)
            await asyncio.gather(*[worker(srv) for srv in SERVERS for _ in range(per)])
    print(f"{time.strftime('%H:%M:%S')} finished {n[0]}", flush=True)
def load(s0, s1):
    R = {}
    for l in open(RES):
        try: x = json.loads(l)
        except Exception: continue
        if not x.get("error") and s0 <= x["sample"] <= s1: R[x["key"]] = x
    return list(R.values())
w = sys.argv[1]
if w == "run":
    per = int(sys.argv[sys.argv.index("--per") + 1]) if "--per" in sys.argv else 32
    asyncio.run(run(int(sys.argv[2]), int(sys.argv[3]), per))
elif w == "table":
    s0, s1 = int(sys.argv[2]), int(sys.argv[3]); L = load(s0, s1)
    base = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in gzip.open(f"{H}/heldout_base_reads_compact.jsonl.gz", "rt"):
        x = json.loads(l); base[x["system"]][x["row"]].append(float(x["ok"]))
    me = collections.defaultdict(list)
    for x in L: me[x["row"]].append(float(bool(x["ok"])))
    if MODEL != "vp2m512" and os.path.exists(f"{OUT}/hv_vp2m512.jsonl"):
        for l in open(f"{OUT}/hv_vp2m512.jsonl"):
            try: x = json.loads(l)
            except Exception: continue
            if not x.get("error") and x["sample"] <= 2: base["vp2m512"][x["row"]].append(float(bool(x["ok"])))
    err = sum(1 for l in open(RES) if '"error"' in l)
    print(f"samples {s0}-{s1}: reads {len(L)}, error lines {err}, length {sum(x['finish'] == 'length' for x in L)}")
    for st in ("id", "ood"):
        rows = [r for r in me if SET[r] == st]; benches = sorted({ROWS[r]["bench"] for r in rows}); line = []
        mac = {k: [] for k in ("vp", "stock", "floor", "tip", "vp2m512")}
        for b in benches:
            rb = [r for r in rows if ROWS[r]["bench"] == b]
            v = sum(sum(me[r]) / len(me[r]) for r in rb) / len(rb); mac["vp"].append(v)
            for sy in ("stock", "floor", "tip", "vp2m512"):
                rr = [r for r in rb if base[sy][r]]
                if rr: mac[sy].append(sum(sum(base[sy][r]) / len(base[sy][r]) for r in rr) / len(rr))
            sb = set(rb); xb = [x for x in L if x["row"] in sb]
            tk = f" gen {sum(x['tokens'] or 0 for x in xb)/max(1,len(xb)):.0f} med {statistics.median([x['tokens'] or 0 for x in xb]) if xb else 0:.0f} tot {sum((x['tokens'] or 0)+(x['prompt_tokens'] or 0) for x in xb)/max(1,len(xb)):.0f}"
            line.append(f"{b} {100*v:.1f}{tk} (stock {100*mac['stock'][-1]:.1f}" + (f", vp2m512 {100*mac['vp2m512'][-1]:.1f})" if len(mac["vp2m512"]) == len(mac["vp"]) else ")"))
        f = lambda k: 100 * sum(mac[k]) / len(mac[k]) if mac[k] else float("nan")
        print(f"  {st.upper()} rows {len(rows)} reads {sum(len(me[r]) for r in rows)} | MACRO {MODEL} {f('vp'):.2f} vs stock {f('stock'):.2f} (d {f('vp')-f('stock'):+.2f}) | floor {f('floor'):.2f} tip {f('tip'):.2f}"
              + (f" | vp2m512 (s0-2) {f('vp2m512'):.2f} (d {f('vp')-f('vp2m512'):+.2f})" if mac["vp2m512"] else ""))
        print("    " + "; ".join(line))
elif w == "table3":                             # s0-2: one read (mean of 3) and "2 reads + third on disagreement", vs cached stock / floor / tip
    def vote(recs):                              # recs: {sample: (pred, ok)} for s0, s1, s2
        if not all(k in recs for k in (0, 1, 2)): return None
        p0, p1, p2 = recs[0][0], recs[1][0], recs[2][0]
        if p0 is not None and p0 == p1: return recs[0][1]
        for a in (p0, p1):
            if a is not None and a == p2: return recs[2][1] if a == p2 else None
        return recs[2][1]
    me = collections.defaultdict(dict)
    for x in load(0, 2): me[x["row"]][x["sample"]] = (x["pred"], bool(x["ok"]))
    base = collections.defaultdict(lambda: collections.defaultdict(dict))
    for l in gzip.open(f"{H}/heldout_base_reads_compact.jsonl.gz", "rt"):
        x = json.loads(l); base[x["system"]][x["row"]][x["sample"]] = (x["pred"], bool(x["ok"]))
    systems = [(MODEL, me)] + [(k, base[k]) for k in ("stock", "floor", "tip")]
    def one(D, r, maxs): v = [D[r][s][1] for s in D[r] if s < maxs]; return sum(v) / len(v) if v else None
    for st in ("id", "ood"):
        rows = sorted(r for r in ROWS if SET[r] == st and len(me[r]) == 3); benches = sorted({ROWS[r]["bench"] for r in rows})
        print(f"== {st.upper()} ({len(rows)} rows with s0-2)   one read = mean over the system's samples (vp: 3; stock 8, floor 5, tip 5) | 2+1 = votes from s0-2")
        res = {}
        for name, D in systems:
            o, v = [], []
            for b in benches:
                rb = [r for r in rows if ROWS[r]["bench"] == b and D.get(r)]
                o.append(sum(one(D, r, 99) for r in rb) / len(rb))
                vv = [vote(D[r]) for r in rb]; vv = [float(x) for x in vv if x is not None]; v.append(sum(vv) / len(vv) if vv else float("nan"))
            res[name] = (o, v)
        print("    " + f"{'system':10s} {'macro1':>7s} {'macro2+1':>8s} | " + " ".join(f"{b[:10]:>10s}" for b in benches) + "   (per bench: one read / 2+1)")
        for name, (o, v) in res.items():
            print(f"    {name:10s} {100*sum(o)/len(o):7.2f} {100*sum(v)/len(v):8.2f} | " + " ".join(f"{100*a:4.1f}/{100*c:4.1f}" for a, c in zip(o, v)))
elif w == "export":
    name = sys.argv[2]; pic = {}
    for l in gzip.open(f"{H}/heldout_base_reads_compact.jsonl.gz", "rt"):
        x = json.loads(l); pic.setdefault(x["row"], x.get("pic"))
    ss = (int(sys.argv[3]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 99)      # export NAME [S]: one sample only
    L = sorted(load(*ss), key=lambda x: (x["set"], x["row"], x["sample"]))
    p = f"{OUT}/{name}"
    with gzip.open(p, "wt") as f:
        for x in L:
            f.write(json.dumps(dict(set=x["set"], bench=x["bench"], row=x["row"], pic=pic.get(x["row"]), system=MODEL, tip=f"{MODEL}_stock_req",
                                    sample=x["sample"], pred=x["pred"], ok=bool(x["ok"]), tokens=x["tokens"], prompt_tokens=x["prompt_tokens"],
                                    finish=x["finish"], key=x["key"]), ensure_ascii=False) + "\n")
    print(f"FILE {p} {os.path.getsize(p)} bytes sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()} lines {len(L)}",
          dict(collections.Counter((x["set"], x["sample"]) for x in L)))
