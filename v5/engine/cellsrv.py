"""v4 cell server: one W card read weight cells as exported checkpoints on `vllm serve` (same engine type as the parent cache).
Loop: claim the most urgent pending W/T group (batch, edit_sha) that no up T server serves (same claim files as w_worker.py, owner = --name)
-> export stock + edit to V4/ckpt/cell_<edit_sha>/ (export_tip.py --n 0 in a subprocess; reused if it exists) -> `vllm serve` it on this card
(serve.sh, port 8200+card, served name cell_<edit_sha>) -> engines.json entry of this card = T server for that edit_sha (state up) -> the
dispatcher routes the group's reads there (W workers skip T-served edit_shas) -> when every request of that edit_sha in the claimed batches
has a q/out line, or a cancel (q/cancel/<batch> or <batch>@<edit_sha>) exists: state drain, stop the server (a cancel stops at once:
in-flight reads are requeued by the dispatcher and the W worker writes their cancel lines) -> state released -> next group.
Requests of the same edit_sha in other batches that arrive while the server is up are claimed too and read by it.
Every checkpoint is recorded in V4/ckpt/cells.list (dir, edit_sha, bytes new on disk, time). Nothing is deleted.
usage: python cellsrv.py --name C6   (inside `gpu bg --gpus 6`; CUDA_VISIBLE_DEVICES selects the card)"""
import argparse, glob, hashlib, json, os, signal, subprocess, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--v4", default="/data/fd/mingdazhang/selfedit/v4")
ap.add_argument("--name", required=True)
ap.add_argument("--card", type=int, default=int(os.environ.get("CARD", "-1")))
ap.add_argument("--cap", type=int, default=40)
ap.add_argument("--only", default="", help="take only batches whose name starts with this (tests)")
args = ap.parse_args()
V4 = args.v4
ED = os.path.dirname(os.path.abspath(__file__))
QIN, QOUT, QCAN = f"{V4}/q/in", f"{V4}/q/out", f"{V4}/q/cancel"
PORT = 8200 + args.card
canon = lambda o: json.dumps(o, sort_keys=True, separators=(",", ":"))
esha = lambda e: hashlib.sha256(canon(e).encode()).hexdigest()[:16]
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{args.name}]", *a, flush=True)
PY = sys.executable


def eng_update(**kw):
    e = json.load(open(f"{V4}/engines.json"))
    for x in e["servers"]:
        if x["port"] == PORT:
            x.update(kw)
    e["updated"] = time.time()
    tmp = f"{V4}/engines.json.{args.name}.tmp"
    json.dump(e, open(tmp, "w"), indent=1)
    os.replace(tmp, f"{V4}/engines.json")


def t_served():
    e = json.load(open(f"{V4}/engines.json"))
    return {x["edit_sha"] for x in e["servers"] if x.get("role") == "T" and x.get("edit_sha") and x.get("state", "up") == "up" and x["port"] != PORT}


def claim(batch, es):
    p = f"{V4}/w/claims/{batch}.{es}"
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, args.name.encode())
        os.close(fd)
        return True
    except FileExistsError:
        return open(p).read().strip() == args.name


def out_keys(batch):
    have = set()
    if os.path.exists(f"{QOUT}/{batch}.jsonl"):
        for l in open(f"{QOUT}/{batch}.jsonl"):
            try:
                have.add(json.loads(l)["key"])
            except ValueError:
                pass
    return have


def pending():
    """{edit_sha: dict(edit, prio, mtime, batches: {batch: [keys]})} for W/T requests without a line, not T-served elsewhere, not cancelled."""
    ts, G = t_served(), {}
    for f in glob.glob(f"{QIN}/*.jsonl"):
        b = os.path.basename(f)[:-6]
        if not b.startswith(args.only) or os.path.exists(f"{QOUT}/{b}.done") or os.path.exists(f"{QCAN}/{b}"):
            continue
        reqs = [json.loads(l) for l in open(f) if l.strip()]
        if not reqs or not all(r.get("engine") in ("W", "T") for r in reqs):
            continue
        have = None
        for r in reqs:
            if r.get("edit") is None or r.get("pin") == "W":
                continue                      # stock-weight reads: not a cell (W worker / stock servers)
            h = r.get("edit_sha") or esha(r.get("edit"))
            if h in ts or os.path.exists(f"{QCAN}/{b}@{h}"):
                continue
            if have is None:
                have = out_keys(b)
            if r["key"] in have:
                continue
            g = G.setdefault(h, dict(edit=r.get("edit"), prio=int(r.get("priority", 5)), mtime=os.path.getmtime(f), batches={}))
            g["prio"] = min(g["prio"], int(r.get("priority", 5)))
            g["batches"].setdefault(b, []).append(r["key"])
    return G


def export(h, edit):
    d = f"{V4}/ckpt/cell_{h}"
    if os.path.exists(f"{d}.export.json"):
        return d
    spec = f"{V4}/ckpt/cell_{h}.edits.json"
    json.dump(dict(edits=edit), open(spec, "w"))
    t = time.time()
    r = subprocess.run([PY, "-u", f"{ED}/export_tip.py", f"cell_{h}", spec, "--n", "0"], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(f"{d}.export.json"):
        raise RuntimeError(f"export failed: {(r.stderr or r.stdout)[-600:]}")
    new = sum(os.path.getsize(p) for p in glob.glob(f"{d}/*") if os.path.isfile(p) and os.stat(p).st_nlink == 1)
    with open(f"{V4}/ckpt/cells.list", "a") as f:
        f.write(json.dumps(dict(dir=d, edit_sha=h, new_bytes=new, t=time.time(), export_s=round(time.time() - t, 1), by=args.name)) + "\n")
    log(f"exported {d} in {time.time() - t:.0f}s, {new / 1e9:.1f} GB new")
    return d


def serve(d, h):
    env = dict(os.environ, MODEL_DIR=d, SERVED=f"cell_{h}")
    p = subprocess.Popen(["bash", f"{ED}/serve.sh", str(PORT), "96", "--gpu-memory-utilization", "0.92"], env=env,
                         stdout=open(f"{V4}/w/{args.name}.serve.log", "a"), stderr=subprocess.STDOUT, start_new_session=True)
    t = time.time()
    while True:
        if p.poll() is not None:
            raise RuntimeError(f"vllm serve exited {p.returncode} (log {V4}/w/{args.name}.serve.log)")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5).read()
            break
        except Exception:
            time.sleep(5)
    log(f"serving cell_{h} on port {PORT} after {time.time() - t:.0f}s")
    return p


def stop(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(120)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
        p.wait()


def group_state(h, batches):
    """-> (left, cancelled): unanswered keys of edit_sha h in the claimed batches; cancelled if every claimed batch has a cancel."""
    left, can = 0, True
    for b in list(batches):
        c = os.path.exists(f"{QCAN}/{b}") or os.path.exists(f"{QCAN}/{b}@{h}")
        can = can and c
        if c:
            continue
        have = out_keys(b)
        left += sum(1 for k in batches[b] if k not in have)
    return left, can


os.makedirs(f"{V4}/w/claims", exist_ok=True)
NAME = next(x.get("name", f"127.0.0.1:{PORT}") for x in json.load(open(f"{V4}/engines.json"))["servers"] if x["port"] == PORT)
eng_update(role="C", state="released", edit_sha=None, model=None, weights=None, cap=args.cap, job=os.environ.get("GPU_JOB_NAME"), note=f"cell server {args.name}")
log(f"cell server up, card {args.card}, port {PORT}")
json.dump(dict(name=args.name, card=args.card, t=time.time(), pid=os.getpid()), open(f"{V4}/w/{args.name}.up.json", "w"))
while True:
    G = pending()
    pick = None
    for h, g in sorted(G.items(), key=lambda kv: (kv[1]["prio"], kv[1]["mtime"])):
        mine = {b: ks for b, ks in g["batches"].items() if claim(b, h)}
        if mine:
            pick = (h, g, mine)
            break
    if not pick:
        time.sleep(5)
        continue
    h, g, batches = pick
    log(f"cell {h}: {sum(len(v) for v in batches.values())} reads in {len(batches)} batches; edit {canon(g['edit'])[:120]}")
    p = None
    try:
        t0 = time.time()
        d = export(h, g["edit"])
        p = serve(d, h)
        eng_update(role="T", state="up", edit_sha=h, model=f"cell_{h}", weights=f"cell_{h}", cap=args.cap, started=time.time())
        while True:
            time.sleep(10)
            g2 = pending().get(h)                      # same edit_sha arriving in new batches while up: claim and read them too
            for b, ks in (g2 or {}).get("batches", {}).items():
                if b not in batches and claim(b, h):
                    batches[b] = ks
                    log(f"cell {h}: also reading {len(ks)} reads of {b}")
            left, can = group_state(h, batches)
            if can:
                log(f"cell {h}: cancelled, stopping at once")
                break
            if left == 0:
                break
        eng_update(state="drain")
        if not can:
            t = time.time()
            while time.time() - t < 600:
                try:
                    if json.load(open(f"{V4}/q/drained/{NAME}.json"))["t"] > t - 5:
                        break
                except Exception:
                    pass
                time.sleep(5)
        log(f"cell {h}: done in {time.time() - t0:.0f}s")
    except Exception as ex:
        log(f"cell {h} FAILED: {ex!r}; left to the offline W worker")
        with open(f"{V4}/w/{args.name}.failed", "a") as f:
            f.write(json.dumps(dict(edit_sha=h, error=repr(ex)[:500], t=time.time())) + "\n")
        for b in batches:                               # the offline W worker may now take these claims
            open(f"{V4}/w/claims/{b}.{h}.failed", "w").write(args.name)
    finally:
        eng_update(role="C", state="released", edit_sha=None, model=None, weights=None)
        if p is not None:
            stop(p)
