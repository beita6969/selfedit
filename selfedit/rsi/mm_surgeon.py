"""The model as its own surgeon. It reads its own exam and its own failures, sees its own checkpoint (every number in its config files, every tensor) and writes a patch in three low-level operations.
Nothing in the prompt names a fix, and no result of any change is shown except its own earlier patches. A patch that cannot apply goes back to the model with the error, once.
The control is blind patches in the same operations and the same mix, at the same number of reads. Both are judged by rsi/mm_judge.py, which the model never sees.

Usage: python -u rsi/mm_surgeon.py propose WORK --exam DIR:LABEL --digest DIAG.md [--samples 64] [--think-tokens 8000] [--ledger LEDGER.jsonl] [loop.py args]    writes WORK/proposals.jsonl, WORK/patches.json
       python rsi/mm_surgeon.py blind WORK --model M [--n 16] [--seed 0]                                                                                        writes WORK/blind.json
       python rsi/mm_surgeon.py judge WORK --family F --base M --exam DIR --cards 1,5 [--top 16] [--tag sg]                                                     writes WORK/judge/*.jsonl, WORK/report.md, WORK/ledger.jsonl
"""
import argparse, collections, fnmatch, glob, json, os, random, re, shutil, struct, sys, threading

CONFIGS = ("config.json", "preprocessor_config.json", "processor_config.json")
TASK_TEXT = {"ring": "a black ring with a small gap, which way the gap faces; level = pixels across the ring", "square": "the colour of one small square; level = its side in pixels", "gap": "whether a line is broken; level = the gap in pixels",
             "cell": "the colour of the cell named by row and column in a grid; level = rows and columns", "odd": "where the one circle with a tail is in a grid; level = rows and columns", "dots": "how many dots; level = dot radius in pixels",
             "trace": "which end a line reaches among lines that cross; level = number of lines"}
OPS = """- {"op": "scale", "tensor": NAME, "alpha": X}: multiply a tensor by X. NAME may hold * as a wildcard. Optional "dim": D and "range": [A, B] scale only that slice.
- {"op": "resample", "tensor": NAME, "size": [H, W], "mode": "area" | "bilinear" | "bicubic" | "nearest", "alpha": X}: resize the last two dimensions of a tensor to H x W and multiply it by X (default 1).
- {"op": "set", "file": FILE, "key": DOTTED.KEY, "value": V}: set a number in a config file. The key must already exist."""
PROMPT = """You are an open-weight vision-language model. You can change your own checkpoint: no training, and no change to any prompt or sampling setting. The changed checkpoint then runs on the same inference engine.

Your exam. Drawn images on a 384 x 384 canvas with known answers, one forward pass each, no reasoning. Accuracy per level, four options unless the task says otherwise:
{exam}

How you failed on real visual problems, in your own words after reading your own failed attempts:
{digest}

Your checkpoint. Every number in your config files:
{config}

Your tensors (name, shape, count; {{i}} stands for a block or layer index):
{tensors}

Operations you can use:
{ops}

Earlier patches of yours and what they did:
{ledger}

Work out what in your network causes the weakest results above. Then write the one patch most likely to raise your accuracy on visual problems the most. It has to load and run as it stands, so everything that depends on a value you change must change with it. End with one line of JSON: {{"why": "<under 30 words>", "ops": [<operations>]}}"""
RETRY = """Applying that patch failed: {error}
Write the corrected patch. End with one line of JSON: {{"why": "<under 30 words>", "ops": [<operations>]}}"""


def pop(argv, flag, default=None):
    if flag not in argv:
        return default
    i = argv.index(flag)
    val = argv[i + 1]
    del argv[i:i + 2]
    return val


def last_json(text, key):
    dec = json.JSONDecoder()
    for i in reversed([m.start() for m in re.finditer(r"\{", text or "")]):
        try:
            obj, _ = dec.raw_decode(text, i)
        except ValueError:
            continue
        if isinstance(obj, dict) and key in obj:
            return obj
    return None


def flat(o, prefix=""):
    """Every number in a config, with its dotted key."""
    out = {}
    for k, v in o.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and not k.endswith("token_id"):
            out[key] = v
    return out


def config_numbers(model):
    return {f: flat(json.load(open(os.path.join(model, f)))) for f in CONFIGS if os.path.exists(os.path.join(model, f))}


def shapes(model):
    """Tensor name to shape, from the shard headers, without loading anything."""
    out = {}
    for path in sorted(glob.glob(os.path.join(model, "*.safetensors"))):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
        out.update({k: v["shape"] for k, v in head.items() if k != "__metadata__" and not k.startswith("mtp.")})   # mtp: multi-token prediction, the engine does not run it
    return out


def groups(model):
    """Tensors with their block or layer index taken out: (pattern with {i}, shape) -> indices."""
    g = collections.defaultdict(list)
    for name, shape in shapes(model).items():
        m = re.search(r"\.(\d+)\.", name)
        g[(name[: m.start(1)] + "{i}" + name[m.end(1):], tuple(shape)) if m else (name, tuple(shape))].append(int(m.group(1)) if m else -1)
    return g


def inventory(model):
    cfg = "\n".join(f"{f}: " + ", ".join(f"{k}={v}" for k, v in nums.items()) for f, nums in config_numbers(model).items())
    lines = []
    for (name, shape), ix in sorted(groups(model).items()):
        ix = sorted(ix)
        which = "" if ix == [-1] else f" i in {ix[0]}..{ix[-1]}" if ix == list(range(ix[0], ix[-1] + 1)) else " i in " + ",".join(map(str, ix[:6])) + (",.." if len(ix) > 6 else "")
        lines.append(f"{name} {list(shape)}{which}")
    return cfg, "\n".join(lines)


def exam_table(path, label):
    row = next(r for r in map(json.loads, open(os.path.join(path, "summary.jsonl"))) if r["label"] == label)
    by = collections.defaultdict(dict)
    for k, v in row["acc"].items():
        t, l = k.split("/")
        by[t][int(l)] = v
    return "\n".join(f"- {t}: {TASK_TEXT.get(t, t)}: " + " ".join(f"{l}:{v:.2f}" for l, v in sorted(by[t].items())) for t in by)


def check(ops, model):
    """What is wrong with a patch, or None."""
    if not isinstance(ops, list) or not ops or len(ops) > 16:
        return "ops must be a list of 1 to 16 operations"
    names, nums = shapes(model), config_numbers(model)
    for o in ops:
        kind = o.get("op") if isinstance(o, dict) else None
        if kind in ("scale", "resample"):
            hit = fnmatch.filter(names, str(o.get("tensor")))
            if not hit:
                return f"no tensor matches {o.get('tensor')}"
            if not isinstance(o.get("alpha", 1), (int, float)):
                return "alpha must be a number"
            if kind == "resample":
                size = o.get("size")
                if not (isinstance(size, list) and len(size) == 2 and all(isinstance(x, int) and x > 0 for x in size)) or o.get("mode") not in ("area", "bilinear", "bicubic", "nearest"):
                    return "resample needs size [H, W] and a mode"
                if any(len(names[h]) < 3 for h in hit):
                    return f"{o.get('tensor')} has too few dimensions to resample"
            elif "range" in o and not (isinstance(o["range"], list) and len(o["range"]) == 2 and isinstance(o.get("dim"), int)):
                return "a slice needs dim and range [A, B]"
        elif kind == "set":
            if o.get("file") not in nums:
                return f"no config file {o.get('file')}"
            if o.get("key") not in nums[o["file"]]:
                return f"no number at {o.get('key')} in {o['file']}"
            if not isinstance(o.get("value"), (int, float)) or isinstance(o.get("value"), bool):
                return "value must be a number"
        else:
            return f"unknown operation {kind}"
    return None


def apply_patch(src, dst, ops):
    """A new checkpoint: changed shards and config files written, everything else linked."""
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    os.makedirs(dst)
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    touched = {}
    for o in ops:
        if o["op"] != "set":
            for name in fnmatch.filter(index, o["tensor"]):
                touched.setdefault(index[name], set()).add(name)
    edited = set()
    for shard, names in touched.items():
        tensors = load_file(os.path.join(src, shard))
        for o in ops:
            if o["op"] == "set":
                continue
            for name in fnmatch.filter(names, o["tensor"]):
                t = tensors[name]
                x = t.float()
                if o["op"] == "scale":
                    if "range" in o:
                        cut = [slice(None)] * x.dim()
                        cut[o["dim"]] = slice(*o["range"])
                        x[tuple(cut)] *= o["alpha"]
                    else:
                        x = x * o["alpha"]
                else:
                    head = x.shape[:-2]
                    x = F.interpolate(x.reshape(-1, 1, *x.shape[-2:]), size=o["size"], mode=o["mode"], **({"align_corners": False} if o["mode"] in ("bilinear", "bicubic") else {}))
                    x = x.reshape(*head, *o["size"]) * o.get("alpha", 1)
                tensors[name] = x.to(t.dtype)
        save_file(tensors, os.path.join(dst, shard), metadata={"format": "pt"})
        edited.add(shard)
    files = {}
    for o in ops:
        if o["op"] == "set":
            cfg = files.setdefault(o["file"], json.load(open(os.path.join(src, o["file"]))))
            *path, last = o["key"].split(".")
            for k in path:
                cfg = cfg[k]
            cfg[last] = o["value"]
    for name, cfg in files.items():
        json.dump(cfg, open(os.path.join(dst, name), "w"), indent=2)
    for name in os.listdir(src):
        if name not in edited and name not in files:
            os.symlink(os.path.join(src, name), os.path.join(dst, name))


def canon(ops):
    def num(v):
        return round(v, 4) if isinstance(v, float) else v
    return json.dumps([{k: num(v) for k, v in sorted(o.items())} for o in ops], sort_keys=True)


def propose(work, argv):
    exam = pop(argv, "--exam")
    digest = open(pop(argv, "--digest")).read()
    samples = int(pop(argv, "--samples", "64"))
    think = int(pop(argv, "--think-tokens", "8000"))
    ledger = pop(argv, "--ledger")
    os.makedirs(work, exist_ok=True)
    sys.argv = ["loop.py", "--run-dir", work] + argv
    CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
    import rsi.loop as L
    from surgery import Surgeon

    a = L.args
    S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens,
                gpu_memory_utilization=a.gpu_util, kv_cache_dtype=a.kv_cache_dtype)
    cfg, tensors = inventory(a.model)
    past = "none yet"
    if ledger and os.path.exists(ledger):
        rows = [json.loads(l) for l in open(ledger)]
        past = "\n".join(f"- {json.dumps(r['ops'])}: " + (f"did not load: {r['error'][:160]}" if r.get("error") else f"exam margin {r['exam']}, real problems margin {r['pool']}") for r in rows) or past
    d, t = exam.rsplit(":", 1)
    text = PROMPT.format(exam=exam_table(d, t), digest=digest, config=cfg, tensors=tensors, ops=OPS, ledger=past)
    L.say(f"mm_surgeon: prompt {len(S.tokenizer(text).input_ids)} tokens")
    with open(os.path.join(work, "prompt.txt"), "w") as f:
        f.write(text)
    first = [{"role": "user", "content": text}]

    def ask(chats):
        return [o["samples"][0] for o in S.generate(chats, n=1, sampling=dict(L.SAMPLING), max_tokens=think, reasoning_effort=a.effort)]

    rows = []
    for s in ask([first] * samples):
        obj = last_json(s["content"], "ops") or last_json(s["reasoning"], "ops")
        rows.append({"content": s["content"] or "", "reasoning": (s["reasoning"] or "")[-3000:], "finish": s["finish_reason"], "tokens": s["num_tokens"], "obj": obj, "error": check((obj or {}).get("ops"), a.model) if obj else "no JSON patch in the reply"})
    bad = [r for r in rows if r["error"] and r["obj"]]
    L.say(f"mm_surgeon: {samples} samples, {sum(not r['error'] for r in rows)} apply, {len(bad)} sent back, {sum(not r['obj'] for r in rows)} with no patch")
    if bad:
        chats = [first + [{"role": "assistant", "content": r["content"] or json.dumps(r["obj"])}, {"role": "user", "content": RETRY.format(error=r["error"])}] for r in bad]
        for r, s in zip(bad, ask(chats)):
            obj = last_json(s["content"], "ops") or last_json(s["reasoning"], "ops")
            r.update(retried=True, first=r["obj"], first_error=r["error"], obj=obj, error=check((obj or {}).get("ops"), a.model) if obj else "no JSON patch in the reply", content=s["content"] or "")
    with open(os.path.join(work, "proposals.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    count, why, ops_of = collections.Counter(), collections.defaultdict(list), {}
    for r in rows:
        if not r["error"]:
            k = canon(r["obj"]["ops"])
            count[k] += 1
            ops_of[k] = r["obj"]["ops"]
            why[k].append(str(r["obj"].get("why", ""))[:200])
    patches = [{"id": f"m{i:02d}", "count": n, "ops": ops_of[k], "why": why[k][:3]} for i, (k, n) in enumerate(count.most_common())]
    json.dump(patches, open(os.path.join(work, "patches.json"), "w"), indent=1, ensure_ascii=False)
    L.say(f"mm_surgeon: {len(patches)} different patches from {sum(count.values())} that apply")
    for p in patches[:12]:
        L.say(f"  {p['id']} x{p['count']}: {json.dumps(p['ops'])[:300]}")


def blind(work, argv):
    """The same operations in the same mix, with nothing known about what they should do."""
    model = pop(argv, "--model")
    made = json.load(open(os.path.join(work, "patches.json")))
    n = int(pop(argv, "--n", str(min(16, len(made)))))
    rng = random.Random(int(pop(argv, "--seed", "0")))
    bag = [o for p in made for _ in range(p["count"]) for o in p["ops"]]
    sizes = [len(p["ops"]) for p in made for _ in range(p["count"])]
    alphas = [o["alpha"] for o in bag if isinstance(o.get("alpha"), (int, float))] or [0.7, 1.3]
    gd = groups(model)
    g = list(gd)
    nums = config_numbers(model)
    out = []
    for i in range(n):
        ops = []
        for _ in range(rng.choice(sizes)):
            kind = rng.choice(bag)["op"]
            if kind == "set":
                f = rng.choice(list(nums))
                k = rng.choice(list(nums[f]))
                v = nums[f][k]
                ops.append({"op": "set", "file": f, "key": k, "value": rng.choice([v * 2, max(1, v // 2) if isinstance(v, int) else v / 2])})
                continue
            name, shape = rng.choice([x for x in g if kind == "scale" or len(x[1]) >= 3] or g)
            name = name.replace("{i}", "*") if rng.random() < .5 else name.replace("{i}", str(rng.choice(gd[(name, shape)])))
            ops.append({"op": "scale", "tensor": name, "alpha": rng.choice(alphas)} if kind == "scale" else
                       {"op": "resample", "tensor": name, "size": [max(1, shape[-2] // 2), max(1, shape[-1] // 2)], "mode": rng.choice(["area", "bilinear", "bicubic", "nearest"]), "alpha": rng.choice([1, 4])})
        out.append({"id": f"b{i:02d}", "count": 0, "ops": ops, "why": []})
    json.dump(out, open(os.path.join(work, "blind.json"), "w"), indent=1)
    print(f"{n} blind patches", flush=True)


def judge(work, argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--exam", required=True)
    ap.add_argument("--cards", required=True)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--tag", default="sg")
    ap.add_argument("--models", default="/dev/shm/selfedit/models")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    from mm_pipeline import Pipe, R
    P = Pipe(argparse.Namespace(work=work, family=a.family, cards=a.cards, dry_run=False, tag=a.tag))
    jd = os.path.join(work, "judge")
    os.makedirs(jd, exist_ok=True)
    arms = json.load(open(os.path.join(work, "patches.json")))[: a.top] + json.load(open(os.path.join(work, "blind.json")))

    def read(label, model):
        out = os.path.join(jd, f"{label}.jsonl")
        return P.job(f"{a.tag}_{label}", f"python -u rsi/mm_judge.py {out} --exam {a.exam} {P.args(model, R + '/pool.jsonl', maxlen=65536, seqs=64)}", model, out)     # a real image at patch 8 reaches 49029 tokens

    read("base", a.base)
    slots = threading.Semaphore(len(a.cards.split(",")))

    def one(p):
        if os.path.exists(os.path.join(jd, p["id"] + ".jsonl")):
            return
        dst = os.path.join(a.models, f"{a.tag}_{p['id']}")
        with slots:
            try:
                apply_patch(a.base, dst, p["ops"])
                if not read(p["id"], dst):
                    p["error"] = "did not load or crashed"
            except Exception as ex:
                p["error"] = f"{type(ex).__name__}: {ex}"
            # the patched checkpoint stays: nothing is deleted on the server; it goes on the owner's cleanup list
    P.par(*[(lambda p=p: one(p)) for p in arms])

    def load(label):
        path = os.path.join(jd, f"{label}.jsonl")
        return {r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else None
    base = load("base")
    rng = np.random.default_rng(0)

    def stat(d):
        d = np.array(d)
        if not d.size:
            return None
        m = d[rng.integers(0, d.size, (2000, d.size))].mean(1)
        return {"mean": round(float(d.mean()), 3), "lo": round(float(np.quantile(m, .05)), 3), "hi": round(float(np.quantile(m, .95)), 3)}
    lines, rows = [], []
    for p in arms:
        r = load(p["id"]) if not p.get("error") else None
        if r is None:
            row = dict(id=p["id"], count=p["count"], ops=p["ops"], error=p.get("error", "no read"))
        else:
            ids = [i for i in base if i in r]
            ex = [i for i in ids if base[i]["set"] == "exam"]
            po = [i for i in ids if base[i]["set"] == "pool"]
            row = dict(id=p["id"], count=p["count"], ops=p["ops"], exam=stat([r[i]["margin"] - base[i]["margin"] for i in ex]), pool=stat([r[i]["margin"] - base[i]["margin"] for i in po]),
                       right_exam=round(float(np.mean([r[i]["right"] for i in ex]) - np.mean([base[i]["right"] for i in ex])), 3), right_pool=round(float(np.mean([r[i]["right"] for i in po]) - np.mean([base[i]["right"] for i in po])), 3),
                       by={g: stat([r[i]["margin"] - base[i]["margin"] for i in ids if base[i]["group"] == g]) for g in sorted({base[i]["group"] for i in ids})})
        rows.append(row)
    with open(os.path.join(work, "ledger.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    good = [r for r in rows if not r.get("error")]
    for r in sorted(good, key=lambda r: -r["pool"]["mean"]):
        lines.append(f"{r['id']} x{r['count']}: real {r['pool']['mean']:+.3f} [{r['pool']['lo']:+.3f},{r['pool']['hi']:+.3f}] right {r['right_pool']:+.3f}; exam {r['exam']['mean']:+.3f} right {r['right_exam']:+.3f}; {json.dumps(r['ops'])[:220]}")
    for r in rows:
        if r.get("error"):
            lines.append(f"{r['id']} x{r['count']}: {r['error'][:120]}; {json.dumps(r['ops'])[:220]}")
    summary = []
    for name, pre in (("model", "m"), ("blind", "b")):
        got = [r for r in rows if r["id"].startswith(pre)]
        ok = [r["pool"]["mean"] for r in got if not r.get("error")]
        summary.append(f"{name}: {len(got)} patches, {len(ok)} ran, best real-problem margin {max(ok, default=float('nan')):+.3f}, median {float(np.median(ok)) if ok else float('nan'):+.3f}, clearly better than base {sum(r['pool']['lo'] > 0 for r in got if not r.get('error'))}")
    text = "\n".join(summary + [""] + lines) + "\n"
    open(os.path.join(work, "report.md"), "a").write(text)
    print(text, flush=True)


if __name__ == "__main__":
    cmd, work = sys.argv[1], os.path.abspath(sys.argv[2])
    {"propose": propose, "blind": blind, "judge": judge}[cmd](work, sys.argv[3:])
