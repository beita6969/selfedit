"""The fixed loop on a vision model: find its own limits with no benchmark, change the part of the network that matches, keep what survives. What passes is the next round's base.
Round: exam -> pool probe -> self-diagnosis and vote -> candidates -> pool generation paired with the base -> held-out -> ledger.
Candidates: the resolution family for an acuity limit; a blind scan of the network's writers, confirmed on a fresh exam seed, for any other limit. The model's vote orders the scan.
Every step is one job on one free card and leaves a file; a step whose file exists is not run again, so a stopped run resumes where it stopped.
Writes WORK/ledger.jsonl (symptom, candidate, numbers, decision) and WORK/report.md.

Usage: python rsi/mm_pipeline.py WORK --family qwen38|gemma4 --base MODEL --cards 1,2 [--rounds 2] [--no-heldout] [--dry-run]
"""
import argparse, json, os, queue, random, re, shlex, subprocess, sys, threading, time

N = "/home/bedicloud/sharestore2/zxc/selfedit"
R = f"{N}/rsi/mm"
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = f"{N}/venv/bin/python"
HELD = "babyvision mmvp mmstar blink erqa vstar countbench cvbench realworldqa hallusionbench mathvista charxiv".split()
FAMILY = {"qwen38": {"dirlib": f"{N}/xf/low/qwen38/library", "util": 0.85, "resolution": [("p8", {"patch": 8, "max": 4096}), ("p8eq", {"patch": 8, "max": 4096, "min": 3600})],
                     "vision": lambda b, part: f"visual.blocks.{b}." + ("attn.proj" if part == "attn" else "mlp.linear_fc2"), "vblocks": "vision_config.depth",
                     "text": lambda L: f"layers.{L}.self_attn.o_proj" if L % 4 == 3 else f"layers.{L}.linear_attn.out_proj", "tlayers": "text_config.num_hidden_layers"},
          "gemma4": {"dirlib": f"{N}/xf/gemma4/library", "util": 0.90, "resolution": [("t560", {"soft": 560}), ("t1120", {"soft": 1120})],
                     "vision": None, "text": lambda L: f"layers.{L}.post_attention_layernorm", "tlayers": "text_config.num_hidden_layers"}}   # the engine does not map Gemma's vision tower, so its scan covers the text layers; its post-norm undoes any scale of o_proj, so the gain of that norm is the writer
SCAN_LEVELS = {"locate": "cell:6,8,12,16,24", "trace": "trace:3,4,6", "count": "dots:1,2,3,4", "acuity": "ring:12,16/square:1,2"}
TASK_OF = {"locate": "cell", "trace": "trace", "count": "dots", "acuity": "ring"}
VOTE = {"B": "vision", "D": "text", "H": "text"}                    # menu items of self_propose the scan can serve; A is the resolution family, the rest are not wired here
LIMIT = 0.20                                                        # easy minus hard level of a task, in accuracy
TOP = 8                                                             # scan arms that go to the fresh seed
KEEP = 3                                                            # arms that go on to generation


def symptoms(acc):
    """A limit is a task the model does well when it is easy and badly when it is hard: no benchmark, no label of what should be wrong."""
    def mean(*keys):
        vals = [acc[k] for k in keys if k in acc]
        return sum(vals) / len(vals) if vals else None
    pairs = {"acuity": (mean("ring/64", "ring/96", "square/16", "square/24", "gap/16", "gap/24"), mean("ring/12", "ring/16", "square/1", "gap/1")),
             "locate": (mean("cell/4"), mean("cell/12", "cell/16")),
             "trace": (mean("trace/2"), mean("trace/4", "trace/6")),
             "count": (mean("dots/12", "dots/8"), mean("dots/1", "dots/2"))}
    out = {}
    for name, (easy, hard) in pairs.items():
        if easy is not None and hard is not None and easy >= 0.7 and easy - hard >= LIMIT:
            out[name] = {"easy": round(easy, 3), "hard": round(hard, 3)}
    return out


def build_resolution(family, src, dst, spec):
    """A candidate checkpoint of the resolution family: a finer patch kernel, or more image tokens where the family sets them in its config."""
    if os.path.exists(os.path.join(dst, "config.json")):
        return
    if "patch" in spec:
        cmd = [PY, f"{CODE}/rsi/vision_patch.py", "--model", src, "--out", dst, "--patch", str(spec["patch"]), "--kernel", "box", "--max-tokens", str(spec["max"])]
        if "min" in spec:
            cmd += ["--min-tokens", str(spec["min"])]
        subprocess.run(cmd, check=True)
        return
    os.makedirs(dst)

    def edit(o):
        for k, v in o.items():
            if k in ("vision_soft_tokens_per_image", "default_output_length", "image_seq_length", "max_soft_tokens"):
                o[k] = spec["soft"]
            elif isinstance(v, dict):
                edit(v)
    for name in os.listdir(src):
        if name in ("config.json", "processor_config.json"):
            cfg = json.load(open(os.path.join(src, name)))
            edit(cfg)
            json.dump(cfg, open(os.path.join(dst, name), "w"), indent=2)
        else:
            os.symlink(os.path.join(src, name), os.path.join(dst, name))


def scan_arms(family, model, votes):
    """Every writer of the network, down and up, the groups the model voted for first and each group shuffled."""
    fam, cfg = FAMILY[family], json.load(open(os.path.join(model, "config.json")))

    def get(path):
        o = cfg
        for k in path.split("."):
            o = o[k]
        return o
    groups = {"vision": [], "text": []}
    for a in (0.7, 1.3):
        if fam["vision"]:
            for b in range(get(fam["vblocks"])):
                for part in ("attn", "mlp"):
                    w = fam["vision"](b, part)
                    groups["vision"].append({"label": f"v{b}.{part} x{a}", "engine": [{"op": "scale", "param": w + ".weight", "alpha": a}, {"op": "scale", "param": w + ".bias", "alpha": a}]})
        for L in range(get(fam["tlayers"])):
            groups["text"].append({"label": f"t{L}.attn x{a}", "engine": [{"op": "scale", "param": fam["text"](L), "alpha": a}]})
    order = sorted(groups, key=lambda g: -votes.get(g, 0))
    rng, arms = random.Random(0), []
    for g in order:
        rng.shuffle(groups[g])
        arms += groups[g]
    return arms


def read_votes(path):
    counts = {}
    for r in map(json.loads, open(path)):
        g = VOTE.get(r.get("pick"))
        if g:
            counts[g] = counts.get(g, 0) + 1
    return counts


def load_traces(path):
    d = {}
    for r in map(json.loads, open(path)):
        d.setdefault(r["id"], []).append(bool(r["ok"]) and r.get("finish") != "length")
    return d


def paired(a, b, seed=0):
    """Mean difference b - a over shared problems, in points, with a 90 % bootstrap interval."""
    ids = sorted(set(a) & set(b))
    d = [sum(b[i]) / len(b[i]) - sum(a[i]) / len(a[i]) for i in ids]
    if not d:
        return {"n": 0, "diff": 0.0, "lo": 0.0, "hi": 0.0}
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    return {"n": len(d), "diff": round(100 * sum(d) / len(d), 2), "lo": round(100 * means[100], 2), "hi": round(100 * means[1899], 2)}


class Pipe:
    def __init__(self, a):
        self.a, self.family, self.fam = a, a.family, FAMILY[a.family]
        self.W = os.path.abspath(a.work)
        os.makedirs(self.W, exist_ok=True)
        self.free = queue.Queue()
        for c in a.cards.split(","):
            self.free.put(c)
        self.lock = threading.Lock()

    def say(self, text):
        line = f"[{time.strftime('%H:%M:%S')}] {text}"
        print(line, flush=True)
        with self.lock, open(os.path.join(self.W, "report.md"), "a") as f:
            f.write(line + "\n")

    def ledger(self, **row):
        with self.lock, open(os.path.join(self.W, "ledger.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")

    def args(self, model, pool, maxlen=36864, seqs=32):
        return (f"--mm --model {model} --pool {pool} --dirlib {self.fam['dirlib']} --seed 31 --tp 1 --max-model-len {maxlen} --max-num-batched-tokens 4096 "
                f"--kv-cache-dtype fp8 --effort medium --budget 16384 --mb 128 --samples 2 --max-num-seqs {seqs} --gpu-util {self.fam['util']}")

    def job(self, name, exe, model, done):
        """Run exe on the next free card through the normal launcher and return when it has exited; done is the file that shows it worked."""
        if os.path.exists(done):
            return True
        if self.a.dry_run:
            self.say(f"would run {name}: {exe[:160]}")
            return True
        card = self.free.get()
        try:
            while int(subprocess.check_output(["nvidia-smi", "-i", card, "--query-gpu=memory.used", "--format=csv,noheader,nounits"])) >= 8000:
                time.sleep(5)
            env = dict(os.environ, SE_FAMILY=self.family, SE_MODEL=model, SE_LINEAGE=f"{CODE}/jobs/xf_n15.sh", SE_FRAC="0.92", SE_EXEC=exe)
            self.say(f"{name} on card {card}")
            subprocess.run(["bash", f"{N}/code/jobs/n15_start.sh", card, name], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pat = re.escape(exe[:160])
            for _ in range(24):
                if subprocess.run(["pgrep", "-f", pat], stdout=subprocess.DEVNULL).returncode == 0:
                    break
                time.sleep(5)
            while subprocess.run(["pgrep", "-f", pat], stdout=subprocess.DEVNULL).returncode == 0:
                time.sleep(15)
        finally:
            self.free.put(card)
        if not os.path.exists(done):
            self.say(f"{name} ended without {done}")
            return False
        return True

    def par(self, *thunks):
        out = [None] * len(thunks)

        def run(i, t):
            out[i] = t()
        ts = [threading.Thread(target=run, args=(i, t)) for i, t in enumerate(thunks)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        return out

    def exam(self, d, label, model):
        path = os.path.join(d, "summary.jsonl")

        def row():
            return next((r for r in map(json.loads, open(path)) if r["label"] == label), None) if os.path.exists(path) else None
        marker = os.path.join(d, f"{label}.jsonl")
        self.job(f"{self.a.tag}_exam_{label}", f"python -u rsi/eye_exam.py run {d} --label {label} {self.args(model, os.path.join(d, 'items.jsonl'), seqs=64)}", model, marker)
        return (row() or {}).get("acc", {})

    def probe(self, d, label, model, sets, maxlen=36864, engine=None):
        os.makedirs(d, exist_ok=True)
        spec = os.path.join(d, f"spec_{label}.json")
        json.dump([{"label": label, "effort": "medium", **({"engine": engine} if engine else {})}], open(spec, "w"))
        first = sets[0].split(":")[0]
        done = os.path.join(d, f"{label}.{sets[-1].split(':')[0]}.jsonl")
        self.job(f"{self.a.tag}_{label}", f"python -u rsi/probe_base.py {d} --probes {spec} --sets {' '.join(sets)} --curve 2048,4096,8192,16384 {self.args(model, R + '/pool.jsonl', maxlen=maxlen)}", model, done)
        return os.path.join(d, f"{label}.{first}.jsonl")

    def scan(self, d, base, arms_all, limits):
        """Blind scan of the writers against the limits found, then the best arms again on a seed they were not picked on."""
        only = "/".join(SCAN_LEVELS[s] for s in limits if s in SCAN_LEVELS)
        tasks = [TASK_OF[s] for s in limits if s in TASK_OF]

        def ranked(sd, arms, seed, label):
            os.makedirs(sd, exist_ok=True)
            if not os.path.exists(os.path.join(sd, "items.jsonl")) and not self.a.dry_run:
                subprocess.run([PY, f"{CODE}/rsi/eye_exam.py", "build", sd, "--per-level", "60", "--seed", str(seed), "--only", only], check=True)
            ef = os.path.join(sd, "arms.json")
            json.dump(arms, open(ef, "w"))
            self.job(f"{self.a.tag}_{label}", f"python -u rsi/eye_exam.py run {sd} --label {label} --edits {ef} {self.args(base, os.path.join(sd, 'items.jsonl'), seqs=64)}", base, os.path.join(sd, "scan_done"))
            out = {}
            path = os.path.join(sd, "summary.jsonl")
            for r in map(json.loads, open(path)) if os.path.exists(path) else []:
                if ":" in r["label"] and "margin" in r:
                    out[r["label"].split(":", 1)[1]] = r["margin"]
            return out
        first = ranked(os.path.join(d, "scan1"), arms_all, 1, "scan1")

        def gain(m):
            return min((m[t]["mean"] for t in tasks if t in m), default=-9.0)      # an arm must help every limit it is run on, or one task's gain could be another's loss
        best = sorted(first, key=lambda k: -gain(first[k]))[: TOP]
        for k in best:
            self.say(f"scan1 {k}: " + " ".join(f"{t} {first[k][t]['mean']:+.3f}" for t in tasks if t in first[k]))
        arms = [x for x in arms_all if x["label"] in best]
        second = ranked(os.path.join(d, "scan2"), arms, 2, "scan2") if arms else {}
        keep = [k for k in sorted(second, key=lambda k: -gain(second[k])) if all(second[k][t]["lo"] > 0 for t in tasks if t in second[k])][: KEEP]
        for k in second:
            self.say(f"scan2 {k}: " + " ".join(f"{t} {second[k][t]['mean']:+.3f} [{second[k][t]['lo']:+.3f},{second[k][t]['hi']:+.3f}]" for t in tasks if t in second[k]) + (" kept" if k in keep else ""))
        return [x for x in arms_all if x["label"] in keep]

    def round(self, i, base):
        d = os.path.join(self.W, f"r{i}")
        os.makedirs(d, exist_ok=True)
        tag = self.a.tag
        eye = os.path.join(d, "eye")
        if not os.path.exists(os.path.join(eye, "items.jsonl")) and not self.a.dry_run:
            subprocess.run([PY, f"{CODE}/rsi/eye_exam.py", "build", eye, "--per-level", "40", "--seed", "0"], check=True)
        acc, _ = self.par(lambda: self.exam(eye, "base", base), lambda: self.probe(os.path.join(d, "probe"), "base", base, [f"v:{R}/pool.jsonl"]))
        trace = os.path.join(d, "probe", "base.v.jsonl")
        limits = symptoms(acc)
        self.say(f"round {i}, base {base}: limits {limits or 'none'}")
        digest, votes = os.path.join(d, "diagnosis.md"), os.path.join(d, "votes.jsonl")
        self.job(f"{tag}_diag", f"python -u rsi/self_diagnose.py {d}/diag --traces {trace} --cases 120 --out {digest} {self.args(base, R + '/pool.jsonl')}", base, digest)
        if os.path.exists(digest):
            self.par(lambda: self.job(f"{tag}_vote", f"python -u rsi/self_propose.py --digest {digest} --out {votes} --samples 64 {self.args(base, R + '/pool.jsonl').replace('--mm ', '')}", base, votes),
                     lambda: self.job(f"{tag}_free", f"python -u rsi/self_propose.py --digest {digest} --out {d}/free.jsonl --samples 16 --free 1 {self.args(base, R + '/pool.jsonl').replace('--mm ', '')}", base, f"{d}/free.jsonl"))
        counts = read_votes(votes) if os.path.exists(votes) else {}
        self.say(f"model's vote among what the scan can serve: {counts or 'none'}")
        cands = []
        if "acuity" in limits:
            for name, spec in self.fam["resolution"]:
                dst = os.path.join(self.a.models, f"{os.path.basename(base)}-{name}")
                if not self.a.dry_run:
                    build_resolution(self.family, base, dst, spec)
                cands.append({"name": name, "family": "resolution", "symptom": "acuity", "model": dst})
        rest = [s for s in limits if s != "acuity"]
        if rest:
            for x in self.scan(d, base, scan_arms(self.family, base, counts), rest):
                cands.append({"name": x["label"].replace(" ", "_"), "family": "scan", "symptom": ",".join(rest), "model": base, "engine": x["engine"]})
        if not cands:
            self.say(f"round {i}: nothing to try, stopping")
            return None

        def pool_test(c):
            run = os.path.join(d, "cand", c["name"])
            self.probe(run, "c", c["model"], [f"v:{R}/pool.jsonl"], engine=c.get("engine"))
            exam_acc = {} if c.get("engine") else self.exam(eye, c["name"], c["model"])
            path = os.path.join(run, "c.v.jsonl")
            c["pool"] = paired(load_traces(trace), load_traces(path)) if os.path.exists(path) else {"n": 0, "diff": 0.0, "lo": 0.0, "hi": 0.0}
            c["limits_after"] = symptoms(exam_acc) if exam_acc else None
            self.say(f"pool {c['name']}: {c['pool']['diff']:+.2f} [{c['pool']['lo']:+.2f},{c['pool']['hi']:+.2f}] n={c['pool']['n']}; limits after {c['limits_after']}")
        self.par(*[(lambda c=c: pool_test(c)) for c in cands])
        for c in sorted(cands, key=lambda c: -c["pool"]["diff"]):
            row = dict(round=i, base=base, limits=limits, vote=counts, candidate=c["name"], family=c["family"], symptom=c["symptom"], pool=c["pool"])
            if c["pool"]["lo"] <= 0:
                self.ledger(**row, decision="rejected on pool")
                continue
            if self.a.no_heldout:
                self.ledger(**row, decision="passed pool, held-out not run")
                return c["model"] if not c.get("engine") else self.export(d, c, base)
            sets = [f"{s}:{N}/data/mm/sets/{s}.jsonl" for s in HELD]
            h0, h1 = os.path.join(d, "held_base"), os.path.join(d, "held", c["name"])
            self.par(lambda: self.probe(h0, "h", base, sets, maxlen=81920), lambda: self.probe(h1, "h", c["model"], sets, maxlen=81920, engine=c.get("engine")))
            per, ok = {}, True
            pooled = {}
            for s in HELD:
                a, b = os.path.join(h0, f"h.{s}.jsonl"), os.path.join(h1, f"h.{s}.jsonl")
                if os.path.exists(a) and os.path.exists(b):
                    x, y = load_traces(a), load_traces(b)
                    per[s] = paired(x, y)
                    pooled.update({f"{s}/{k}": sum(y[k]) / len(y[k]) - sum(x[k]) / len(x[k]) for k in set(x) & set(y)})
            mean = sum(p["diff"] for p in per.values()) / max(1, len(per))
            hurt = [s for s, p in per.items() if p["hi"] < 0]
            ok = sum(p["lo"] > 0 for p in per.values()) >= 2 and mean > 0
            self.say(f"held-out {c['name']}: mean of sets {mean:+.2f}, " + " ".join(f"{s} {p['diff']:+.1f}" for s, p in per.items()) + f"; clearly hurt: {hurt or 'none'}")
            self.ledger(**row, heldout=per, heldout_mean=round(mean, 2), hurt=hurt, decision="accepted" if ok else "rejected on held-out")
            if ok:
                return c["model"] if not c.get("engine") else self.export(d, c, base)
        return None

    def export(self, d, c, base):
        """An engine edit becomes the next base as a checkpoint on disk."""
        dst = os.path.join(self.a.models, f"{os.path.basename(base)}-{c['name']}")
        spec = os.path.join(d, f"export_{c['name']}.json")
        json.dump(c["engine"], open(spec, "w"))
        self.job(f"{self.a.tag}_export", f"python -u rsi/mm_export.py --edits {spec} --out {dst} {self.args(base, R + '/pool.jsonl')}", base, os.path.join(dst, "config.json"))
        return dst

    def run(self):
        base = self.a.base
        for i in range(self.a.rounds):
            new = self.round(i, base)
            if new is None:
                self.say(f"stopped in round {i}, final base {base}")
                return
            self.say(f"round {i} accepted, new base {new}")
            base = new
        self.say(f"finished {self.a.rounds} rounds, final base {base}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--family", required=True, choices=list(FAMILY))
    ap.add_argument("--base", required=True)
    ap.add_argument("--cards", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--models", default="/dev/shm/selfedit/models")
    ap.add_argument("--tag", default="pipe")
    ap.add_argument("--no-heldout", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    Pipe(ap.parse_args()).run()


if __name__ == "__main__":
    main()
