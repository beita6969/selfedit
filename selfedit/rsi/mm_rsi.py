"""The outer RSI harness on a vision model: one lineage of small scale edits to the model's own writer tensors, with no training. A read says which way each writer pulls, a chooser picks among the rules' candidates, generation on real problems decides.
Round: read at the current scales -> candidates at equal estimated cost, composed with the scales already accepted -> choice -> generation on the learn set -> paired test on the half not used for selection -> accept, or keep the state and halve the budgets.
The chooser is --mode: fixed (the harness's own rule), model (the checkpoint with its current scales reads its ledger, its accuracy and its own failures on the selection half, and the table, and may ask for the next read to use one bench), pipeline (as model, but it writes the rule itself: an expression over the read, its budget and its clip; the test stays locked), random (the control).
Every step is one job on one card under 8 GiB used and leaves a file; a step whose file exists is not run again, so a stopped run resumes where it stopped. A job that fails twice halts the run with nothing decided. Held-out sets are never read here; a separate script takes WORK/final.json to them.
Writes WORK/state.json (round, cumulative scales), WORK/ledger.jsonl (every choice and decision, rejected arms too), WORK/report.md, WORK/final.json (final scales and engine edits as one arm), WORK/rN/ (one round), WORK/learn/LABEL.learn.jsonl (generation).
A base result already on disk can be shared between lineages by linking it to WORK/learn/base.learn.jsonl before the start. Do not give dispatch.py the same cards: lineages lock cards against each other, not against it.

Usage: python rsi/mm_rsi.py WORK --mode fixed|model|pipeline|random --rounds 4 --cards 4,5 --base MODEL --learn POOL.jsonl --items ITEMSDIR [--dry-run]
       [--budgets .02,.1,.4 --pick 2 --floor -0.5 --min-gain 0.5 --patience 2 --draws 2 --cut 8192 --seed 0 --table-rules index,raw,bang,natural --cost COST.json]
       [--root N --dirlib D --py PY --start SCRIPT|direct --env-file F --lock-dir D]    a second server: --root /data/fd/mingdazhang/selfedit --start direct --py VENV/bin/python --env-file VENV/bin/activate, run inside one job that holds the cards
       python3 rsi/mm_rsi.py --self-test DIR     the tests, on the system python, fake numbers only
"""
import argparse, ast, contextlib, fcntl, hashlib, io, json, os, random, re, subprocess, sys, threading, time, types, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_pipeline import paired                                      # both stdlib only at import
from self_diagnose import last_json

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = "/home/bedicloud/sharestore2/zxc/selfedit"
FIXED_RULES = ("index", "raw", "bang", "natural")
FREE_MIB = 8000                                                     # a card takes a job under this much used, as dispatch.py
FOCUS_MIN = 16                                                      # selection items a bench needs before a read can focus on it
DIGEST = "@@DIGEST@@"                                               # the job fills in its own failures
RULE_TEXT = {"raw": "the writers with the largest gain, step gain over curvature", "index": "every writer, gain shrunk by its reliability, over curvature", "bang": "the same step on every writer whose sign is established",
             "natural": "gain shrunk by reliability, over the cost per writer", "flip": "the index with its sign reversed", "random": "the writers and sizes of the index with random signs"}
CHOOSE = """You are an open-weight vision-language model. You are being improved by small scale edits to your own writer tensors, with no training. A writer is the last linear before a residual add: t<L>.attn in text layer L, v<B>.attn and v<B>.mlp in vision block B. Edits that were kept stay in you: you are running with them now.
Each round a read of your own gradients on half of the learn problems proposes candidates. The ones chosen are generated with, and measured on the other half, which you do not see here. The best one with a mean above {min_gain:+.1f} points and a lower bound above {floor:+.1f} points is kept.

Your ledger, what was tried and what it measured, in points of accuracy with the 90 % interval:
{ledger}

Your accuracy now on the half you may see, per bench:
{acc}

How you failed on those problems, in your own words after reading your own failed attempts:
{digest}

The rules:
{rules}

Candidates this round. Predicted gain is in log-odds of the right option, from the read with thinking off, not measured on reasoning. Cost is the estimated size of the change. Top writers show the largest changes of scale, as a fraction:
{table}

Choose exactly {pick} candidates. You may also ask for the next round's read to use only the problems of one bench, one of {benches}, or null to use all of them.
Reply with one JSON object and nothing else: {{"picks": [<{pick} labels>], "why": "<one line>", "focus": <bench name or null>}}"""
WRITE = """You are an open-weight vision-language model. You are being improved by small scale edits to your own writer tensors, with no training. A writer is the last linear before a residual add: t<L>.attn in text layer L, v<B>.attn and v<B>.mlp in vision block B. Edits that were kept stay in you: you are running with them now.
Each round a read of your own gradients on half of the learn problems gives, per writer: g, the change in log-odds of the right option per unit of scale, thinking off; se, its standard error; z = g/se; rel, its reliability, 0 to 1; h, its curvature, negative; F, its estimated cost per dose squared.
In this mode the pipeline is yours: you write the rule that turns the read into an edit, how large the edit is, and its clip. The harness has no rule of its own here. You cannot change the test: each edit is generated with and measured on the other half of the problems, which you do not see. The better of your rules with a mean above {min_gain:+.1f} points and a lower bound above {floor:+.1f} points is kept.

Rules the harness knows, as examples. You may use them, change them or write something else:
- rel*g/-h: every writer, gain shrunk by reliability, over curvature
- (abs(z)>2.5)*sign(g): the same step on every writer whose sign is established
- rel*g/F: gain shrunk by reliability, over cost
A rule is one expression giving one number per writer, the direction of the edit. It is then rescaled so the whole edit has the budget you give. Budget is sum F d^2 / 2, between .005 and .8. For scale: a joint edit built on drawn exams cost about 1.6 at full dose and lost points on real problems; at a quarter of that dose it cost about .1. Clip is the largest change of scale on one writer, between .02 and .45.
Names: {names}, tower text attn mlp (each 1 or 0 for a writer). pos is 0 at the first block of a writer's stack and 1 at the last. Functions: {funcs}. Operators: + - * / ** > < >= <= & |. Nothing else.

Your ledger, what was tried and what it measured, in points of accuracy with the 90 % interval:
{ledger}

Your accuracy now on the half you may see, per bench:
{acc}

How you failed on those problems, in your own words after reading your own failed attempts:
{digest}

This round's read, by group of writers, then the strongest single writers:
{read}

Write exactly {pick} different rules. You may also ask for the next round's read to use only the problems of one bench, one of {benches}, or null to use all of them.
Reply with one JSON object and nothing else: {{"rules": [{{"expr": "<expression>", "budget": <number>, "clip": <number>}}, ...], "why": "<one line>", "focus": <bench name or null>}}"""
RETRY_RULE = """

Your last reply could not be used: {error}. Reply with one JSON object and nothing else: {{"rules": [{{"expr": "<expression>", "budget": <number>, "clip": <number>}}, ...], "why": "<one line>", "focus": <bench name or null>}}"""
RULE_VARS = ("g", "h", "se", "z", "rel", "F", "pos", "tower", "text", "attn", "mlp")
RULE_FUNCS = ("abs", "sign", "sqrt", "tanh", "exp", "where", "clip", "maximum", "minimum", "mean", "median", "quantile")
RULE_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.BitAnd, ast.BitOr, ast.Load)
RETRY = """

Your last reply could not be used: {error}. Reply with one JSON object and nothing else: {{"picks": [<{pick} labels>], "why": "<one line>", "focus": <bench name or null>}}"""


class Halt(Exception):
    """A step could not be done; nothing is decided, a new start resumes here."""


class Stop(BaseException):
    """The self-test's kill."""


def half(i):
    return zlib.crc32(str(i).encode()) % 2


def put(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def engine_one(k, alpha):
    """A writer's engine edits, in the names rsi/mm_select.py writes."""
    if k.startswith("v"):
        b, part = re.fullmatch(r"v(\d+)\.(attn|mlp)", k).groups()
        w = f"visual.blocks.{b}." + ("attn.proj" if part == "attn" else "mlp.linear_fc2")
        return [{"op": "scale", "param": w + ".weight", "alpha": alpha}, {"op": "scale", "param": w + ".bias", "alpha": alpha}]
    L, part = re.fullmatch(r"t(\d+)\.(attn|mlp)", k).groups()
    L = int(L)
    if part == "mlp":
        return [{"op": "scale", "param": f"layers.{L}.mlp.down_proj", "alpha": alpha}]
    return [{"op": "scale", "param": f"layers.{L}.self_attn.o_proj" if L % 4 == 3 else f"layers.{L}.linear_attn.out_proj", "alpha": alpha}]


def engine_of(scales):
    return [e for k, a in sorted(scales.items()) for e in engine_one(k, a)]


def writer_of(e):
    """The writer an engine edit scales. Anything else is an error, so no edit of an arm is lost silently."""
    m = re.fullmatch(r"visual\.blocks\.(\d+)\.(attn\.proj|mlp\.linear_fc2)\.(weight|bias)", str(e.get("param")))
    if m and e.get("op") == "scale":
        return f"v{m[1]}." + ("attn" if m[2] == "attn.proj" else "mlp")
    m = re.fullmatch(r"layers\.(\d+)\.(self_attn\.o_proj|linear_attn\.out_proj|mlp\.down_proj)", str(e.get("param")))
    if m and e.get("op") == "scale":
        return f"t{m[1]}." + ("mlp" if m[2] == "mlp.down_proj" else "attn")
    raise ValueError(f"not a writer scale: {e}")


def compose(scales, edits):
    """An arm's edits on top of the scales already accepted: old scale times new scale; writers not in the arm keep their old scale. The engine starts from the unedited checkpoint, so the result is what it must apply."""
    new = {}
    for e in edits:
        if new.setdefault(writer_of(e), e["alpha"]) != e["alpha"]:
            raise ValueError(f"weight and bias of {writer_of(e)} differ")
    out = dict(scales)
    for k, a in new.items():
        out[k] = round(scales.get(k, 1.0) * a, 6)
    return {k: v for k, v in sorted(out.items()) if v != 1.0}


def load_results(path, cut, which=None):
    """id -> right within the cut, per sample, as rsi/cmp_learn; which keeps one half only."""
    d = {}
    for r in map(json.loads, open(path)):
        if which is None or half(r["id"]) == which:
            d.setdefault(r["id"], []).append(bool(r["ok"]) and int(r["tokens"]) <= cut)
    return d


def bench_means(vals):
    out = {}
    for i, v in vals.items():
        out.setdefault(i.split("/")[0], []).append(v)
    return {k: (sum(v) / len(v), len(v)) for k, v in sorted(out.items())}


def bench_gain(a, b):
    d = {i: sum(b[i]) / len(b[i]) - sum(a[i]) / len(a[i]) for i in set(a) & set(b)}
    return {k: round(100 * m, 2) for k, (m, _) in bench_means(d).items()}


def verdict(st, floor, min_gain, need):
    """Why an arm fails, or None."""
    if st["n"] < need:
        return f"only {st['n']} problems of {need}"
    if st["lo"] <= floor:
        return f"lower bound {st['lo']:+.2f} not above {floor:+.2f}"
    if st["diff"] <= min_gain:
        return f"mean {st['diff']:+.2f} not above {min_gain:+.2f}"
    return None


def fixed_pick(table, pick):
    """Highest predicted gain, one per rule, among the rules that are not controls."""
    best = {}
    for r in table:
        if r["rule"] in FIXED_RULES and (r["rule"] not in best or r["predicted_gain"] > best[r["rule"]]["predicted_gain"]):
            best[r["rule"]] = r
    return [r["label"] for r in sorted(best.values(), key=lambda r: (-r["predicted_gain"], r["label"]))[:pick]]


def make_table(arms, info, rules):
    """The candidates that do something, from mm_select's two files; a rule with no established writer writes an empty arm. Arms with the same edits are one candidate: with no --cost, mm_select's natural is its index, edit for edit, at every budget."""
    out, seen = [], {}
    for arm in arms:
        rule, x = arm["label"].split("_")[1], info[arm["label"]]
        if rule in rules and x["writers"] > 0 and x["predicted_gain"] == x["predicted_gain"]:
            same = seen.get(json.dumps(arm["engine"], sort_keys=True))
            if same:
                same["same"].append(arm["label"])
                continue
            top = sorted(x["doses"].items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:5]
            out.append(dict(label=arm["label"], rule=rule, budget=float(arm["label"].split("_")[2]), writers=x["writers"], predicted_gain=round(x["predicted_gain"], 4), cost=round(x["cost"], 5), top=top, same=[]))
            seen[json.dumps(arm["engine"], sort_keys=True)] = out[-1]
    return out


def parse_choice(text, labels, benches, pick):
    """The model's reply, or the reason it cannot be used: (picks, why, focus), error."""
    obj = last_json(text, "picks")
    if obj is None:
        return None, "no JSON object with picks"
    picks, why, focus = obj["picks"], obj.get("why"), obj.get("focus")
    if not isinstance(picks, list) or len(picks) != pick or not all(isinstance(x, str) for x in picks) or len(set(picks)) != pick:
        return None, f"picks must be {pick} different labels"
    if any(x not in labels for x in picks):
        return None, f"not in the table: {[x for x in picks if x not in labels]}"
    if not isinstance(why, str) or not why.strip():
        return None, "why must be one line of text"
    if focus is not None and focus not in benches:
        return None, f"focus must be null or one of {benches}"
    return (picks, " ".join(why.split()), focus), None


def check_expr(expr):
    """Why a rule the model wrote cannot be run, or None. Arithmetic on the read's arrays and nothing else, so rsi/mm_select.py can evaluate it as written."""
    if not isinstance(expr, str) or not 0 < len(expr) <= 300:
        return "expr must be text of at most 300 characters"
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError) as ex:
        return f"does not parse: {ex}"
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            if n.id not in RULE_VARS + RULE_FUNCS:
                return f"unknown name {n.id}"
        elif isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in RULE_FUNCS or n.keywords:
                return "only calls of the listed functions"
        elif isinstance(n, ast.Constant):
            if isinstance(n.value, bool) or not isinstance(n.value, (int, float)):
                return "only numbers as constants"
        elif not isinstance(n, RULE_NODES):
            return f"{type(n).__name__} is not allowed"
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) and not (isinstance(n.right, ast.Constant) and isinstance(n.right.value, (int, float)) and abs(n.right.value) <= 4):
            return "a power must be a number of size at most 4"
    return None


def parse_rules(text, benches, pick):
    """The rules the model wrote, or the reason they cannot be used: (rules, why, focus), error."""
    obj = last_json(text, "rules")
    if obj is None:
        return None, "no JSON object with rules"
    rules, why, focus = obj["rules"], obj.get("why"), obj.get("focus")
    if not isinstance(rules, list) or len(rules) != pick or not all(isinstance(r, dict) for r in rules):
        return None, f"rules must be a list of {pick} objects"
    out = []
    for r in rules:
        err = check_expr(r.get("expr"))
        if err:
            return None, f"rule {r.get('expr')!r}: {err}"
        b, c = r.get("budget"), r.get("clip", .45)
        if isinstance(b, bool) or not isinstance(b, (int, float)) or not .005 <= b <= .8:
            return None, "budget must be a number from .005 to .8"
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not .02 <= c <= .45:
            return None, "clip must be a number from .02 to .45"
        out.append(dict(expr=" ".join(r["expr"].split()), budget=float(b), clip=float(c)))
    if len({(r["expr"], r["budget"], r["clip"]) for r in out}) != pick:
        return None, f"the {pick} rules must differ"
    if not isinstance(why, str) or not why.strip():
        return None, "why must be one line of text"
    if focus is not None and focus not in benches:
        return None, f"focus must be null or one of {benches}"
    return (out, " ".join(why.split()), focus), None


def ask_model(send, prompt, sp):
    """Ask, retry once with less thinking so the JSON fits. send(text, effort) -> dict with content, reasoning, finish, tokens."""
    tries, err = [], None
    for n in range(2):
        rule = sp.get("kind") == "rule"
        s = send(prompt if n == 0 else prompt + (RETRY_RULE if rule else RETRY).format(error=err, pick=sp["pick"]), sp["effort"] if n == 0 else "low")
        got, err = parse_rules(s["content"], sp["benches"], sp["pick"]) if rule else parse_choice(s["content"], sp["labels"], sp["benches"], sp["pick"])
        tries.append(dict(s, error=err))
        if got:
            return dict(picks=got[0], why=got[1], focus=got[2], tries=tries)
    return dict(picks=None, why=None, focus=None, tries=tries)


def ask_and_save(sp, send):
    prompt = sp["prompt"].replace(DIGEST, open(sp["digest"]).read())
    with open(sp["prompt_file"], "w") as f:
        f.write(prompt)
    put(sp["reply"], ask_model(send, prompt, sp))


def ask_worker(spec, argv):
    """One job: the checkpoint with the current scales applied reads its own failures on the selection half (rsi/self_diagnose.py, unchanged), then chooses. All heavy imports are here; vLLM TP>1 workers re-import this file."""
    sp = json.load(open(spec))
    sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
    import surgery
    real, live = surgery.Surgeon, []

    def scaled(*x, **k):
        S = real(*x, **k)
        if sp["engine"]:
            S.apply(sp["engine"])
        live.append(S)
        return S
    surgery.Surgeon = scaled                                        # self_diagnose imports it by name inside main()
    if not os.path.exists(sp["digest"]):
        import self_diagnose
        sys.argv = ["self_diagnose.py", sp["run"], "--traces", sp["traces"], "--cases", str(sp["cases"]), "--out", sp["digest"]] + argv
        self_diagnose.main()
    else:
        sys.argv = ["loop.py", "--run-dir", sp["run"]] + argv
    import rsi.loop as L
    a = L.args
    S = live[0] if live else scaled(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens, gpu_memory_utilization=a.gpu_util, kv_cache_dtype=a.kv_cache_dtype, multimodal=True)

    def send(text, effort):
        s = S.generate([[{"role": "user", "content": text}]], n=1, sampling=dict(L.SAMPLING), max_tokens=sp["think"], reasoning_effort=effort)[0]["samples"][0]
        return dict(content=s["content"] or "", reasoning=(s["reasoning"] or "")[-3000:], finish=s["finish_reason"], tokens=s["num_tokens"])
    ask_and_save(sp, send)


class Fake:
    """The dry run's world, nothing real and no numpy: a toy model whose accuracy is a fixed linear function of its writer scales, so accepted edits carry over and results do not depend on when a step ran."""
    WRITERS = [f"v{b}.{p}" for b in range(27) for p in ("attn", "mlp")] + [f"t{L}.attn" for L in range(64)]

    def __init__(self, ids):
        self.ids, self.gain, self.reply = ids, self.toy_gain, None

    @staticmethod
    def u(s):
        return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 2 ** 32     # crc32 of similar short strings is far from uniform

    def toy_gain(self, scales):
        return sum(10 * (self.u(k) - .5) * (s - 1) for k, s in scales.items())

    def probe(self, path, scales):
        g = self.gain(scales) / 100
        with open(path, "w") as f:
            for i in self.ids:
                for k in range(2):
                    key, cut = f"{i}/{k}", self.u(f"{i}/{k}/len") < .03                    # the same draw for every arm, so a difference is the arm's
                    f.write(json.dumps(dict(id=i, kind="mc", answer="A", sample=k, ok=not cut and self.u(key) < .6 + g, tokens=8192 if cut else 500 + int(7000 * self.u(key + "/tok")), finish="length" if cut else "stop", reasoning="fake reasoning", content="fake")) + "\n")

    def select(self, out, budgets, prefix):
        arms, info = [], {}
        for b in budgets:
            for rule in ("raw", "index", "bang", "natural", "flip", "random"):
                label = f"{prefix}_{rule}_{b:g}"
                rng = random.Random(label.replace("natural", "index"))     # as mm_select with no --cost
                size = (2 * b / 8) ** .5
                doses = {}
                for k in rng.sample(self.WRITERS, 8):
                    aligned = 1.0 if self.u(k) > .5 else -1.0
                    sign = {"flip": -aligned, "random": rng.choice([-1.0, 1.0]), "bang": rng.choice([-1.0, 1.0]), "raw": aligned if rng.random() < .8 else -aligned}.get(rule, aligned)
                    doses[k] = round(size * sign, 4)
                arms.append({"label": label, "effort": "medium", "engine": [e for k, x in doses.items() for e in engine_one(k, round(1 + x, 4))]})
                info[label] = dict(writers=8, predicted_gain=sum(abs(self.u(k) - .5) * x * (1 if self.u(k) > .5 else -1) for k, x in doses.items()), cost=b, doses=doses)
        json.dump(arms, open(out, "w"))
        json.dump(info, open(out + ".info", "w"), indent=1)

    def own(self, out, rules, prefix):
        """mm_select --expr: an arm per written rule."""
        arms, info = [], {}
        for r in rules:
            label = f"{prefix}_{r['name']}_{r['budget']:g}"
            rng = random.Random(label + r["expr"])
            size = (2 * r["budget"] / 8) ** .5
            doses = {k: round(size * (1.0 if self.u(k) > .5 else -1.0), 4) for k in rng.sample(self.WRITERS, 8)}
            arms.append({"label": label, "effort": "medium", "engine": [e for k, x in doses.items() for e in engine_one(k, round(1 + x, 4))]})
            info[label] = dict(writers=8, predicted_gain=sum(abs(self.u(k) - .5) * x * (1 if self.u(k) > .5 else -1) for k, x in doses.items()), cost=r["budget"], doses=doses)
        json.dump(arms, open(out, "w"))
        json.dump(info, open(out + ".info", "w"), indent=1)

    def send(self, sp, text, effort):
        if self.reply:
            return dict(content=self.reply(text, effort), reasoning="", finish="stop", tokens=10)
        rng = random.Random(text)
        if sp.get("kind") == "rule":
            return dict(content=json.dumps(dict(rules=[dict(expr=f"rel*g/-h*{rng.random():.3f}", budget=.1, clip=.3), dict(expr="(abs(z)>2.5)*sign(g)", budget=rng.choice([.02, .04]), clip=.2)], why="fake reason", focus=rng.choice(sp["benches"] + [None, None]))), reasoning="", finish="stop", tokens=10)
        return dict(content=json.dumps(dict(picks=rng.sample(sp["labels"], sp["pick"]), why="fake reason", focus=rng.choice(sp["benches"] + [None, None]))), reasoning="", finish="stop", tokens=10)

    def ask(self, sp):
        with open(sp["digest"], "w") as f:
            f.write("# Diagnosis: fake, dry run\n")
        ask_and_save(sp, lambda text, effort: self.send(sp, text, effort))


class Rsi:
    def __init__(self, a):
        self.a, self.W = a, os.path.abspath(a.work)
        self.learn_dir = os.path.join(self.W, "learn")
        for d in (self.W, self.learn_dir, os.path.join(self.W, "logs")):
            os.makedirs(d, exist_ok=True)
        self.cards = a.cards.split(",")
        self.lock = threading.Lock()
        self.locks = a.lock_dir or (os.path.join(self.W, "logs") if a.dry_run else "/tmp")     # cross-lineage, one job per card
        self.poll = 0 if a.dry_run else 20
        self.commands, self.ticks, self.stop_after = [], 0, None
        self.ledger_path = os.path.join(self.W, "ledger.jsonl")
        rows = self.read_ledger()
        self.keys = {r.get("key") for r in rows}
        if os.path.exists(self.ledger_path) and open(self.ledger_path).read()[-1:] not in ("", "\n"):      # a kill inside a write
            open(self.ledger_path, "a").write("\n")
        self.ids = self.pool_ids()
        self.need = int(.9 * sum(half(i) == 1 for i in self.ids))    # a shorter result is not a result
        self.fake = Fake(self.ids)
        self.used_of = None

    def pool_ids(self):
        if os.path.exists(self.a.learn):
            return sorted({json.loads(l)["id"] for l in open(self.a.learn)})
        if not self.a.dry_run:
            raise SystemExit(f"no {self.a.learn}")
        return [f"{b}/{k}" for b in ("blink", "erqa", "cvbench", "babyvision") for k in range(100)]

    def tick(self):
        """The self-test's kill point; every file write and job passes here."""
        if self.stop_after is not None:
            with self.lock:
                self.ticks += 1
                if self.ticks > self.stop_after:
                    raise Stop()

    def put(self, path, obj):
        self.tick()
        put(path, obj)

    def say(self, text):
        line = f"[{time.strftime('%H:%M:%S')}] {text}"
        print(line, flush=True)
        with self.lock, open(os.path.join(self.W, "report.md"), "a") as f:
            f.write(line + "\n")

    def read_ledger(self):
        rows = []
        for l in open(self.ledger_path) if os.path.exists(self.ledger_path) else []:
            try:
                rows.append(json.loads(l))
            except ValueError:
                pass
        return rows

    def ledger(self, key, **row):
        """Each row once, also when a resumed step passes here again."""
        self.tick()
        with self.lock:
            if key in self.keys:
                return
            self.keys.add(key)
            with open(self.ledger_path, "a") as f:
                f.write(json.dumps(dict(key=key, mode=self.a.mode, **row, **({"dry_run": True} if self.a.dry_run else {})), ensure_ascii=False) + "\n")

    def used(self, card):
        if self.used_of:
            return self.used_of(card)
        if self.a.dry_run:
            return 0
        try:
            return int(subprocess.check_output(["nvidia-smi", "-i", card, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL))
        except (OSError, ValueError, subprocess.CalledProcessError):
            return 99999

    def take(self):
        """A card of ours with no job of any lineage on it and under 8 GiB used, for as long as that takes."""
        said = False
        while True:
            for c in self.cards:
                f = open(os.path.join(self.locks, f"mm_rsi_card{c}.lock"), "a")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    f.close()
                    continue
                if self.used(c) < FREE_MIB:
                    return c, f
                f.close()
            if not said:
                self.say(f"waiting for one of cards {self.a.cards} to be under {FREE_MIB} MiB")
                said = True
            time.sleep(self.poll)

    def alive(self, key):
        return subprocess.run(["pgrep", "-f", "python.*" + re.escape(key)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

    def launch(self, card, name, exe, key):
        """One job through the server's launcher, back when it has exited. key is a path only this job has on its command line; exe[:160] of mm_pipeline would not tell two arms apart here."""
        if self.a.start == "direct":                                # inside one job that holds the cards: same process group, so stopping that job stops this one
            cmd = (f"unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset $(env | grep -o '^TRITON_[A-Z_]*'); . {self.a.env_file}; "
                   f"export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={card} MPLBACKEND=Agg HF_HOME={self.a.root}/.hf VLLM_CACHE_ROOT={self.a.root}/.vllm-cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0; cd {CODE} && exec {exe}")
            with open(os.path.join(self.W, "logs", name + ".log"), "a") as f:
                subprocess.run(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT)
            return
        env = dict(os.environ, SE_FAMILY="qwen38", SE_MODEL=self.a.base, SE_LINEAGE=f"{CODE}/jobs/xf_n15.sh", SE_FRAC="0.92", SE_EXEC=exe)
        subprocess.run(["bash", self.a.start, card, name], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(24):
            if self.alive(key):
                break
            time.sleep(self.poll / 4)
        while self.alive(key):
            time.sleep(self.poll)

    def job(self, name, exe, done, key, fake):
        """Run exe on one card; done is the file that shows it worked. One more try if it is missing, then the run halts."""
        if os.path.exists(done):
            return
        if not self.a.dry_run and self.alive(key):                  # the launcher's jobs outlive a stopped harness: wait for it, no second copy
            self.say(f"{name} is still running from before, waiting")
            while self.alive(key):
                time.sleep(self.poll)
            if os.path.exists(done):
                return
        for _ in range(2):
            card, lock = self.take()
            try:
                if self.a.dry_run:
                    self.say(f"would run {name} on card {card}: {exe}")
                    with self.lock:
                        self.commands.append(name)
                    self.tick()
                    fake()
                else:
                    self.say(f"{name} on card {card}")
                    self.launch(card, name, exe, key)
            finally:
                lock.close()
            if os.path.exists(done):
                return
            self.say(f"{name} ended without {done}")
        raise Halt(f"{name} failed twice, no {done}")

    def par(self, *thunks):
        errs = []

        def run(t):
            try:
                t()
            except BaseException as ex:
                errs.append(ex)
        ts = [threading.Thread(target=run, args=(t,)) for t in thunks]
        [t.start() for t in ts]
        [t.join() for t in ts]
        if errs:
            raise errs[0]

    def loop_args(self):
        a = self.a                                                  # the settings of add_arms.py
        return (f"--mm --model {a.base} --pool {a.learn} --dirlib {a.dirlib} --seed 31 --tp 1 --max-model-len {a.max_model_len} --max-num-batched-tokens 4096 "
                f"--kv-cache-dtype fp8 --effort medium --budget {a.cut} --mb 128 --samples 2 --max-num-seqs 32 --gpu-util 0.85")

    def result(self, label):
        return os.path.join(self.learn_dir, f"{label}.learn.jsonl")

    def read(self, d, i, s):
        a, out, at = self.a, os.path.join(d, "read"), os.path.join(d, "at.json")
        if not os.path.exists(at):
            self.put(at, s["scales"])

        def fake():
            with open(out + ".base.npz", "w") as f:
                f.write("fake\n")
        self.job(f"{a.tag}_r{i}_read", f"python -u rsi/mm_adjoint.py --model {a.base} --dir {a.items} --tasks {s['focus'] or 'all'} --out {out} --at {at} --no-channels --draws {a.draws} --radius .1", out + ".base.npz", at, fake)

    def probe(self, label, engine, scales):
        spec, done = os.path.join(self.learn_dir, f"{label}.json"), self.result(label)
        if os.path.exists(done):
            return
        self.put(spec, [{"label": label, "effort": "medium", "engine": engine}])
        self.job(f"{self.a.tag}_{label}", f"python -u rsi/probe_base.py {self.learn_dir} --probes {spec} --sets learn:{self.a.learn} --curve 2048,4096,{self.a.cut} {self.loop_args()}", done, spec, lambda: self.fake.probe(done, scales))

    def propose(self, d, i, s, budgets):
        """mm_select on this round's read, then each arm composed with the state."""
        a, out = self.a, os.path.join(d, "arms.json")
        bs = ",".join(f"{b:g}" for b in budgets)
        if not os.path.exists(out + ".info"):
            cmd = [a.py, f"{CODE}/rsi/mm_select.py", os.path.join(d, "read.base.npz"), "--out", out, "--key", "smooth", "--half", "0", "--budgets", bs, "--prefix", f"r{i}", "--seed", str(a.seed)] + (["--cost", a.cost] if a.cost else [])
            if a.dry_run:
                self.say("would run " + " ".join(cmd))
                self.commands.append(f"r{i}_select --budgets {bs}")
                self.tick()
                self.fake.select(out, budgets, f"r{i}")
            else:
                with open(os.path.join(d, "select.txt"), "w") as f:
                    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="")).returncode
                if rc or not os.path.exists(out + ".info"):
                    raise Halt(f"mm_select failed in round {i}, see {d}/select.txt")
        arms, info = json.load(open(out)), json.load(open(out + ".info"))
        composed = {x["label"]: compose(s["scales"], x["engine"]) for x in arms}
        path = os.path.join(d, "composed.json")
        if not os.path.exists(path):
            self.put(path, {k: dict(scales=v, engine=engine_of(v)) for k, v in composed.items()})
        table = make_table(arms, info, a.table_rules.split(","))
        path = os.path.join(d, "table.json")
        if not os.path.exists(path):
            self.put(path, table)
        return table, json.load(open(os.path.join(d, "composed.json")))

    def benches(self):
        """What a read can focus on: the tasks of the items, with enough items on the selection half for mm_select."""
        path = os.path.join(self.a.items, "items.jsonl")
        pairs = [(p["task"], p["id"]) for p in map(json.loads, open(path))] if os.path.exists(path) else [(i.split("/")[0], i) for i in self.ids]
        n = {}
        for t, i in pairs:
            n[t] = n.get(t, 0) + (half(i) == 0)
        return sorted(t for t, c in n.items() if c >= FOCUS_MIN)

    def ledger_text(self, i):
        lines = []
        for r in self.read_ledger():
            if r.get("round", i) >= i:
                continue
            if r["step"] == "arm":
                lines.append(f"- round {r['round']}, {r['arm']} (rule {r['rule']}, budget {r['budget']:g}, {r['writers']} writers, predicted gain {r['predicted_gain']:+.3f}): measured {r['diff']:+.2f} points [{r['lo']:+.2f},{r['hi']:+.2f}] on {r['n']} problems, {r['decision']}")
            elif r["step"] == "choice" and r.get("fallback"):
                lines.append(f"- round {r['round']}: your reply could not be used, the harness's own rule chose instead")
            elif r["step"] == "choice" and r.get("rules"):
                lines += [f"- round {r['round']}: you wrote {x['name']} = {x['expr']}, budget {x['budget']:g}, clip {x['clip']:g}: {r['why']}" for x in r["rules"]]
            elif r["step"] == "choice" and r.get("focus"):
                lines.append(f"- round {r['round']}: you asked the next read to use only {r['focus']}: {r['why']}")
        return "\n".join(lines) or "none yet"

    def ask(self, d, i, s, table, pick):
        """The model mode. Its accuracy and its failures come from the selection half only: the other half decides, and it must not see that half's problems."""
        a, ask = self.a, os.path.join(d, "ask")
        os.makedirs(ask, exist_ok=True)
        reply, spec = os.path.join(ask, "reply.json"), os.path.join(ask, "spec.json")
        cur, traces = self.result(s["result"]), os.path.join(ask, "sel_traces.jsonl")
        if not os.path.exists(reply):
            if not os.path.exists(traces):
                self.tick()
                with open(traces + ".tmp", "w") as f:
                    for l in open(cur):
                        if half(json.loads(l)["id"]) == 0:
                            f.write(l)
                os.replace(traces + ".tmp", traces)
            acc = bench_means({k: sum(v) / len(v) for k, v in load_results(cur, a.cut, which=0).items()})
            shown = list(table)
            random.Random(a.seed * 1000 + i).shuffle(shown)         # no order to lean on
            benches = self.benches()
            acc_text = "\n".join(f"- {k}: {m:.3f} on {n} problems" for k, (m, n) in acc.items())
            read = os.path.join(d, "arms.json.summary")
            text = WRITE.format(min_gain=a.min_gain, floor=a.floor, names=" ".join(RULE_VARS[:7]), funcs=" ".join(RULE_FUNCS), ledger=self.ledger_text(i), acc=acc_text, digest=DIGEST, read=open(read).read() if os.path.exists(read) else "not available", pick=pick, benches=benches) if a.mode == "pipeline" else CHOOSE.format(min_gain=a.min_gain, floor=a.floor, ledger=self.ledger_text(i), acc=acc_text, digest=DIGEST,
                                 rules="\n".join(f"- {k}: {RULE_TEXT[k]}" for k in sorted({r["rule"] for r in table})),
                                 table="\n".join(f"- {r['label']}: rule {r['rule']}, budget {r['budget']:g}, {r['writers']} writers, predicted gain {r['predicted_gain']:+.3f}, cost {r['cost']:.4f}, top writers " + " ".join(f"{k} {x:+.3f}" for k, x in r["top"]) for r in shown),
                                 pick=pick, benches=benches)
            sp = dict(engine=engine_of(s["scales"]), traces=traces, digest=os.path.join(ask, "diagnosis.md"), run=os.path.join(ask, "diag"), cases=a.cases, prompt=text, prompt_file=os.path.join(ask, "prompt.txt"), reply=reply, labels=[r["label"] for r in table], benches=benches, pick=pick, think=a.think_tokens, effort="medium", kind="rule" if a.mode == "pipeline" else "choice")
            self.put(spec, sp)
            try:
                self.job(f"{a.tag}_r{i}_ask", f"python -u rsi/mm_rsi.py --ask-worker {spec} {self.loop_args()}", reply, spec, lambda: self.fake.ask(sp))
            except Halt as ex:
                return dict(picks=fixed_pick(table, pick), why="the harness's own rule, after a fallback", focus=None, fallback=f"no reply: {ex}", tries=0, raw=[])
        rep = json.load(open(reply))
        raw = [t["content"] for t in rep["tries"]]
        if rep["picks"] is None:
            return dict(picks=fixed_pick(table, pick), why="the harness's own rule, after a fallback", focus=None, fallback="; ".join(t["error"] for t in rep["tries"]), tries=len(raw), raw=raw)
        if a.mode == "pipeline":
            return self.own_arms(d, i, s, table, rep, raw, pick)
        return dict(picks=rep["picks"], why=rep["why"], focus=rep["focus"], fallback=None, tries=len(raw), raw=raw)

    def own_arms(self, d, i, s, table, rep, raw, pick):
        """The rules the model wrote become arms, added to this round's table and composed with the state. A rule that gives nothing to edit is dropped; if none is left the fixed rule chooses."""
        a, out = self.a, os.path.join(d, "own.json")
        rules = [dict(r, name=f"own{k + 1}") for k, r in enumerate(rep["picks"])]
        if not os.path.exists(out + ".info"):
            path = os.path.join(d, "own_rules.json")
            self.put(path, rules)
            cmd = [a.py, f"{CODE}/rsi/mm_select.py", os.path.join(d, "read.base.npz"), "--out", out, "--key", "smooth", "--half", "0", "--prefix", f"r{i}", "--seed", str(a.seed), "--expr", path] + (["--cost", a.cost] if a.cost else [])
            if a.dry_run:
                self.say("would run " + " ".join(cmd))
                self.tick()
                self.fake.own(out, rules, f"r{i}")
            else:
                with open(os.path.join(d, "own.txt"), "w") as f:
                    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="")).returncode
                if rc or not os.path.exists(out + ".info"):
                    raise Halt(f"mm_select --expr failed in round {i}, see {d}/own.txt")
        arms, info = json.load(open(out)), json.load(open(out + ".info"))
        own = make_table(arms, info, [r["name"] for r in rules])
        if not own:
            return dict(picks=fixed_pick(table, pick), why="the harness's own rule, after a fallback", focus=None, fallback="no written rule gave an edit: " + "; ".join(r["expr"] for r in rules), tries=len(raw), raw=raw, rules=rules)
        path, have = os.path.join(d, "table.json"), json.load(open(os.path.join(d, "table.json")))
        if not any(r["label"] == own[0]["label"] for r in have):
            self.put(path, have + own)
        path, have = os.path.join(d, "composed.json"), json.load(open(os.path.join(d, "composed.json")))
        if own[0]["label"] not in have:
            self.put(path, dict(have, **{x["label"]: dict(scales=(v := compose(s["scales"], x["engine"])), engine=engine_of(v)) for x in arms}))
        return dict(picks=[r["label"] for r in own], why=rep["why"], focus=rep["focus"], fallback=None, tries=len(raw), raw=raw, rules=rules)

    def choose(self, d, i, s, table):
        a, path = self.a, os.path.join(d, "choice.json")
        if not os.path.exists(path):
            pick, labels = min(a.pick, len(table)), [r["label"] for r in table]
            if not table:
                ch = dict(picks=[], why="no candidates", focus=None, fallback=None)
            elif a.mode == "fixed":
                ch = dict(picks=fixed_pick(table, a.pick), why="highest predicted gain, one per rule", focus=None, fallback=None)
            elif a.mode == "random":
                ch = dict(picks=random.Random(a.seed * 1000 + i).sample(labels, pick), why=f"seeded random pick, seed {a.seed}", focus=None, fallback=None)
            else:
                ch = self.ask(d, i, s, table, pick)
            self.put(path, ch)
        ch = json.load(open(path))
        self.ledger(f"r{i}/choice", step="choice", round=i, candidates=[(r["label"], r["predicted_gain"]) for r in table], same_as={x: r["label"] for r in table for x in r["same"]}, read_focus=s["focus"], **ch)
        self.say(f"round {i} choice: {ch['picks']}" + (f"; focus next {ch['focus']}" if ch["focus"] else "") + (f"; fallback: {ch['fallback']}" if ch["fallback"] else "") + f"; {ch['why']}")
        return ch

    def decide(self, d, i, s, table, composed, ch, bs):
        """Paired on the half not used for selection. The decision file comes first; ledger rows pass once, then the state moves."""
        a, path = self.a, os.path.join(d, "decision.json")
        if not os.path.exists(path):
            info, base, rows = {r["label"]: r for r in table}, load_results(self.result(s["result"]), a.cut, which=1), []
            for l in ch["picks"]:
                arm = load_results(self.result(l), a.cut, which=1)
                st = paired(base, arm)
                rows.append(dict(arm=l, rule=info[l]["rule"], budget=info[l]["budget"], writers=info[l]["writers"], predicted_gain=info[l]["predicted_gain"], **st, by_bench=bench_gain(base, arm), fail=verdict(st, a.floor, a.min_gain, self.need)))
            good = [r for r in rows if not r["fail"]]
            best = max(good, key=lambda r: r["diff"]) if good else None
            for r in rows:
                r["decision"] = "accepted" if r is best else f"rejected: {r['fail'] or 'passed, another arm had the higher mean'}"
            after = dict(s, round=i, focus=ch["focus"])
            if best:
                after.update(scales=composed[best["arm"]]["scales"], result=best["arm"], no_accept=0, accepted=s["accepted"] + [dict(round=i, arm=best["arm"], diff=best["diff"], lo=best["lo"], hi=best["hi"])])
            else:
                after.update(no_accept=s["no_accept"] + 1, budget_factor=s["budget_factor"] * .5)
            if after["no_accept"] >= a.patience:
                after["stopped"] = f"{after['no_accept']} rounds in a row with no acceptance (patience {a.patience})"
            self.put(path, dict(rows=rows, accepted=best["arm"] if best else None, after=after))
        dec = json.load(open(path))
        after = dec["after"]
        for r in dec["rows"]:
            self.ledger(f"r{i}/arm/{r['arm']}", step="arm", round=i, **r)
            self.say(f"round {i} {r['arm']}: {r['diff']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}] n={r['n']}, predicted {r['predicted_gain']:+.3f}, {r['decision']}")
        self.ledger(f"r{i}/decision", step="decision", round=i, accepted=dec["accepted"], budgets=bs, scales_before=s["scales"], scales_after=after["scales"], no_accept=after["no_accept"], next_budget_factor=after["budget_factor"], next_focus=after["focus"])
        if after.get("stopped"):
            self.ledger(f"r{i}/stop", step="stop", round=i, reason=after["stopped"])
            self.say(f"stopped after round {i}: {after['stopped']}")
        self.put(os.path.join(self.W, "state.json"), after)
        return after

    def round(self, s):
        a, i = self.a, s["round"] + 1
        d = os.path.join(self.W, f"r{i}")
        os.makedirs(d, exist_ok=True)
        budgets = [float(x) * s["budget_factor"] for x in a.budgets.split(",")]
        bs = ",".join(f"{b:g}" for b in budgets)
        self.say(f"round {i}: {len(s['scales'])} writers edited, budgets {bs}, read on {s['focus'] or 'all benches'}")
        self.par(lambda: self.read(d, i, s), lambda: self.probe(s["result"], engine_of(s["scales"]), s["scales"]))   # the state's own result, for round 1 the plain base; the model's digest needs it before the choice
        table, composed = self.propose(d, i, s, budgets)
        ch = self.choose(d, i, s, table)
        if a.mode == "pipeline":                                    # the written rules joined the table
            table, composed = json.load(open(os.path.join(d, "table.json"))), json.load(open(os.path.join(d, "composed.json")))
        self.par(*[(lambda l=l: self.probe(l, composed[l]["engine"], composed[l]["scales"])) for l in ch["picks"]])
        return self.decide(d, i, s, table, composed, ch, bs)

    def final(self, s):
        """The final arm, ready for a --probes file, and a table per round."""
        a, path = self.a, os.path.join(self.W, "final.json")
        out = dict(label=f"{a.tag}_final", effort="medium", engine=engine_of(s["scales"]), scales=s["scales"], mode=a.mode, rounds=s["round"], accepted=s["accepted"], stopped=s["stopped"], **({"dry_run": True} if a.dry_run else {}))
        if os.path.exists(path) and json.load(open(path)) == out:
            return
        rows = self.read_ledger()
        lines = ["", f"## Rounds, mode {a.mode}" + (" (DRY RUN, FAKE NUMBERS)" if a.dry_run else ""), ""]
        for i in range(1, s["round"] + 1):
            ch = next(r for r in rows if r["key"] == f"r{i}/choice")
            dec = next(r for r in rows if r["key"] == f"r{i}/decision")
            lines += [f"Round {i}: budgets {dec['budgets']}, read on {ch['read_focus'] or 'all benches'}, chose by {a.mode}" + (f" (fallback to the fixed rule: {ch['fallback']})" if ch.get("fallback") else "") + f", {ch['why']}" + (f"; asked the next read to use {ch['focus']}" if ch["focus"] else ""), "",
                      "| arm | rule | budget | writers | predicted | gain | lo | hi | n | decision |", "|---|---|---|---|---|---|---|---|---|---|"]
            lines += [f"| {r['arm']} | {r['rule']} | {r['budget']:g} | {r['writers']} | {r['predicted_gain']:+.3f} | {r['diff']:+.2f} | {r['lo']:+.2f} | {r['hi']:+.2f} | {r['n']} | {r['decision']} |" for r in rows if r["step"] == "arm" and r["round"] == i]
            lines += [""]
        lines += [f"Final: {len(s['scales'])} writers edited after {s['round']} rounds, {len(s['accepted'])} accepted" + (f"; stopped: {s['stopped']}" if s["stopped"] else "") + ". Measured on the learn set only; held-out not run here.", ""]
        self.tick()
        with self.lock, open(os.path.join(self.W, "report.md"), "a") as f:
            f.write("\n".join(lines))
        self.put(path, out)
        self.say(f"wrote {path}")

    def state(self):
        a, path = self.a, os.path.join(self.W, "state.json")
        cfg = dict(mode=a.mode, base=a.base, learn=a.learn, items=a.items, budgets=a.budgets, pick=a.pick, floor=a.floor, min_gain=a.min_gain, cut=a.cut, seed=a.seed, draws=a.draws, table_rules=a.table_rules, cost=a.cost, dry_run=a.dry_run)
        if os.path.exists(path):
            s = json.load(open(path))
            bad = {k: (s["config"].get(k), v) for k, v in cfg.items() if s["config"].get(k) != v}
            if bad:                                                 # a lineage has one set of rules, and fake files must never pass for real ones
                raise SystemExit(f"{a.work} was started with other settings (was, now): {bad}")
            return s
        s = dict(round=0, scales={}, result="base", focus=None, budget_factor=1.0, no_accept=0, accepted=[], stopped=None, config=cfg)
        self.put(path, s)
        return s

    def run(self):
        a = self.a
        s = self.state()
        self.say(f"mode {a.mode}, round {s['round']} of {a.rounds} done, {len(s['scales'])} writers edited" + (", DRY RUN, FAKE NUMBERS" if a.dry_run else ""))
        try:
            while s["round"] < a.rounds and not s["stopped"]:
                s = self.round(s)
            self.final(s)
        except Halt as ex:
            self.ledger(f"halt/{s['round']}/{time.time()}", step="halt", after_round=s["round"], reason=str(ex))
            self.say(f"halted: {ex}. Nothing was decided on it; run the same command to resume.")
            return False
        return True


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--mode", required=True, choices=["fixed", "model", "pipeline", "random"])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--cards", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--learn", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--budgets", default=".02,.1,.4")
    ap.add_argument("--pick", type=int, default=2)
    ap.add_argument("--floor", type=float, default=-0.5)
    ap.add_argument("--min-gain", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=2)              # rounds in a row with no acceptance
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--cut", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cost")                                       # mm_select's writer to KL file; without it natural is the index
    ap.add_argument("--table-rules", default=",".join(FIXED_RULES))  # what the model and the random control choose from; add flip,random to offer the controls too
    ap.add_argument("--cases", type=int, default=120)               # failures self_diagnose reads
    ap.add_argument("--think-tokens", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=36864)
    ap.add_argument("--tag")
    ap.add_argument("--root", default=N)
    ap.add_argument("--dirlib")
    ap.add_argument("--py")
    ap.add_argument("--start")                                      # a script called as SCRIPT CARD NAME with SE_* in the environment, or direct
    ap.add_argument("--env-file")                                   # sourced before a direct job
    ap.add_argument("--lock-dir")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def finish_args(a):
    a.base, a.learn, a.items = os.path.abspath(a.base), os.path.abspath(a.learn), os.path.abspath(a.items)
    a.tag = a.tag or f"rsi_{a.mode}"
    a.dirlib = a.dirlib or f"{a.root}/xf/low/qwen38/library"
    a.py = a.py or f"{a.root}/venv/bin/python"
    a.start = a.start or f"{a.root}/code/jobs/n15_start.sh"
    a.env_file = a.env_file or f"{a.root}/venv/bin/activate"
    if any(f"{os.sep}sets{os.sep}" in p for p in (a.learn, a.items)):
        raise SystemExit("--learn and --items must not be held-out sets")
    if any(" " in str(p) for p in (a.work, a.base, a.learn, a.items, a.dirlib)):
        raise SystemExit("no spaces in paths: the launcher word-splits SE_EXEC")
    return a


def self_test(root):
    """Everything the dry run can show, with fake numbers. Leaves its files in root."""
    fails = []

    def check(name, ok, extra=""):
        print(("ok   " if ok else "FAIL ") + name + (f"  [{extra}]" if extra else ""), flush=True)
        if not ok:
            fails.append(name)

    def raises(fn, *x):
        try:
            fn(*x)
        except ValueError:
            return True
        return False
    fix = os.path.join(root, "fix", "items")
    os.makedirs(fix, exist_ok=True)
    learn = os.path.join(root, "fix", "pool.jsonl")
    ids = [f"{b}/{k}" for b in ("blink", "erqa", "cvbench") for k in range(40)] + [f"tiny/{k}" for k in range(8)]
    with open(learn, "w") as f, open(os.path.join(fix, "items.jsonl"), "w") as g:
        for i in ids:
            f.write(json.dumps(dict(id=i, bench=i.split("/")[0], kind="mc", prompt="p", answer="A")) + "\n")
            g.write(json.dumps(dict(id=i, task=i.split("/")[0])) + "\n")
    count = [0]

    def go(mode, rounds=3, setup=None, stop=None, work=None, extra=()):
        count[0] += 1
        w = work or os.path.join(root, f"w{count[0]}_{mode}")
        a = finish_args(parser().parse_args([w, "--mode", mode, "--rounds", str(rounds), "--cards", "4,5", "--base", "/x/MODEL", "--learn", learn, "--items", fix, "--dry-run", *extra]))
        R = Rsi(a)
        if setup:
            setup(R)
        R.stop_after = stop
        out, killed, ok = io.StringIO(), False, None
        with contextlib.redirect_stdout(out):
            try:
                ok = R.run()
            except Stop:
                killed = True
        R.out, R.killed, R.ok, R.w = out.getvalue(), killed, ok, w
        return R

    def steps(R, step):
        return [r for r in R.read_ledger() if r["step"] == step]

    def state_of(w):
        return json.load(open(os.path.join(w, "state.json")))

    def flat(g):                                                    # a fake world where only the budget of an arm matters: exact gains, by label
        def setup(R):
            R.fake.gain = g
        return setup

    # scale composition
    old = {"v1.attn": 1.2, "t3.attn": .9}
    arm = engine_one("v1.attn", 1.1) + engine_one("v2.mlp", .8) + engine_one("t4.attn", 1.05) + engine_one("t5.mlp", 1.01)
    new = compose(old, arm)
    check("compose: old times new, untouched writers keep their scale", new == {"v1.attn": 1.32, "t3.attn": .9, "v2.mlp": .8, "t4.attn": 1.05, "t5.mlp": 1.01}, new)
    check("compose: no change to the old dict", old == {"v1.attn": 1.2, "t3.attn": .9})
    check("compose: back to 1 leaves the writer out", compose({"t0.attn": .5}, engine_one("t0.attn", 2.0)) == {})
    check("compose: engine_of then compose on nothing is the identity", compose({}, engine_of(new)) == new)
    check("engine names as mm_select: L % 4 == 3 is self_attn", engine_one("t3.attn", 1.5)[0]["param"] == "layers.3.self_attn.o_proj" and engine_one("t4.attn", 1.5)[0]["param"] == "layers.4.linear_attn.out_proj")
    check("engine names as mm_select: vision weight and bias", [e["param"] for e in engine_one("v2.mlp", 1.5)] == ["visual.blocks.2.mlp.linear_fc2.weight", "visual.blocks.2.mlp.linear_fc2.bias"])
    check("compose: an edit it cannot name is an error", raises(compose, {}, [{"op": "scale", "param": "layers.1.input_layernorm", "alpha": 2.0}]))
    check("compose: another op is an error", raises(compose, {}, [{"op": "add", "param": "layers.0.linear_attn.out_proj", "value": 1.0}]))
    check("compose: weight and bias that differ are an error", raises(compose, {}, [{"op": "scale", "param": "visual.blocks.0.attn.proj.weight", "alpha": 1.1}, {"op": "scale", "param": "visual.blocks.0.attn.proj.bias", "alpha": 1.2}]))
    # small rules
    path = os.path.join(root, "fix", "cut.jsonl")
    with open(path, "w") as f:
        f.write("".join(json.dumps(dict(id=i, ok=o, tokens=t)) + "\n" for i, o, t in (("a/1", True, 100), ("a/1", True, 9000), ("a/1", False, 100))))
    check("results: right within the cut only", load_results(path, 8192)["a/1"] == [True, False, False])
    odd = sorted(i for i in ids if half(i) == 1)
    check("halves: same crc32 as mm_select and cmp_learn", all(half(i) == zlib.crc32(i.encode()) % 2 for i in ids) and 0 < len(odd) < len(ids))
    st = dict(n=100, diff=.51, lo=-.49)
    check("accept: mean and lower bound must both clear", verdict(st, -.5, .5, 90) is None and verdict(dict(st, diff=.5), -.5, .5, 90) and verdict(dict(st, lo=-.5), -.5, .5, 90) and verdict(dict(st, n=89), -.5, .5, 90))
    table = [dict(label=f"r1_{rule}_{b:g}", rule=rule, predicted_gain=p) for rule, b, p in (("raw", .1, 1.), ("raw", .4, 3.), ("index", .4, 2.), ("bang", .1, .5), ("natural", .1, .4), ("flip", .4, 9.), ("random", .4, 8.))]
    check("fixed rule: best of each rule, top two, no controls", fixed_pick(table, 2) == ["r1_raw_0.4", "r1_index_0.4"], fixed_pick(table, 2))
    # the reply parser
    L, B = ["a_x_1", "b_y_1", "c_z_1"], ["blink", "erqa"]
    good = '{"picks": ["a_x_1", "c_z_1"], "why": "two\\nlines", "focus": "blink"}'
    check("parse: good reply, why made one line", parse_choice(good, L, B, 2) == ((["a_x_1", "c_z_1"], "two lines", "blink"), None))
    check("parse: JSON after words and null focus", parse_choice('sure: {"picks": ["a_x_1", "b_y_1"], "why": "w", "focus": null}', L, B, 2)[0] == (["a_x_1", "b_y_1"], "w", None))
    for name, text in (("no JSON", "I pick a"), ("empty", ""), ("one pick", '{"picks": ["a_x_1"], "why": "w"}'), ("twice the same", '{"picks": ["a_x_1", "a_x_1"], "why": "w"}'), ("not in the table", '{"picks": ["a_x_1", "q"], "why": "w"}'), ("no why", '{"picks": ["a_x_1", "b_y_1"]}'), ("bad focus", '{"picks": ["a_x_1", "b_y_1"], "why": "w", "focus": "nope"}'), ("cut off", '{"picks": ["a_x_1",')):
        check(f"parse: {name} is refused", parse_choice(text, L, B, 2)[0] is None)
    # all three modes, whole loop
    for mode in ("fixed", "model", "random"):
        R = go(mode, 4)
        s, fin = state_of(R.w), json.load(open(os.path.join(R.w, "final.json")))
        chain = {}
        for x in s["accepted"]:
            arms = {y["label"]: y for y in json.load(open(os.path.join(R.w, f"r{x['round']}", "arms.json")))}
            chain = compose(chain, arms[x["arm"]]["engine"])
        report = open(os.path.join(R.w, "report.md")).read()
        check(f"{mode}: finishes, each round in the ledger", R.ok and len(steps(R, "decision")) == s["round"] and len(steps(R, "choice")) == s["round"], f"rounds {s['round']}, accepted {len(s['accepted'])}, picks {[c['picks'] for c in steps(R, 'choice')][:1]}")
        check(f"{mode}: final scales are the accepted arms composed in order", fin["scales"] == s["scales"] == {k: v for k, v in chain.items()} and compose({}, fin["engine"]) == fin["scales"])
        check(f"{mode}: rejected arms and numbers in the ledger", all("diff" in r and "lo" in r and "hi" in r and "decision" in r for r in steps(R, "arm")) and len(steps(R, "arm")) == 2 * s["round"])
        check(f"{mode}: report has a table per round, marked fake", report.count("| arm | rule |") == s["round"] and "FAKE NUMBERS" in report and all(r.get("dry_run") for r in R.read_ledger()))
        check(f"{mode}: never loads numpy", "numpy" not in sys.modules)
    rows = steps(go("random", 2, setup=flat(lambda sc: 0.0)), "choice")
    again = steps(go("random", 2, setup=flat(lambda sc: 0.0)), "choice")
    check("random: seeded, the same picks twice", [r["picks"] for r in rows] == [r["picks"] for r in again], rows[0]["picks"])
    check("random: from the theory table, one seed per round", all(p.split("_")[1] in FIXED_RULES for r in rows for p in r["picks"]) and rows[0]["picks"] != [p.replace("r2", "r1") for p in rows[1]["picks"]])
    other = steps(go("random", 1, setup=flat(lambda sc: 0.0), extra=["--seed", "7", "--table-rules", "index,raw,bang,natural,flip,random"]), "choice")[0]
    check("--table-rules offers the controls to the choosers", len(other["candidates"]) == 15 and len(rows[0]["candidates"]) == 9)
    check("identical arms are one candidate, so the fixed rule never tests one arm twice", rows[0]["same_as"] == {f"r1_natural_{b}": f"r1_index_{b}" for b in ("0.02", "0.1", "0.4")} and not any(c[0].split("_")[1] == "natural" for c in rows[0]["candidates"]))
    arm = [{"label": "r1_index_0.1", "engine": engine_one("t0.attn", 1.1)}, {"label": "r1_natural_0.1", "engine": engine_one("t0.attn", 1.1)}, {"label": "r1_bang_0.1", "engine": []}, {"label": "r1_raw_0.1", "engine": engine_one("t1.attn", 1.1)}]
    x = dict(writers=1, predicted_gain=1.0, cost=.1, doses={"t0.attn": .1})
    check("table: an empty arm and a NaN prediction are left out", [r["label"] for r in make_table(arm, {"r1_index_0.1": x, "r1_natural_0.1": x, "r1_bang_0.1": dict(x, writers=0), "r1_raw_0.1": dict(x, predicted_gain=float("nan"))}, FIXED_RULES)] == ["r1_index_0.1"])
    # model mode: what it sees, focus, retry, fallback
    seen = []

    def by_text(replies):
        def setup(R):
            def reply(text, effort):
                seen.append((text, effort))
                out = replies.pop(0)
                return out(text) if callable(out) else out
            R.fake.reply = reply
        return setup

    def two(text, focus="null", why="w"):
        lab = re.findall(r"^- (r\d+_\w+_[\d.]+):", text, re.M)
        return '{"picks": ["%s", "%s"], "why": "%s", "focus": %s}' % (lab[0], lab[1], why, focus)
    R = go("model", 2, setup=by_text([lambda t: two(t, '"blink"', "blink is worst"), two]))
    reads = [c for c in R.out.splitlines() if "mm_adjoint" in c]
    check("focus: asked in round 1, the read of round 2 only", "--tasks all" in reads[0] and "--tasks blink" in reads[1] and steps(R, "choice")[0]["focus"] == "blink" and steps(R, "choice")[1]["read_focus"] == "blink")
    check("focus: choice and reason in the ledger", steps(R, "choice")[0]["why"] == "blink is worst" and "blink is worst" in steps(R, "choice")[0]["raw"][0])
    p1, p2 = seen[0][0], seen[1][0]
    check("prompt 1: bench accuracy, digest, table, no ledger yet", "none yet" in p1 and "blink:" in p1 and "Diagnosis: fake" in p1 and DIGEST not in p1 and "predicted gain" in p1)
    check("prompt 2: its ledger with the measured gain and interval, and its focus", "round 1, r1_" in p2 and "measured" in p2 and "[" in p2 and "you asked the next read to use only blink" in p2)
    check("prompt: a bench too small for a read cannot be asked for", "tiny" not in p1.split("Choose exactly")[1])
    sel = os.path.join(R.w, "r1", "ask", "sel_traces.jsonl")
    check("model sees the selection half only: traces", {half(json.loads(l)["id"]) for l in open(sel)} == {0})
    n0 = sum(half(i) == 0 and i.startswith("blink") for i in ids)
    check("model sees the selection half only: accuracy", f"on {n0} problems" in p1, n0)
    spec = json.load(open(os.path.join(R.w, "r2", "ask", "spec.json")))
    accepted = state_of(R.w)["accepted"]
    check("the checkpoint it runs on has the current scales", spec["engine"] == engine_of(json.load(open(os.path.join(R.w, "r2", "at.json")))), f"{len(spec['engine'])} edits, accepted so far {len(accepted)}")
    seen.clear()
    R = go("model", 1, setup=by_text(["garbage", two]))
    ch = steps(R, "choice")[0]
    check("retry: one bad reply, the second is used, raw kept", not ch["fallback"] and ch["tries"] == 2 and ch["raw"][0] == "garbage" and seen[1][1] == "low" and "could not be used" in seen[1][0])
    R = go("model", 1, setup=by_text(["garbage", "{still not"]))
    ch = steps(R, "choice")[0]
    fixed_now = fixed_pick(json.load(open(os.path.join(R.w, "r1", "table.json"))), 2)
    check("fallback: two bad replies, the fixed rule, recorded with both raw replies", ch["fallback"] and ch["picks"] == fixed_now and ch["raw"] == ["garbage", "{still not"] and ch["focus"] is None and "fallback" in open(os.path.join(R.w, "report.md")).read())

    def no_reply(R):
        R.fake.ask = lambda sp: None
    R = go("model", 1, setup=no_reply)
    ch = steps(R, "choice")[0]
    check("fallback: the job leaves no reply, one more try, then the fixed rule", R.ok and "no reply" in ch["fallback"] and ch["picks"] == fixed_pick(json.load(open(os.path.join(R.w, "r1", "table.json"))), 2) and R.commands.count("rsi_model_r1_ask") == 2)
    # accept, reject, shrink, stop
    R = go("fixed", 5, setup=flat(lambda sc: 0.0))
    s, sel = state_of(R.w), [c for c in R.commands if "select" in c]
    check("no gain: state stays, budgets halve, stops at patience 2 with the reason", s["round"] == 2 and s["scales"] == {} and s["result"] == "base" and s["stopped"].startswith("2 rounds") and steps(R, "stop") and sel == ["r1_select --budgets 0.02,0.1,0.4", "r2_select --budgets 0.01,0.05,0.2"], sel)
    check("no gain: every arm rejected on the mean, final arm has no edits", all("not above" in r["decision"] for r in steps(R, "arm")) and json.load(open(os.path.join(R.w, "final.json")))["engine"] == [])
    R = go("fixed", 3, setup=flat(lambda sc: len(sc) * 1.0))
    s = state_of(R.w)
    check("gain: state moves each round, baseline is the accepted arm, base generated once", len(s["accepted"]) == 3 and s["result"] == s["accepted"][-1]["arm"] and R.commands.count("rsi_fixed_base") == 1 and R.commands.count("rsi_fixed_" + s["accepted"][0]["arm"]) == 1, [x["arm"] for x in s["accepted"]])
    rows = steps(R, "arm")
    check("gain: the better arm of a round wins, the other is recorded as passed", all(sum(r["decision"] == "accepted" for r in rows if r["round"] == i) == 1 for i in (1, 2, 3)) and any("another arm" in r["decision"] for r in rows))

    def flip_flop(sc):                                              # accepted, then two rounds of nothing: the count is of rounds in a row
        return 5.0 if any(x > 1 for x in sc.values()) else 0.0
    R = go("fixed", 6, setup=flat(flip_flop))
    s = state_of(R.w)
    check("patience: resets on acceptance", len(s["accepted"]) == 1 and s["round"] == 3 and s["no_accept"] == 2, f"round {s['round']}")
    R = go("fixed", 1, setup=flat(lambda sc: -3.0 if sc else 0.0))
    check("loss: rejected on the lower bound", all("lower bound" in r["decision"] for r in steps(R, "arm")), steps(R, "arm")[0]["diff"])

    def empty(R):                                                   # short result: a job that stopped early cannot be accepted
        real = R.fake.probe

        def short(path, sc):
            real(path, sc)
            if sc:
                lines = open(path).readlines()
                open(path, "w").writelines(lines[:len(lines) // 2])
        R.fake.probe = short
        R.fake.gain = lambda sc: 20.0 if sc else 0.0
    R = go("fixed", 1, setup=empty)
    check("short result: too few problems, rejected", all("only" in r["decision"] for r in steps(R, "arm")), steps(R, "arm")[0]["decision"])
    # halt on a job that cannot be done, and resume
    w = os.path.join(root, "halt_fixed")

    def broken(R):
        R.fake.probe = lambda path, sc: None
    R = go("fixed", 2, setup=broken, work=w)
    check("halt: a job that fails twice stops the run, nothing decided", R.ok is False and state_of(w)["round"] == 0 and steps(R, "halt") and not steps(R, "decision") and not os.path.exists(os.path.join(w, "final.json")))
    R = go("fixed", 2, work=w)
    check("halt: the same command resumes and finishes", R.ok and state_of(w)["round"] == 2)
    # resume: killed at every point of a run, then the same command: same result as a run with no kill, no finished job run again
    for mode in ("model", "fixed", "random"):
        def clean(w):
            rows = [{k: v for k, v in r.items()} for r in map(json.loads, open(os.path.join(w, "ledger.jsonl")))]
            return json.dumps([rows, state_of(w), json.load(open(os.path.join(w, "final.json")))], sort_keys=True)
        ref = go(mode, 3, stop=10 ** 9)
        total, want, bad, twice = ref.ticks, clean(ref.w), [], 0
        cuts = list(range(total))
        for k in cuts:
            A = go(mode, 3, stop=k, work=os.path.join(root, f"kill_{mode}_{k}"))
            B = go(mode, 3, work=A.w)
            if not A.killed or not B.ok or clean(A.w) != want:
                bad.append(k)
            twice = max(twice, len(set(A.commands) & set(B.commands)))
        check(f"resume, {mode}: killed at each of {len(cuts)} points of {total}, same ledger, state, final as an unbroken run", not bad, f"differs at {bad}" if bad else "")
        check(f"resume, {mode}: only jobs caught by the kill run again", twice <= 2, f"at most {twice} at once, cards 2")
        B = go(mode, 3, work=ref.w)
        check(f"resume, {mode}: a finished run runs nothing", B.commands == [] and clean(ref.w) == want)
    # settings of a lineage
    R0 = go("fixed", 1)

    def refused(*extra, mode="fixed"):
        try:
            go(mode, 1, work=R0.w, extra=extra)
        except SystemExit as ex:
            return "other settings" in str(ex)
        return False
    check("resume refused with another mode or rule", refused(mode="random") and refused("--floor", "1") and refused("--budgets", ".5"))
    real = finish_args(parser().parse_args([R0.w, "--mode", "fixed", "--cards", "4", "--base", "/x/MODEL", "--learn", learn, "--items", fix]))
    try:
        Rsi(real).state()
        check("a real run refuses a dry run's WORK", False)
    except SystemExit as ex:
        check("a real run refuses a dry run's WORK", "dry_run" in str(ex))
    try:
        finish_args(parser().parse_args(["w", "--mode", "fixed", "--cards", "4", "--base", "M", "--learn", "/n/data/mm/sets/blink.jsonl", "--items", fix]))
        check("a held-out path is refused", False)
    except SystemExit:
        check("a held-out path is refused", True)
    # the commands, against add_arms.py and mk_q1.py
    R = go("model", 1, extra=["--root", "/data/fd/mingdazhang/selfedit"])
    out = R.out
    arg = "--curve 2048,4096,8192 --mm --model /x/MODEL --pool %s --dirlib /data/fd/mingdazhang/selfedit/xf/low/qwen38/library --seed 31 --tp 1 --max-model-len 36864 --max-num-batched-tokens 4096 --kv-cache-dtype fp8 --effort medium --budget 8192 --mb 128 --samples 2 --max-num-seqs 32 --gpu-util 0.85" % learn
    check("probe command is add_arms.py's, paths from --root", arg in out and f"--sets learn:{learn}" in out and "rsi/probe_base.py" in out)
    check("read command: --at, thinking off, no channels, draws", re.search(r"rsi/mm_adjoint\.py --model /x/MODEL --dir \S+ --tasks all --out \S+/r1/read --at \S+/r1/at\.json --no-channels --draws 2 --radius \.1", out))
    check("select command: half 0, smooth key, prefix", re.search(r"/data/fd/mingdazhang/selfedit/venv/bin/python \S+/rsi/mm_select\.py \S+/read\.base\.npz --out \S+/arms\.json --key smooth --half 0 --budgets 0\.02,0\.1,0\.4 --prefix r1", out))
    arm = json.load(open(os.path.join(R.w, "learn", steps(R, "arm")[0]["arm"] + ".json")))
    check("probe spec is one arm with its composed engine", len(arm) == 1 and set(arm[0]) == {"label", "effort", "engine"} and arm[0]["effort"] == "medium")
    # cards
    R = go("fixed", 1)
    c1, f1 = R.take()
    c2, f2 = R.take()
    got = []
    t = threading.Thread(target=lambda: got.append(R.take()))
    with contextlib.redirect_stdout(io.StringIO()):
        t.start()
        time.sleep(.3)
        waited = not got
        f1.close()
        t.join(5)
    check("cards: one job per card, a third waits until one is free", {c1, c2} == {"4", "5"} and waited and got and got[0][0] == c1)
    f2.close()
    got[0][1].close()
    reads = iter([20000, 9000, 8000, 7999])
    R.used_of = lambda c: next(reads)
    R.cards = ["4"]
    with contextlib.redirect_stdout(io.StringIO()):
        c, f = R.take()
    f.close()
    check("cards: waits for under 8 GiB", next(reads, "gone") == "gone")
    # the job itself, with a stub engine and the real self_diagnose.py
    log, R = [], go("model", 1)
    d = os.path.join(root, "worker")
    os.makedirs(d, exist_ok=True)
    sp = json.load(open(os.path.join(R.w, "r1", "ask", "spec.json")))
    sp.update(digest=os.path.join(d, "diagnosis.md"), reply=os.path.join(d, "reply.json"), prompt_file=os.path.join(d, "prompt.txt"), run=os.path.join(d, "diag"), engine=engine_one("v1.attn", 1.1))
    put(os.path.join(d, "spec.json"), sp)

    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return types.SimpleNamespace(input_ids=text.split())

        def decode(self, x):
            return " ".join(x)

    class Stub:
        def __init__(self, model, **k):
            log.append(("load", model, k.get("multimodal")))
            self.tokenizer = Tok()

        def apply(self, edits):
            log.append(("apply", edits))

        def generate(self, prompts, n=1, sampling=None, max_tokens=0, reasoning_effort=None):
            log.append(("generate", len(prompts)))
            out = []
            for p in prompts:
                text = p if isinstance(p, str) else p[0]["content"]
                text = text if isinstance(text, str) else str(text)
                c = ('{"modes": [{"name": "fake mode", "definition": "d", "indices": [0, 1]}]}' if "Merge them" in text else '{"picks": ["%s", "%s"], "why": "w", "focus": null}' % tuple(sp["labels"][:2]) if "Choose exactly" in text else '{"mode": "fake", "what_happened": "x", "first_bad_line": "", "better_line": "", "avoidable": true}')
                out.append(dict(samples=[dict(content=c, reasoning="", finish_reason="stop", num_tokens=5)]))
            return out
    loop = types.ModuleType("rsi.loop")
    loop.args = types.SimpleNamespace(model="/x/MODEL", tp=1, max_model_len=1, max_num_seqs=1, max_num_batched_tokens=1, gpu_util=.1, kv_cache_dtype="fp8", mm=False, pool=[learn], seed=0)
    loop.SAMPLING, loop.say, loop.load_jsonl = dict(temperature=1.0), lambda *x: None, lambda paths: [json.loads(l) for p in paths for l in open(p)]
    surg = types.ModuleType("surgery")
    surg.Surgeon, surg.image_message = Stub, lambda text, images: [{"role": "user", "content": text}]
    sys.modules.update({"rsi.loop": loop, "surgery": surg})
    argv = sys.argv
    with contextlib.redirect_stdout(io.StringIO()):
        ask_worker(os.path.join(d, "spec.json"), ["--mm"])
    rep = json.load(open(sp["reply"]))
    check("job: scales applied right after load, before any generation", log[0][0] == "load" and log[1] == ("apply", sp["engine"]) and all(x[0] == "generate" for x in log[2:]) and len(log) > 3, [x[0] for x in log][:5])
    check("job: real self_diagnose.py wrote the digest from the selection half, then the choice parsed", "fake mode" in open(sp["digest"]).read() and rep["picks"] == sp["labels"][:2] and "fake mode" in open(sp["prompt_file"]).read())
    log.clear()
    os.replace(sp["reply"], sp["reply"] + ".first")
    surg.Surgeon = Stub                                             # a new job is a new process
    with contextlib.redirect_stdout(io.StringIO()):
        ask_worker(os.path.join(d, "spec.json"), ["--mm"])
    check("job: on resume with the digest there, one load, scales applied, no second diagnosis", [x[0] for x in log] == ["load", "apply", "generate"] and os.path.exists(sp["reply"]))
    sys.argv = argv
    # the written rules
    check("rule: arithmetic on the read passes", all(check_expr(x) is None for x in ("rel*g/-h", "(abs(z)>2.5)*sign(g)", "where(tower>0, g, g*.5)", "tanh(z)*(1-pos)**2", "g/quantile(abs(g), .9)")))
    check("rule: anything else is refused", all(check_expr(x) for x in ("__import__('os')", "g.__class__", "g[0]", "(lambda: 1)()", "'a'", "9**9**9", "g**g", "open(g)", "foo", "sign(g, out=g)", "", "g" * 400)))
    good = '{"rules": [{"expr": "rel*g/-h", "budget": 0.1, "clip": 0.3}, {"expr": "sign(g)", "budget": 0.02}], "why": "w", "focus": null}'
    got, err = parse_rules(good, ["blink"], 2)
    check("rule: reply parsed, clip defaults", err is None and got[0][1] == dict(expr="sign(g)", budget=.02, clip=.45), err)
    for bad, why in (('{"rules": [{"expr": "g", "budget": 0.1}], "why": "w", "focus": null}', "one rule"), ('{"rules": [{"expr": "g", "budget": 9}, {"expr": "h", "budget": 0.1}], "why": "w", "focus": null}', "budget too large"), ('{"rules": [{"expr": "g", "budget": 0.1}, {"expr": "g", "budget": 0.1}], "why": "w", "focus": null}', "same rule"), ('{"rules": [{"expr": "g.real", "budget": 0.1}, {"expr": "h", "budget": 0.1}], "why": "w", "focus": null}', "attribute"), ('{"rules": [{"expr": "g", "budget": 0.1}, {"expr": "h", "budget": 0.1}], "why": "w", "focus": "nope"}', "unknown bench")):
        check(f"rule: reply refused, {why}", parse_rules(bad, ["blink"], 2)[0] is None)
    R = go("pipeline", rounds=2)
    arms = steps(R, "arm")
    check("pipeline: two rounds, each arm a written rule, measured and decided", R.ok and len(arms) == 4 and all(r["rule"] in ("own1", "own2") for r in arms) and all("rules" in r for r in steps(R, "choice")), [r["rule"] for r in arms])
    check("pipeline: the ledger shows the written rules to the next round", "you wrote own1 = rel*g/-h" in open(os.path.join(R.w, "r2", "ask", "prompt.txt")).read())
    check("pipeline: the written arms are in the table and composed", all(any(x["label"] == r["arm"] for x in json.load(open(os.path.join(R.w, f"r{r['round']}", "table.json")))) and r["arm"] in json.load(open(os.path.join(R.w, f"r{r['round']}", "composed.json"))) for r in arms))
    first = json.load(open(os.path.join(R.w, "final.json")))
    stops = len(go("pipeline", rounds=2, work=os.path.join(root, "pipe_kill0"), stop=10**9).commands)
    for stop in (7, 25):
        w = os.path.join(root, f"pipe_kill{stop}")
        K = go("pipeline", rounds=2, work=w, stop=stop)
        K = go("pipeline", rounds=2, work=w)
        check(f"pipeline: killed after {stop} ticks, resumed to the same final scales", K.ok and json.load(open(os.path.join(w, "final.json")))["scales"] == json.load(open(os.path.join(root, "pipe_kill0", "final.json")))["scales"])
    # this file
    src = open(os.path.abspath(__file__)).read().split("def self_test")[0]
    check("no deletion in this file", not re.search(r"os\.(remove|unlink|rmdir)|shutil|rmtree|\brm\b -|\.unlink\(|find .*-delete|git clean", src))
    check("no network in this file", not re.search(r"\b(urllib|requests|socket|http\.client|curl|wget|ssh)\b", src))
    check("no held-out path or list in this file", not re.search(r"HELD|mm/sets|heldout", src))
    print(f"{len(fails)} failed" + (": " + "; ".join(fails) if fails else ""), flush=True)
    return not fails


def main():
    a = finish_args(parser().parse_args())
    if not a.dry_run:
        for p in (os.path.join(a.base, "config.json"), a.learn, os.path.join(a.items, "items.jsonl")):
            if not os.path.exists(p):
                raise SystemExit(f"no {p}")
    sys.exit(0 if Rsi(a).run() else 1)


if __name__ == "__main__":                                          # vLLM TP>1 workers re-import this file: nothing at top level
    if sys.argv[1:2] == ["--ask-worker"]:
        ask_worker(sys.argv[2], sys.argv[3:])
    elif sys.argv[1:2] == ["--self-test"]:
        os.makedirs(sys.argv[2], exist_ok=True)
        sys.exit(0 if self_test(sys.argv[2]) else 1)
    else:
        main()
