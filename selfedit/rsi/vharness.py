"""RSI on a model's own weights and structure. The RSI loop is the core; the harness below it only combines what the loop accepted.

CORE (class Lineage):  read the residual on IID half 0 -> a driver proposes one edit with any operator -> write the child checkpoint or engine arm
                       -> screen on half 0 -> test on half 1 -> an accepted child becomes a possible parent; the next proposer sees the parent's edits.
COMBINING LAYER (from Paper B HarnessBoost, part "combine"): each edit is a column of answer probabilities, the residual g = 1 - p is the functional gradient,
                       the proposer is the weak learner, the refit is a dose change of accepted edits. Default combination: join accepted edits into ONE checkpoint
                       with disjoint units and report epsilon_AB. Routing per problem is optional and off.
OPERATORS (OPS):       gate (last-layer gated write), patch, kernel, tokens (caps), direction (place, layer, dose), scale, tune (dose of an accepted edit), verbosity (edits of thinking length).
                       Every operator has d = number of free parameters changed; it is recorded and reported, nothing branches on it.
DRIVERS (DRIVERS):     one interface, propose(ctx, n): model (the parent checkpoint itself), claude, astra, random. Never Fable. --experience FILE adds the experience library, and the arm label says so.
DATA:                  IID half 0 = residual, fit, screen; IID half 1 = test; OOD = frozen by class in the registry, scored once, by `finalize`.
Every step leaves a file, a step whose file exists is not run again, a failed job writes .failed and the candidate is rejected with that reason.

Usage (theory/HARNESS.md has the protocol):
  vharness.py register REGISTRY --iid-files F... --ood-files F...             fix classes and OOD lists, before any result; refuses to overwrite
  vharness.py plan     WORK --registry REGISTRY --start stock|p8eq --driver D [--experience FILE] [--rounds 3] [--slots 4] [--stage2 2]     freeze prereg.json, print the power table
  vharness.py run      WORK                                                   the driver process, CPU only; jobs go to the queue file
  vharness.py job-propose --ctx CTX --out OUT --n N [loop.py args]            GPU job: the checkpoint reads its context and proposes
  vharness.py relay    WORK...                                                on the Mac: answers claude and astra requests
  vharness.py finalize WORK --registry REGISTRY --i-have-frozen-the-lineage   queue and read the one OOD score of the final child and its start
  vharness.py report   WORK...                                                story table per arm, Delta_RSI between arms on their own OOD files
  vharness.py drop-class REGISTRY CLASS --reason TEXT                         a whole class only; never single rows
"""
import argparse, glob, hashlib, json, math, os, random, re, shlex, subprocess, sys, time, zlib, collections
import numpy as np

D = "/data/fd/mingdazhang/selfedit"
W = f"{D}/rsi/mm"
STOCK = f"{D}/models/Qwen3.8-27B"
DIRLIB = f"{D}/probe/library"
TAIL = ("--mm --dirlib {lib} --crn 1 --seed 101 --tp 1 --max-model-len 46000 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --effort medium --budget 8192 --mb 128 --max-num-seqs 64 --gpu-util 0.85")
half = lambda i: zlib.crc32(i.encode()) % 2
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
canon = lambda o: json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
NOW = lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z")

# ---------------------------------------------------------------- data: classes -------------------------------------------------------------
# Membership is by the name of the bench and category, written before any result of the class is read. A row belongs to the first class that matches.
CLASSES = {
    "counting": {"iid": r"^(countbench\||cvbench\|2D/Count|blink\|Counting|babyvision\|(Fine-grained Discrimination/Count|Spatial Perception/Count))", "ood": r"^pixmocount\|",
                 "evidence": "eye exam dots; BabyVision Count Same Patterns"},
    "tracing": {"iid": r"^babyvision\|Visual Tracking/", "ood": None, "evidence": "eye exam trace; tracing barely moves with resolution"},
    "acuity": {"iid": r"^(babyvision\|Fine-grained Discrimination/|mmstar\|fine-grained perception)", "ood": r"^hrbench4k\|", "evidence": "eye exam ring, square; gain by image side"},
    "place": {"iid": r"^(cvbench\|2D/Relation|blink\|(Object_Localization|Spatial_Relation)|erqa\|Pointing|vstar\|relative_position)", "ood": None, "evidence": "eye exam cell 12x12 .40"},
    "camera": {"iid": r"^blink\|Multi-view_Reasoning", "ood": r"^(vrrpi\||spar\|spar camera_motion_infer|mmsi\|mmsi (Motion _Cam\._|Positional Relationship _Cam\._Cam\._))", "evidence": "orbit rule, BLINK Multi-view .383"},
    "multiview": {"iid": r"^(erqa\|Multi-view Reasoning|blink\|(Visual_Correspondence|Semantic_Correspondence|Functional_Correspondence))", "ood": r"^mmsi\|mmsi ", "evidence": "ERQA multi-view -16.7 at p4, .361 to .611 at p8eq"},
}
GENERAL_OOD = r"^naturalbench\|"       # no class of its own: frozen, scored once as the general transfer row
# Which lines already touched which set. "chose" = a decision was taken from its scores; "read" = scores were seen, nothing chosen from them.
TOUCHED = {
    "p8box chose": ["babyvision pool", "erqa pool", "blink pool", "cvbench pool (100 rows)"],
    "p8box read, nothing chosen": ["12 held sets: babyvision blink charxiv countbench cvbench erqa hallusionbench mathvista mmstar mmvp realworldqa vstar (sets/)"],
    "run8 and rule lines chose": ["mathvista realworldqa mmvp mmstar charxiv hallusionbench (sets/, categories scanned and rules written)", "countbench vstar (in pool8)", "blink Multi-view (orbit rule)"],
    "read, nothing chosen": ["vrrpi: partial records of the written orbit rule and of a prompt note were read on 2026-11-06, +1.5 and +1.4", "mmsi, spar, vrrpi: jobs sim_base, sim_base2, sim_orbitv, sim_orbitv2, sim_qw_note, sim_gm_base, sim_mmsi2, sim_stock, sim_stock_mmsi are running on them for the stock against p8box question; those are stock, p8box and rule checkpoints, none of them a child of this harness"],
    "never scored in any log found": ["hrbench4k pixmocount naturalbench (Vesta logs show only mmprep; node queue and log names show none; worker08 cannot be checked, its login is down)"],
}


def key(p):
    return f"{p.get('bench') or p['id'].split('/')[0]}|{p.get('category')}"


def class_of(p, side="iid"):
    k = key(p)
    for c, spec in CLASSES.items():
        if spec[side] and re.search(spec[side], k):
            return c
    return None


def read_jsonl(paths):
    out = {}
    for f in paths:
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if "id" in r:
                    out.setdefault(r["id"], r)
    return out


def cmd_register(a):
    if os.path.exists(a.registry):
        raise SystemExit(f"{a.registry} exists: the lists are fixed. Dropping a whole class: drop-class")
    iid, ood = read_jsonl(a.iid_files), read_jsonl(a.ood_files)
    both = set(iid) & set(ood)
    if both:
        raise SystemExit(f"{len(both)} ids are on both sides, e.g. {sorted(both)[:3]}")
    reg = dict(written=NOW(), note="Fixed before any result of a class is read by this harness. Rows are not removed after seeing scores; a class may be dropped as a whole with a reason.",
               iid_files=a.iid_files, ood_files=a.ood_files, touched=TOUCHED, dropped={}, classes={}, general_ood=sorted(i for i, p in ood.items() if re.search(GENERAL_OOD, key(p))))
    for c, s in CLASSES.items():
        reg["classes"][c] = dict(evidence=s["evidence"], iid_pattern=s["iid"], ood_pattern=s["ood"], iid=sorted(i for i, p in iid.items() if class_of(p) == c),
                                 ood=sorted(i for i, p in ood.items() if class_of(p, "ood") == c), clean_test=bool(s["ood"]))
    reg["ood_sha"] = sha(canon({c: v["ood"] for c, v in reg["classes"].items()}) + canon(reg["general_ood"]))
    os.makedirs(os.path.dirname(os.path.abspath(a.registry)), exist_ok=True)
    json.dump(reg, open(a.registry, "w"), indent=0, ensure_ascii=False)
    for c, v in reg["classes"].items():
        print(f"{c:10s} iid {len(v['iid']):5d} ood {len(v['ood']):5d}{'' if v['clean_test'] else '  NO CLEAN TEST SET'}")
    print(f"general ood {len(reg['general_ood'])}; other iid rows {len(iid) - sum(len(v['iid']) for v in reg['classes'].values())}; written {reg['written']} sha {reg['ood_sha'][:12]}")


def cmd_drop_class(a):
    reg = json.load(open(a.registry))
    if a.cls not in reg["classes"]:
        raise SystemExit("unknown class")
    reg["dropped"][a.cls] = dict(when=NOW(), reason=a.reason)
    json.dump(reg, open(a.registry, "w"), indent=0, ensure_ascii=False)
    print(f"dropped {a.cls}: {a.reason}")


# ---------------------------------------------------------------- traces and statistics ------------------------------------------------------
def load_traces(path):
    d = collections.defaultdict(dict)
    for l in open(path):
        r = json.loads(l)
        if "id" in r:
            d[r["id"]][r["sample"]] = r
    return dict(d)


def right(r, budget=8192):
    return bool(r["ok"]) and r["finish"] != "length" and r["tokens"] <= budget


def rate(rs):
    return sum(right(r) for r in rs.values()) / len(rs)


def behaviour(T, ids):
    """Tokens, truncation, finished accuracy: an edit of thinking length can gain through truncation alone, so these are always reported."""
    rs = [r for i in ids for r in T[i].values()]
    fin = [r for r in rs if r["finish"] != "length" and r["tokens"] <= 8192]
    return dict(acc=sum(right(r) for r in rs) / max(1, len(rs)), tokens=sum(r["tokens"] for r in rs) / max(1, len(rs)), trunc=1 - len(fin) / max(1, len(rs)), fin_acc=sum(bool(r["ok"]) for r in fin) / max(1, len(fin)))


def paired(parent, child, ids, pool, alpha=.05, B=4000, macro=True, seed=0):
    """Child minus parent, one-sided lower bound at alpha and upper at alpha. Requires the same samples per problem on both sides; a partial file is an error, not a smaller denominator.
    Macro: every dataset weighs the same. A two-option category with both right answers at 5+ problems is balanced by right answer, so a lean to one option does not count."""
    ids = [i for i in ids if i in parent or i in child]
    for i in ids:
        if i not in parent or i not in child or set(parent[i]) != set(child[i]):
            raise ValueError(f"unpaired samples at {i}")
    if not ids:
        return None
    d = np.array([rate(child[i]) - rate(parent[i]) for i in ids])
    bench = np.array([i.split("/")[0] for i in ids])
    w = np.ones(len(ids))
    bycat = collections.defaultdict(list)
    for k, i in enumerate(ids):
        if len(pool[i].get("options") or []) == 2:
            bycat[key(pool[i])].append(k)
    for ks in bycat.values():
        lab = collections.defaultdict(list)
        for k in ks:
            lab[re.sub(r"\W", "", str(pool[ids[k]]["answer"])).lower()].append(k)
        if len(lab) == 2 and min(map(len, lab.values())) >= 5:
            for v in lab.values():
                w[v] = len(ks) / 2 / len(v)
    names = sorted(set(bench)) if macro else ["all"]
    groups = [np.where(bench == n)[0] for n in names] if macro else [np.arange(len(ids))]
    stat = lambda g: float(np.mean([np.sum(w[x] * d[x]) / np.sum(w[x]) for x in g]))
    rng = np.random.default_rng(seed)
    bs = np.empty(B)
    for b in range(B):
        bs[b] = stat([rng.choice(x, len(x)) for x in groups])
    return dict(gain=stat(groups), lo=float(np.quantile(bs, alpha)), hi=float(np.quantile(bs, 1 - alpha)), n=len(ids), datasets=len(names), sd=float(d.std()))


def mde(sd, n, alpha, power=.8):
    from statistics import NormalDist
    z = NormalDist().inv_cdf
    return (z(1 - alpha) + z(power)) * sd / math.sqrt(max(1, n))


# ---------------------------------------------------------------- combine: the layer from Paper B --------------------------------------------
def column(parent, child, ids):
    """The Bernoulli column of an edit: logit change per problem. Paper B's columns are per option from answer logprobs; here p comes from samples, (k+1)/(n+2)."""
    out = {}
    for i in ids:
        a, c = (sum(right(r) for r in parent[i].values()) + 1) / (len(parent[i]) + 2), (sum(right(r) for r in child[i].values()) + 1) / (len(child[i]) + 2)
        out[i] = (a, math.log(c / (1 - c)) - math.log(a / (1 - a)))
    return out


def score_test(col):
    """q = <c, g>, s = c'Hc, the Tipping & Faul statistics of adding the column to the parent, at weight 0. Logged and used to rank a tie; not the acceptance (theory/HARNESS.md: on 24 written rules it did not separate)."""
    q = sum((1 - a) * c for a, c in col.values())
    s = sum(a * (1 - a) * c * c for a, c in col.values())
    return dict(q=q, s=s, z2=q * q / s if s else 0., passes=bool(q > 0 and q * q > s))


def reach(parent, ids, n_all):
    """What an edit limited to the rows `ids` can buy: the accuracy ceiling (always true) and energy, 1/2 ||g_G||^2 (a reference, broken by the orbit write at 1.34 times, see theory/HARNESS.md)."""
    g = [1 - rate(parent[i]) for i in ids]
    return dict(ceiling=sum(g) / max(1, n_all), energy=sum(x * x for x in g) / max(1, n_all))


def units(engine):
    out = collections.defaultdict(set)
    for e in engine:
        if e.get("op") == "set":
            out[e["param"].rsplit(".", 1)[0]] |= set(e["sel"]["neurons"])
    return out


def join(A, B, parent):
    """One checkpoint from two children of one parent: the parent's edits, then A's own, then B's own. Refuses shared units and two different vision settings."""
    pe = parent["engine"]
    own = lambda g: g["engine"][len(pe):]
    if A["engine"][:len(pe)] != pe or B["engine"][:len(pe)] != pe:
        raise ValueError("not children of this parent")
    if A["vision"] != parent["vision"] and B["vision"] != parent["vision"] and A["vision"] != B["vision"]:
        raise ValueError("two different vision settings")
    ua, ub = units(own(A)), units(own(B))
    for k in ua:
        if ua[k] & ub.get(k, set()):
            raise ValueError(f"units shared at {k}: {sorted(ua[k] & ub[k])[:5]}")
    return dict(vision=A["vision"] if A["vision"] != parent["vision"] else B["vision"], engine=pe + own(A) + own(B))


def interaction(tP, tA, tB, tAB, ids, pool):
    """epsilon_AB = Y(AB) - Y(A) - Y(B) + Y(parent), by problem; two children of one parent on the same samples."""
    d = {i: rate(tAB[i]) - rate(tA[i]) - rate(tB[i]) + rate(tP[i]) for i in ids}
    v = np.array(list(d.values()))
    rng = np.random.default_rng(0)
    bs = [float(rng.choice(v, len(v)).mean()) for _ in range(2000)]
    return dict(epsilon=float(v.mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), n=len(v))


def route_report(tA, tB, pool):
    """Optional routing among accepted edits, against the best single checkpoint: fitted by category on half 0, scored on half 1. Off by default: stock against p8box gave -1.8 [-3.3,-.1]."""
    ids = sorted(set(tA) & set(tB))
    fit = collections.defaultdict(lambda: [0., 0.])
    for i in ids:
        if half(i) == 0:
            fit[key(pool[i])][0] += rate(tA[i]); fit[key(pool[i])][1] += rate(tB[i])
    te = [i for i in ids if half(i) == 1]
    best = "A" if sum(rate(tA[i]) for i in ids if half(i) == 0) >= sum(rate(tB[i]) for i in ids if half(i) == 0) else "B"
    g = []
    for i in te:
        k = fit.get(key(pool[i]))
        pick = best if not k or k[0] == k[1] else ("A" if k[0] > k[1] else "B")
        g.append(rate(tA[i] if pick == "A" else tB[i]) - rate(tA[i] if best == "A" else tB[i]))
    return dict(gain=float(np.mean(g)), n=len(g))


# ---------------------------------------------------------------- operators ------------------------------------------------------------------
WRITERS = {"attn": lambda L: f"layers.{L}.self_attn.o_proj" if L % 4 == 3 else f"layers.{L}.linear_attn.out_proj", "mlp": lambda L: f"layers.{L}.mlp.down_proj"}
DIRECTIONS = ["copyloop_vs_normal", "correct_vs_wrong", "late_vs_early_correct", "long_vs_short_correct", "trunc_vs_correct", "verbose_perp_position"]
VERBOSE = ["copyloop_vs_normal", "late_vs_early_correct", "long_vs_short_correct", "trunc_vs_correct", "verbose_perp_position"]
HIDDEN, TLAYERS, VBLOCKS, VDIM = 5120, 64, 27, 1152
GATE_MAX_WORDS = 60


class Bad(Exception):
    pass


def ch(x, name, lo=None, hi=None, among=None):
    if among is not None and x not in among:
        raise Bad(f"{name} must be one of {among}")
    if lo is not None and not (isinstance(x, (int, float)) and lo <= x <= hi):
        raise Bad(f"{name} must lie in {lo}..{hi}")
    return x


def direction_edit(name, kind, L, w, alpha, lib=DIRLIB):
    z = np.load(os.path.join(lib, name + ".npz"))
    v = np.asarray(z[f"{kind}/dir"][L], dtype=np.float64)
    return {"op": "amplify", "param": WRITERS[w](L), "side": "out", "alpha": alpha, "u": (v / np.linalg.norm(v)).round(7).tolist()}


class OP:
    """validate(p, parent) gives (child genome change, d). d = free parameters this edit changes, reported in the ledger and story, used for nothing else."""


def op_patch(p, g):
    v = dict(g["vision"] or {"patch": 16, "kernel": "box", "max": None, "min": None, "merge": 0})
    ps = ch(p.get("patch"), "patch", among=[4, 8, 5, 11])
    k = ch(p.get("kernel", "box"), "kernel", among=["box", "bilinear", "bicubic"])
    if k == "box" and 16 % ps:
        raise Bad("box needs a patch size dividing 16")
    v.update(patch=ps, kernel=k, max=v["max"] or 4096)
    return dict(vision=v), 2 if k != "box" else 1


def op_kernel(p, g):
    if not g["vision"]:
        raise Bad("kernel needs a patch change first")
    k = ch(p.get("kernel"), "kernel", among=["box", "bilinear", "bicubic"])
    return dict(vision=dict(g["vision"], kernel=k)), 3 * 2 * g["vision"]["patch"] ** 2 * VDIM   # the kernel tensor is rewritten; its numbers follow from the rule, d counts what changes


def op_tokens(p, g):
    v = dict(g["vision"] or {"patch": 16, "kernel": "box", "max": None, "min": None, "merge": 0})
    v["max"] = ch(p.get("max"), "max", 256, 16384)
    v["min"] = ch(p.get("min", 0), "min", 0, v["max"]) or None
    return dict(vision=v), 2


def op_scale(p, g):
    L = ch(p.get("layer"), "layer", 0, TLAYERS - 1)
    w = ch(p.get("writer"), "writer", among=list(WRITERS))
    a = ch(p.get("alpha"), "alpha", .5, 1.5)
    return dict(add=[{"op": "scale", "param": WRITERS[w](L), "alpha": a}]), 1


def op_skip(p, g):
    """Removes one residual writer of a text layer."""
    L = ch(p.get("layer"), "layer", 0, TLAYERS - 1)
    w = ch(p.get("writer"), "writer", among=list(WRITERS))
    return dict(add=[{"op": "scale", "param": WRITERS[w](L), "alpha": 0.0}]), 1


def op_attn_temp(p, g):
    """Query scale of the self-attention layers in a range: above 1 sharpens attention, below 1 flattens it. Only layers 3, 7, 11, ... have self-attention."""
    L0, L1 = ch(p.get("first"), "first", 0, TLAYERS - 1), ch(p.get("last"), "last", 0, TLAYERS - 1)
    if L1 < L0:
        raise Bad("layers: first..last")
    a = ch(p.get("alpha"), "alpha", .5, 2)
    L = [i for i in range(L0, L1 + 1) if i % 4 == 3]
    if not L:
        raise Bad("no self-attention layer in the range: use layers 3, 7, 11, ...")
    return dict(add=[{"op": "scale", "param": f"layers.{i}.self_attn.q_proj", "alpha": a} for i in L]), len(L)


def op_vision_scale(p, g):
    """Scales the vision features the text layers receive."""
    return dict(add=[{"op": "scale", "param": "model.visual.merger.linear_fc2", "alpha": ch(p.get("alpha"), "alpha", .7, 1.4)}]), 1


def op_direction(p, g, among=DIRECTIONS):
    L0, L1 = ch(p.get("first"), "first", 0, TLAYERS - 1), ch(p.get("last"), "last", 0, TLAYERS - 1)
    if L1 < L0 or L1 - L0 > 8:
        raise Bad("layers: first..last, at most 9")
    w = ch(p.get("writer", "attn"), "writer", among=list(WRITERS))
    n = ch(p.get("name"), "name", among=among)
    a = ch(p.get("alpha"), "alpha", -8, 8)
    return dict(add_dir=dict(name=n, kind=ch(p.get("kind", "resid"), "kind", among=["resid", "mixer", "mlp"]), first=L0, last=L1, writer=w, alpha=a)), HIDDEN * (L1 - L0 + 1)


def op_verbosity(p, g):
    return op_direction(p, g, VERBOSE)


def op_gate(p, g):
    n = p.get("note", "")
    if not isinstance(n, str) or not n.strip() or len(n.split()) > GATE_MAX_WORDS:
        raise Bad(f"note: text of 1 to {GATE_MAX_WORDS} words")
    if p.get("scope", "").startswith("cat:") is False:
        raise Bad("scope must be cat:<category> for a gated write")
    return dict(gate=dict(note=n.strip().replace("\t", " ").replace("\n", " "))), (3 * HIDDEN + 17408) * len(n.split())   # roughly one unit per token, three vectors each


def op_tune(p, g):
    k = ch(p.get("factor"), "factor", .25, 2.)
    i = p.get("edit")
    if not isinstance(i, int) or not 0 <= i < len(g["engine"]):
        raise Bad(f"edit: index of an accepted edit, 0..{len(g['engine']) - 1}")
    e = dict(g["engine"][i])
    if e["op"] == "scale":
        e["alpha"] = round(1 + (e["alpha"] - 1) * k, 4)
    elif e["op"] == "amplify":
        e["alpha"] = round(e["alpha"] * k, 4)
    elif e["op"] == "set" and e["param"].endswith("down_proj"):
        e["value"] = (np.array(e["value"]) * k).round(5).tolist()
    else:
        raise Bad("this edit has no dose")
    en = list(g["engine"])
    en[i] = e
    return dict(replace=en), 1


OPS = {"gate": op_gate, "patch": op_patch, "kernel": op_kernel, "tokens": op_tokens, "scale": op_scale, "direction": op_direction, "tune": op_tune, "verbosity": op_verbosity, "skip": op_skip, "attn_temp": op_attn_temp, "vision_scale": op_vision_scale}


def validate(p, genome, categories):
    if not isinstance(p, dict) or p.get("op") not in OPS:
        raise Bad(f"op must be one of {list(OPS)}")
    if isinstance(p.get(p["op"]), dict):         # drivers often nest the fields under the op name, as the menu reads
        p = {**p, **p[p["op"]]}
    change, d = OPS[p["op"]](p, genome)
    s = p.get("scope", "all")
    if s != "all":
        if not s.startswith("cat:") or s[4:] not in categories:
            raise Bad("scope: all, or cat:<a category from the table>")
    return change, d, s


def gid(genome):
    slim = dict(vision=genome["vision"], engine=[{k: (sha(canon(v)) if k in ("value", "u") else v) for k, v in e.items()} for e in genome["engine"]], dirs=genome.get("dirs", []))
    return sha(canon(slim))[:14]


# ---------------------------------------------------------------- drivers: one interface -----------------------------------------------------
MENU = """Edits you can make to the network, without training. Answer with a JSON list; each item has "op", "scope" ("all", or "cat:<category>" with a category from the table), "why", and:
- gate: {"note": text up to 60 words, "scope": "cat:..."}. Writes the note into the last layer with one repurposed neuron per token, so that the checkpoint recites it when it starts on such a problem. No prompt is added.
- patch: {"patch": 4|5|8|11, "kernel": "box"|"bilinear"|"bicubic"}. Vision patch size in pixels, now 16 unless the record says otherwise. The new kernel is derived from the old, no training.
- kernel: {"kernel": ...}. Changes the kernel rule of a patch edit already made.
- tokens: {"max": 256..16384, "min": 0..max}. Image tokens per image; images outside are resized by the processor.
- scale: {"layer": 0..63, "writer": "attn"|"mlp", "alpha": .5..1.5}. Scales one residual writer of a text layer.
- direction: {"name": one of NAMES, "first": L, "last": L, "writer": "attn"|"mlp", "alpha": -8..8}. Adds or removes a direction in the writers' output over at most 9 layers.
- verbosity: same fields as direction, for the names that change thinking length (VERB). Graded on accuracy, tokens, truncation and finished accuracy.
- skip: {"layer": 0..63, "writer": "attn"|"mlp"}. Removes one residual writer of a text layer.
- attn_temp: {"first": L, "last": L, "alpha": .5..2}. Query scale of the self-attention layers in the range (3, 7, 11, ...): above 1 sharpens attention, below 1 flattens it.
- vision_scale: {"alpha": .7..1.4}. Scales the vision features the text layers receive.
- tune: {"edit": index in the record below, "factor": .25..2}. Changes the dose of an accepted edit.
Propose changes that fix the failures below. Make each item different from the others and from what the record says was tried. End with the JSON list."""


def context_text(ctx, exp=None):
    s = "You are working on a vision-language model, editing its own weights and structure.\n\n"
    if exp:
        s += "Experience from earlier work on this:\n" + exp + "\n\n"
    s += "Record of this lineage (decisions only; test numbers are not shown):\n" + ctx["record"] + "\n\nThe model now, on the problems it may look at:\n" + ctx["table"] + "\n\nSome failures, in its own words:\n" + ctx["demos"] + "\n\n"
    return s + MENU.replace("NAMES", ", ".join(DIRECTIONS)).replace("VERB", ", ".join(VERBOSE))


def parse_list(text):
    """The last JSON list in a text, or nothing. Single objects are accepted."""
    for m in reversed(list(re.finditer(r"\[\s*\{.*?\}\s*\]", text or "", re.S))):
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    out = []
    for m in re.finditer(r"\{[^{}]*\"op\"[^{}]*\}", text or ""):
        try:
            out.append(json.loads(m.group(0)))
        except Exception:
            pass
    return out


class Driver:
    """propose(ctx, n, lin) returns proposal dicts, or None while waiting. Every driver gets the same context, the same n and the same validation."""
    name = "?"

    def __init__(self, a):
        self.a = a


class RandomDriver(Driver):
    """Control: uniform over the same operators and ranges. Notes come from a pool of notes written for other categories and shuffled, so a note is real text not aimed at the category."""
    name = "random"

    def propose(self, ctx, n, lin):
        rng = random.Random(zlib.crc32(f"{lin.round}{lin.a['seed']}".encode()))
        pool = [json.loads(l)["rule"] for l in open(lin.a["notes_pool"])] if lin.a.get("notes_pool") else []
        cats = ctx["cards"] or ["?"]
        out = []
        for _ in range(n):
            k = rng.choice([o for o in OPS if o != "gate" or pool])
            c = "cat:" + rng.choice(cats)
            L = rng.randrange(TLAYERS)
            p = {"gate": lambda: dict(note=rng.choice(pool), scope=c),
                 "patch": lambda: dict(patch=rng.choice([4, 8, 5, 11]), kernel="bilinear"),
                 "kernel": lambda: dict(kernel=rng.choice(["box", "bilinear", "bicubic"])),
                 "tokens": lambda: dict(max=rng.choice([1024, 2048, 4096, 8192]), min=rng.choice([0, 1024])),
                 "scale": lambda: dict(layer=L, writer=rng.choice(list(WRITERS)), alpha=rng.choice([.7, .85, 1.15, 1.3])),
                 "direction": lambda: dict(name=rng.choice(DIRECTIONS), first=L, last=min(63, L + rng.randrange(5)), writer="attn", alpha=rng.choice([-4, -2, -1, 1, 2, 4])),
                 "verbosity": lambda: dict(name=rng.choice(VERBOSE), first=L, last=min(63, L + rng.randrange(5)), writer="attn", alpha=rng.choice([-4, -2, -1, 1, 2, 4])),
                 "tune": lambda: dict(edit=rng.randrange(max(1, len(lin.parent["engine"]))), factor=rng.choice([.5, .7, 1.4, 2.]))}[k]()
            out.append(dict(p, op=k, why="random control"))
        return out


class ModelDriver(Driver):
    """The parent checkpoint reads its own context: a GPU job on the queue. Text only, thinking on."""
    name = "model"

    def propose(self, ctx, n, lin):
        out = f"{lin.dir}/proposals.model.json"
        if os.path.exists(out):
            return json.load(open(out))
        if os.path.exists(out + ".failed"):
            return []
        f = f"{lin.dir}/ctx.json"
        json.dump(dict(prompt=context_text(ctx, lin.experience())), open(f, "w"))
        arm = lin.arm_file(lin.parent, "propose")
        lin.submit(f"prop_{lin.short}", f"python -u rsi/vharness.py job-propose --ctx {f} --out {out} --n {n * 4} --arm {arm} --run-dir {lin.dir} --model {lin.model_path(lin.parent)} --pool {lin.a['iidfile']} " + TAIL.format(lib=DIRLIB), [out], out + ".failed")
        return None


class RelayDriver(Driver):
    """Claude Opus 5.5 or Codex gpt-6-astra, outside the node: leaves the prompt in the lineage directory, the Mac relay writes the answer back."""
    def propose(self, ctx, n, lin):
        out = f"{lin.dir}/proposals.{self.name}.json"
        if os.path.exists(out):
            return parse_list(open(out).read())
        if os.path.exists(out + ".failed"):
            return []
        q = f"{lin.dir}/relay.{self.name}.prompt"
        if not os.path.exists(q):
            open(q, "w").write(context_text(ctx, lin.experience()) + f"\n\nGive {n} items.\n")
        return None


class ClaudeDriver(RelayDriver):
    name = "claude"


class AstraDriver(RelayDriver):
    name = "astra"


DRIVERS = {d.name: d for d in (ModelDriver, ClaudeDriver, AstraDriver, RandomDriver)}
FORBIDDEN_MODELS = ("fable",)


def cmd_job_propose(argv):
    """GPU job. argv keeps the loop.py arguments."""
    def pop(flag):
        i = argv.index(flag); v = argv[i + 1]; del argv[i:i + 2]; return v
    ctx, out, n, arm = pop("--ctx"), pop("--out"), int(pop("--n")), pop("--arm")
    try:
        sys.argv = ["loop.py"] + argv
        CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
        import rsi.loop as L
        from surgery import Surgeon
        a = L.args
        S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens, gpu_memory_utilization=a.gpu_util, kv_cache_dtype=a.kv_cache_dtype, multimodal=True)
        en = json.load(open(arm))[0]["engine"]
        h = S.apply(en) if en else None
        prompt = json.load(open(ctx))["prompt"] + "\n\nGive 1 item.\n"
        outs = S.generate([[{"role": "user", "content": prompt}]] * n, n=1, sampling=dict(L.SAMPLING), max_tokens=6000, reasoning_effort="medium")
        if h is not None:
            S.revert(h)
        got, raw = [], []
        for o in outs:
            s = o["samples"][0]
            raw.append(dict(content=(s["content"] or "")[-1500:], reasoning=(s["reasoning"] or "")[-800:], finish=s["finish_reason"]))
            got += parse_list(s["content"] or "")
        json.dump(raw, open(out + ".raw.json", "w"), ensure_ascii=False)
        json.dump(got, open(out + ".tmp", "w"), ensure_ascii=False)
        os.replace(out + ".tmp", out)
    except Exception:
        open(out + ".failed", "w").write(repr(sys.exc_info()[1]))
        raise


def cmd_relay(a):
    """Mac. Only the prompt text crosses, only proposals come back, through `gpu run`, `gpu put`. Claude: needs CLAUDE_BIN; without it the prompt stays in place for an Opus 5.5 subagent."""
    claude = os.environ.get("CLAUDE_BIN")
    while True:
        for wd in a.work:
            lst = subprocess.run(["gpu", "run", f"ls {wd}/r*/relay.*.prompt 2>/dev/null"], capture_output=True, text=True).stdout.split()
            for q in lst:
                dr, name = os.path.dirname(q), os.path.basename(q).split(".")[1]
                if subprocess.run(["gpu", "run", f"test -e {dr}/proposals.{name}.json && echo y"], capture_output=True, text=True).stdout.strip() == "y":
                    continue
                prompt = subprocess.run(["gpu", "run", f"cat {q}"], capture_output=True, text=True).stdout
                ans = None
                if name == "astra":
                    tf = f"/tmp/vh.{os.getpid()}.out"
                    subprocess.run(["codex-retry", "-m", "gpt-6-astra", "-c", "model_reasoning_effort=xhigh", "-s", "read-only", "--skip-git-repo-check", "-o", tf, prompt], capture_output=True, text=True)
                    ans = open(tf).read() if os.path.exists(tf) else None
                elif name == "claude" and claude:
                    ans = subprocess.run([claude, "-p", "--model", "claude-opus-5-5"], input=prompt, capture_output=True, text=True).stdout
                if ans:
                    tf = f"/tmp/vh.{os.getpid()}.ans"
                    open(tf, "w").write(ans)
                    subprocess.run(["gpu", "put", tf, f"{dr}/proposals.{name}.json"], capture_output=True)
                    print(NOW(), "relayed", q, flush=True)
        if a.once:
            return
        time.sleep(120)


# ---------------------------------------------------------------- backend ----------------------------------------------------------------------
class Queue:
    """Lines go to QFILE, which vesta_bridge copies to the Mac queue. A job's fate: its output file, or .rc != 0 in the gpu log directory."""
    def __init__(self, qfile=f"{W}/evoq.txt", logs="/data/fd/mingdazhang/gpu-logs"):
        self.qfile, self.logs = qfile, logs

    def submit(self, name, cmd):
        q = open(self.qfile).read() if os.path.exists(self.qfile) else ""
        if not any(l.startswith(name + "\t") for l in q.split("\n")):
            open(self.qfile, "a").write(f"{name}\t{cmd}\n")

    def failed(self, name):
        f = f"{self.logs}/selfedit-{name}.rc"
        return os.path.exists(f) and open(f).read().strip() not in ("0", "")


# ---------------------------------------------------------------- the core loop ----------------------------------------------------------------
class Lineage:
    def __init__(self, work, backend=None, sleep=60, patience=8 * 3600):
        self.work = work
        self.a = json.load(open(f"{work}/prereg.json"))
        self.q = backend or Queue()
        self.sleep, self.patience = sleep, patience
        self.pool = read_jsonl([self.a["iidfile"]])
        self.state = json.load(open(f"{work}/state.json")) if os.path.exists(f"{work}/state.json") else dict(round=0, accepted=[], tried=[], root=None)
        self.round = self.state["round"]
        self.short = ""
        self.drv = DRIVERS[self.a["driver"]](self.a)
        self.cache = self.a["cache"]
        self.parent = None
        self.dir = None

    # ledger and queue
    def log(self, **kw):
        kw.update(t=NOW(), arm=self.a["label"], round=self.round)
        open(f"{self.work}/ledger.jsonl", "a").write(json.dumps(kw, ensure_ascii=False) + "\n")

    def say(self, s):
        line = time.strftime("%H:%M:%S ") + s
        print(line, flush=True)
        open(f"{self.work}/lineage.txt", "a").write(line + "\n")

    def experience(self):
        return open(self.a["experience"]).read() if self.a.get("experience") else None

    def save(self):
        self.state["round"] = self.round
        json.dump(self.state, open(f"{self.work}/state.json.new", "w"))
        os.replace(f"{self.work}/state.json.new", f"{self.work}/state.json")

    def submit(self, name, cmd, outs, failflag=None):
        name = f"{name}_{sha(self.cache)[:4]}"                   # queue names stay done for ever: a different cache must not match them
        self.q.submit(name, cmd)
        self.pending = getattr(self, "pending", [])
        self.pending.append(dict(name=name, outs=outs, flag=failflag))

    def wait(self):
        """Until every job has its file or has failed. A job past the patience counts as failed, and says so."""
        t0 = time.time()
        while True:
            left = []
            for j in getattr(self, "pending", []):
                if all(os.path.exists(o) for o in j["outs"]):
                    continue
                if j["flag"] and os.path.exists(j["flag"]):
                    continue
                if self.q.failed(j["name"]):
                    self.say(f"job failed: {j['name']}")
                    if j["flag"]:
                        open(j["flag"], "w").write("job failed\n")
                    continue
                if time.time() - t0 > self.patience:
                    self.say(f"job past patience: {j['name']}")
                    if j["flag"]:
                        open(j["flag"], "w").write("past patience\n")
                    continue
                left.append(j)
            if not left:
                self.pending = []
                return
            if hasattr(self.q, "step"):
                self.q.step()
                continue
            time.sleep(self.sleep)

    # checkpoints and arms
    def model_path(self, g):
        if not g["vision"]:
            return self.a["stock"]
        return f"{self.a['models']}/{sha(canon(g['vision']))[:12]}"

    def arm_file(self, g, tag):
        f = f"{self.cache}/arms/{gid(g)}.json"
        if not os.path.exists(f):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            json.dump([{"label": "a", "effort": "medium", "engine": g["engine"]}], open(f + ".new", "w"))
            os.replace(f + ".new", f)
        return f

    def build(self, g):
        """A vision genome needs its checkpoint, built from stock with the kernel rule; the video settings follow in rsi/vision_patch.py."""
        v = g["vision"]
        if not v:
            return None
        m = self.model_path(g)
        if os.path.exists(f"{m}/config.json"):
            return None
        cmd = f"python -u rsi/vision_patch.py --model {self.a['stock']} --out {m} --patch {v['patch']} --kernel {v['kernel']} --max-tokens {v['max'] or 4096}"
        cmd += f" --min-tokens {v['min']}" if v.get("min") else ""
        cmd += f" --merge {v['merge']}" if v.get("merge") else ""
        self.submit(f"vh_ck_{sha(canon(v))[:10]}", cmd, [f"{m}/config.json"], f"{m}.failed")
        return m

    def rows_file(self, name, ids):
        f = f"{self.cache}/rows/{name}.jsonl"
        if not os.path.exists(f):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f + ".new", "w") as o:
                for i in sorted(ids):
                    assert i not in self.ood_ids, f"OOD row {i} reached an IID row file"
                    o.write(json.dumps(self.pool[i], ensure_ascii=False) + "\n")
            os.replace(f + ".new", f)
        return f

    def evaluate(self, g, rows, samples):
        """Same seed, same settings, for every checkpoint: the file names the checkpoint and the rows, so lineages share parents and repeats."""
        i = gid(g)
        d = f"{self.cache}/ev/{i}"
        os.makedirs(d, exist_ok=True)
        out = f"{d}/a.{rows}x{samples}.jsonl"
        if os.path.exists(out):
            return out
        if os.path.exists(out + ".failed"):
            return None
        src = f"{self.cache}/rows/{rows}.jsonl"
        self.submit(f"vh_{i[:10]}_{rows[:18]}x{samples}", f"python -u rsi/vharness.py job-eval {d} {rows}x{samples} {self.arm_file(g, rows)} --samples {samples} --sets {rows}x{samples}:{src} --curve 8192 --model {self.model_path(g)} --pool {self.a['iidfile']} " + TAIL.format(lib=DIRLIB), [out], out + ".failed")
        return out

    def resolve(self, g):
        """Direction edits are named in the genome and resolved to vectors here, once."""
        for x in g.pop("dirs_new", []):
            for L in range(x["first"], x["last"] + 1):
                g["engine"].append(direction_edit(x["name"], x["kind"], L, x["writer"], x["alpha"], self.a.get("dirlib", DIRLIB)))
            g.setdefault("dirs", []).append(x)
        return g

    # the steps of one round
    def read_residual(self):
        """Half 0 only. Parent's table per category, ranked by residual energy, some failures. No half 1 number can reach a driver."""
        T = load_traces(self.evaluate_now(self.parent, "h0", 2))
        cats = collections.defaultdict(list)
        for i in T:
            assert half(i) == 0, i
            cats[key(self.pool[i])].append(i)
        tot = sum((1 - rate(T[i])) ** 2 for i in T) or 1.
        rows = []
        for c, ids in cats.items():
            g = sum((1 - rate(T[i])) ** 2 for i in ids)
            rows.append(dict(cat=c, n=len(ids), acc=sum(rate(T[i]) for i in ids) / len(ids), energy=g / tot))
        rows.sort(key=lambda r: -r["energy"])
        b = behaviour(T, list(T))
        table = f"half 0 overall acc {b['acc']:.3f} tokens {b['tokens']:.0f} truncated {b['trunc']:.3f} finished acc {b['fin_acc']:.3f}\ncategory | problems | acc | share of the residual\n" + "\n".join(f"{r['cat']} | {r['n']} | {r['acc']:.2f} | {r['energy']:.3f}" for r in rows[:30])
        demos, rng = [], random.Random(self.round)
        for r in rows[:6]:
            bad = [i for i in cats[r["cat"]] if rate(T[i]) < 1]
            for i in rng.sample(bad, min(2, len(bad))):
                t = next(x for x in T[i].values() if not right(x))
                demos.append(f"[{r['cat']}] {self.pool[i]['prompt'][:300]}\nright answer {t['answer']}; said: {(t.get('content') or '')[-120:]}\nend of its thinking: {(t.get('reasoning') or '')[-500:]}\n")
        rec = []
        for s in self.state["tried"]:
            rec.append(f"round {s['round']}: {s['op']} {canon(s['proposal'])[:200]} -> {'ACCEPTED' if s.get('accepted') else 'rejected'}" + (f" (half 0 gain {100 * s['fit']:+.1f})" if s.get("fit") is not None else ""))
        rec.insert(0, f"Now: vision {canon(self.parent['vision'])}; {len(self.parent['engine'])} engine edits; earlier edits, numbered for tune: " + "; ".join(f"{k}: {e['op']} {e.get('param', '')}" for k, e in enumerate(self.parent["engine"])))
        return dict(table=table, demos="\n".join(demos), record="\n".join(rec), cards=[r["cat"].split("|", 1)[1] for r in rows[:12] if r["cat"].split("|", 1)[1] != "None"]), T

    def evaluate_now(self, g, rows, samples):
        m = self.build(g)
        if m:
            self.wait()
        out = self.evaluate(g, rows, samples)
        self.wait()
        if not out or not os.path.exists(out):
            raise RuntimeError(f"evaluation of {gid(g)} on {rows} failed")
        return out

    def scope_rows(self, s):
        """All IID rows of a category, by half, and rows of the same class outside it (transfer)."""
        cat = s[4:]
        G = [i for i, p in self.pool.items() if key(p) == cat]
        cls = {class_of(self.pool[i]) for i in G} - {None}
        same = [i for i, p in self.pool.items() if class_of(p) in cls and i not in set(G)]
        return G, same

    def child_of(self, p, change):
        c = dict(vision=change.get("vision", self.parent["vision"]), engine=list(change.get("replace", self.parent["engine"])), dirs=list(self.parent.get("dirs", [])))
        c["engine"] += change.get("add", [])
        if "add_dir" in change:
            c["dirs_new"] = [change["add_dir"]]
        return self.resolve(c)

    def write_gate(self, p, change, s, tag, h0):
        """The last-layer write as the first-class operator: the parent's traces and half 0 others; half 1 traces reach the write only for its printed check, never for unit choice (--strict-half0)."""
        d = f"{self.cache}/gate/{tag}"
        os.makedirs(d, exist_ok=True)
        f = f"{d}/gt_{tag}.json"
        if os.path.exists(f) or os.path.exists(f + ".failed"):
            return f
        open(f"{d}/note.tsv", "w").write(f"{tag}\t{change['gate']['note']}\n")
        G, _ = self.scope_rows(s)
        tr = self.evaluate_now(self.parent, self.rows_name(s), 2)
        en = f" --engine {self.arm_file(self.parent, 'gate')}" if self.parent["engine"] else ""
        self.submit(f"vh_w_{tag}", f"python -u rsi/mm_gate.py --model {self.model_path(self.parent)} --traces {tr} --others {h0} --pool {self.a['iidfile']} --notes {d}/note.tsv --out {d}/gt --arms {d} --cat {s[4:].split("|", 1)[1].replace(' ', '~')} --strict-half0{en}", [f], f + ".failed")
        return f

    def rows_name(self, s):
        G, _ = self.scope_rows(s)
        name = "c" + sha(s)[:10]
        self.rows_file(name, G)
        return name

    def gate_child(self, f, tag):
        """Read the written arm back: every number finite, the parent's edits first and unchanged, new units disjoint from them; the write's own check of its reach and of false starts decides whether it may be tested."""
        arm = json.load(open(f))[0]["engine"]
        pe = self.parent["engine"]
        if canon(arm[:len(pe)]) != canon(pe):
            raise Bad("the write changed the parent's edits")
        for e in arm[len(pe):]:
            if not np.isfinite(np.array(e["value"], dtype=float)).all():
                raise Bad("not finite")
        new, old = units(arm[len(pe):]), units(pe)
        for k in new:
            if new[k] & old.get(k, set()):
                raise Bad("units reused")
        txt = glob.glob(f"{os.path.dirname(f)}/gt.*.txt")
        if txt:
            t = open(txt[0]).read()
            m = re.search(r"unseen: all-text p median ([\d.]+)", t)
            n = re.search(r"p\(start the note\) mean ([\d.]+)", t)
            if m and float(m.group(1)) < .9:
                raise Bad(f"write did not reach unseen problems, median {m.group(1)}")
            if n and float(n.group(1)) > .01:
                raise Bad(f"note starts on other tasks, {n.group(1)}")
        return dict(vision=self.parent["vision"], engine=arm, dirs=list(self.parent.get("dirs", [])))

    def propose_round(self, ctx):
        done = f"{self.dir}/candidates.json"
        if os.path.exists(done):
            return json.load(open(done))
        while True:
            raw = self.drv.propose(ctx, self.a["slots"], self)
            if raw is not None:
                break
            self.wait()
            if not hasattr(self.q, "step"):
                time.sleep(self.sleep)
            else:
                self.q.step()
        cats = {key(p) for p in self.pool.values()}
        seen = {t["key"] for t in self.state["tried"]}
        cands, bad = [], []
        for p in raw:
            try:
                change, d, s = validate(p, self.parent, cats)
                k = sha(canon({x: p[x] for x in p if x not in ("why", "predict", "confidence")}))[:12]
                if k in seen:
                    raise Bad("repeat")
                seen.add(k)
                cands.append(dict(proposal=p, d=d, scope=s, key=k, change=change))
            except Bad as e:
                bad.append(dict(p=p, why=str(e)))
            if len(cands) == self.a["slots"]:
                break
        self.log(step="proposals", valid=len(cands), invalid=bad[:20], raw=len(raw))
        json.dump(cands, open(done, "w"), ensure_ascii=False)
        return cands

    def run_round(self):
        self.dir = f"{self.work}/r{self.round}"
        os.makedirs(self.dir, exist_ok=True)
        self.short = f"{self.a['label'][:16]}_r{self.round}"
        self.parent = self.pick_parent()
        ctx, TP0 = self.read_residual()
        cands = self.propose_round(ctx)
        h0file = f"{self.cache}/ev/{gid(self.parent)}/a.h0x2.jsonl"
        screen = self.screen_ids(TP0)
        self.rows_file("screen", screen)
        pf = self.evaluate_now(self.parent, "screen", 2)
        P_s = load_traces(pf)
        kids = []
        for k, c in enumerate(cands):
            tag = f"{self.short}_{c['key']}"
            try:
                if c["proposal"]["op"] == "gate":
                    f = self.write_gate(c["proposal"], c["change"], c["scope"], tag, h0file)
                    self.wait()
                    if os.path.exists(f + ".failed") or not os.path.exists(f):
                        raise Bad("write failed")
                    g = self.gate_child(f, tag)
                else:
                    g = self.child_of(c["proposal"], c["change"])
                    if "vision" in c["change"]:
                        self.build(g)
                        self.wait()
                        if not os.path.exists(f"{self.model_path(g)}/config.json"):
                            raise Bad("checkpoint failed")
                kids.append(dict(c, genome=g, gid=gid(g)))
            except (Bad, KeyError, ValueError, IndexError, OSError) as e:
                self.record(c, None, False, f"write: {e}")
        # the placebo: same size of change, no aim; a screen gain under it is noise
        placebo = self.placebo()
        rows_of = lambda c: ("screen", 2) if c["scope"] == "all" else (self.rows_name(c["scope"]), 8)
        for c in kids:
            r, sm = rows_of(c)
            if r != "screen":
                self.evaluate(self.parent, r, sm)
            self.evaluate(c["genome"], r, sm)
            self.evaluate(c["genome"], "screen", 2)
        pl_out = self.evaluate(placebo, "screen", 2)
        self.wait()
        fit_all = paired_or_none(P_s, pl_out, sorted(P_s), self.pool)
        pl_gain = fit_all["gain"] if fit_all else 0.
        self.log(step="placebo", screen_gain=pl_gain)
        for c in kids:
            r, sm = rows_of(c)
            pfile, cfile = f"{self.cache}/ev/{gid(self.parent)}/a.{r}x{sm}.jsonl", f"{self.cache}/ev/{c['gid']}/a.{r}x{sm}.jsonl"
            if not (os.path.exists(pfile) and os.path.exists(cfile)):
                c["screen"] = None
                continue
            P, C = load_traces(pfile), load_traces(cfile)
            ids = [i for i in P if half(i) == 0]
            c["screen"] = paired_or_none(P, cfile, ids, self.pool, macro=False)
            c["score_test"] = score_test(column(P, C, ids))
            if r != "screen":
                sf = f"{self.cache}/ev/{c['gid']}/a.screenx2.jsonl"
                c["screen_all"] = paired_or_none(P_s, sf, sorted(P_s), self.pool) if os.path.exists(sf) else None
            self.log(step="screen", gid=c["gid"], op=c["proposal"]["op"], d=c["d"], scope=c["scope"], gain=c["screen"] and c["screen"]["gain"], score_test=c["score_test"], whole=c.get("screen_all") and c["screen_all"]["gain"])
        # to the test: at most stage2 by the gain on half 0, and above the placebo when the scope is the whole pool
        rank = sorted([c for c in kids if c.get("screen")], key=lambda c: -(c["screen"]["gain"] * (1 if c["scope"] == "all" else c["screen"]["n"] / max(1, len(screen)))))
        go = [c for c in rank if c["screen"]["gain"] > 0 and (c["scope"] != "all" or c["screen"]["gain"] > pl_gain)][: self.a["stage2"]]
        for c in rank:
            if c not in go:
                self.record(c, None, False, "not chosen on half 0")
        joinable = self.join_candidate()
        if joinable and len(go) >= self.a["stage2"]:
            go = go[:-1]
        if joinable:
            go.append(joinable)
        self.rows_file("h1", self.h1ids)
        self.evaluate(self.parent, "h1", 2)
        for c in go:
            self.evaluate(c["genome"], "h1", 2)
            if c["scope"] != "all":
                self.evaluate(c["genome"], self.rows_name(c["scope"]), 8)
                self.evaluate(self.parent, self.rows_name(c["scope"]), 8)
        self.wait()
        for c in go:
            self.decide(c)
        self.round += 1
        self.save()

    def screen_ids(self, TP0):
        """A fixed part of half 0 by dataset, the same for every checkpoint of every lineage."""
        by = collections.defaultdict(list)
        for i in sorted(TP0, key=lambda x: zlib.crc32(("s" + x).encode())):
            by[i.split("/")[0]].append(i)
        per = max(1, self.a["screen"] // max(1, len(by)))
        return [i for v in by.values() for i in v[:per]]

    def placebo(self):
        rng = random.Random(zlib.crc32(f"pl{self.round}{self.a['seed']}".encode()))
        L = rng.randrange(TLAYERS)
        return dict(vision=self.parent["vision"], engine=self.parent["engine"] + [{"op": "scale", "param": WRITERS[rng.choice(list(WRITERS))](L), "alpha": rng.choice([.9, 1.1])}], dirs=list(self.parent.get("dirs", [])))

    def pick_parent(self):
        """A real lineage: the parent is the accepted child with the best half 0 score, or the start. With --fixed-parent the start stays the parent (the ablation of the audit's root cause 2)."""
        root = self.state["root"]
        if self.a.get("fixed_parent") or not self.state["accepted"]:
            return root
        return max(self.state["accepted"] + [dict(root, fit_acc=-1)], key=lambda g: g.get("fit_acc", -1))

    def join_candidate(self):
        """The default combination: the two accepted children of this parent that were not yet joined, as ONE checkpoint, tested like any child."""
        kids = [g for g in self.state["accepted"] if g.get("parent") == gid(self.parent)]
        for i, A in enumerate(kids):
            for B in kids[i + 1:]:
                k = sha(A["gid"] + B["gid"])[:12]
                if k in {t["key"] for t in self.state["tried"]}:
                    continue
                try:
                    j = join(A, B, self.parent)
                except ValueError as e:
                    self.log(step="join refused", a=A["gid"], b=B["gid"], why=str(e))
                    continue
                self.build(j)
                self.wait()
                return dict(proposal=dict(op="join", a=A["gid"], b=B["gid"]), d=A["d"] + B["d"], scope="all", key=k, change={}, genome=j, gid=gid(j), joined=(A["gid"], B["gid"]))
        return None

    def decide(self, c):
        """Written in prereg.json before any job ran; nothing in it depends on the operator."""
        a = self.a["acceptance"]
        alpha = a["alpha"] / a["tests"]
        g = c["genome"]
        pfile = f"{self.cache}/ev/{gid(self.parent)}/a.h1x2.jsonl"
        cfile = f"{self.cache}/ev/{gid(g)}/a.h1x2.jsonl"
        if not os.path.exists(cfile) or not os.path.exists(pfile):
            return self.record(c, None, False, "test evaluation failed")
        P, C = load_traces(pfile), load_traces(cfile)
        ids = sorted(P)
        assert all(half(i) == 1 for i in ids)
        try:
            whole = paired(P, C, ids, self.pool, alpha=alpha, macro=False)
        except ValueError as e:
            return self.record(c, None, False, f"partial: {e}")
        out = dict(whole=whole, behaviour=dict(parent=behaviour(P, ids), child=behaviour(C, ids)))
        out["datasets"] = {b: paired(P, C, [i for i in ids if i.split("/")[0] == b], self.pool, alpha=.025, macro=False) for b in sorted({i.split("/")[0] for i in ids})}
        out["classes"] = {k: paired(P, C, [i for i in ids if class_of(self.pool[i]) == k], self.pool, alpha=.025) for k in CLASSES if any(class_of(self.pool[i]) == k for i in ids)}
        harm = [b for b, r in out["datasets"].items() if r["hi"] < 0]
        out["harm"] = harm
        why = []
        if c["scope"] == "all":
            if not (whole["lo"] > 0 and whole["gain"] >= a["min_whole"]):
                why.append(f"whole {100 * whole['gain']:+.1f}, lower bound {100 * whole['lo']:+.1f} at corrected alpha, needs above 0 and at least {100 * a['min_whole']:+.0f}")
        else:
            r = self.rows_name(c["scope"])
            G, same = self.scope_rows(c["scope"])
            PC, CC = load_traces(f"{self.cache}/ev/{gid(self.parent)}/a.{r}x8.jsonl"), load_traces(f"{self.cache}/ev/{gid(g)}/a.{r}x8.jsonl")
            G1 = [i for i in G if half(i) == 1]
            cat = paired(PC, CC, G1, self.pool, alpha=alpha, macro=False)
            out["category"] = cat
            out["reach"] = reach(P, [i for i in G1 if i in P], len(ids))
            rest = [i for i in ids if i not in set(G1)]
            out["rest"] = paired(P, C, rest, self.pool, alpha=.025)
            out["transfer_same_class_iid"] = paired(P, C, [i for i in same if i in P], self.pool, alpha=.025) if any(i in P for i in same) else None
            if not (cat["lo"] > 0 and cat["gain"] >= a["min_category"]):
                why.append(f"category {100 * cat['gain']:+.1f}, lower bound {100 * cat['lo']:+.1f} at corrected alpha, needs above 0 and at least {100 * a['min_category']:+.0f}")
            if out["rest"]["hi"] < 0:
                why.append(f"rest of the pool {100 * out['rest']['gain']:+.1f}, significantly worse")
        if harm and c["scope"] == "all":
            why.append(f"datasets significantly worse: {harm}")
        b0, b1 = out["behaviour"]["parent"], out["behaviour"]["child"]
        out["through_truncation"] = bool(b1["acc"] > b0["acc"] and b1["fin_acc"] <= b0["fin_acc"])
        if c.get("joined"):
            A, B = [x for x in self.state["accepted"] if x["gid"] in c["joined"]]
            TA, TB = load_traces(f"{self.cache}/ev/{A['gid']}/a.h1x2.jsonl"), load_traces(f"{self.cache}/ev/{B['gid']}/a.h1x2.jsonl")
            out["epsilon_AB"] = interaction(P, TA, TB, C, ids, self.pool)
        ok = not why
        self.record(c, out, ok, "; ".join(why) or "accepted")
        if ok:
            T0 = load_traces(self.evaluate_now(g, "screen", 2))
            acc = dict(g, gid=gid(g), parent=gid(self.parent), d=c["d"], fit_acc=sum(rate(T0[i]) for i in T0) / len(T0))
            self.state["accepted"].append(acc)

    def record(self, c, out, ok, why):
        self.state["tried"].append(dict(round=self.round, op=c["proposal"]["op"], proposal={k: v for k, v in c["proposal"].items() if k != "why"}, key=c["key"], accepted=ok, fit=(c.get("screen") or {}).get("gain")))
        self.log(step="decision", gid=c.get("gid"), op=c["proposal"]["op"], d=c["d"], scope=c["scope"], parent=gid(self.parent), accepted=ok, why=why, why_proposed=c["proposal"].get("why"), result=out)
        self.say(f"r{self.round} {c['proposal']['op']} d={c['d']} scope {c['scope']}: {'ACCEPTED' if ok else 'rejected'}, {why[:200]}")
        self.save()

    def run(self):
        reg = json.load(open(self.a["registry"]))
        self.ood_ids = {i for v in reg["classes"].values() for i in v["ood"]} | set(reg["general_ood"])
        leak = self.ood_ids & set(self.pool)
        if leak:
            raise SystemExit(f"{len(leak)} OOD rows in the IID file")
        self.h1ids = [i for i in self.pool if half(i) == 1]
        self.rows_file("h0", [i for i in self.pool if half(i) == 0])
        if not self.state["root"]:
            self.state["root"] = dict(self.a["root"], gid=gid(self.a["root"]))
            self.save()
        while self.round < self.a["rounds"]:
            self.run_round()
        self.say("lineage done")


def paired_or_none(P, childfile, ids, pool, macro=True):
    if not childfile or not os.path.exists(childfile):
        return None
    try:
        return paired(P, load_traces(childfile), ids, pool, macro=macro)
    except ValueError:
        return None


# ---------------------------------------------------------------- protocol -------------------------------------------------------------------
STARTS = {"stock": dict(vision=None, engine=[]), "p8eq": dict(vision=dict(patch=8, kernel="box", max=4096, min=3600, merge=0), engine=[])}


def cmd_plan(a):
    """Freezes the acceptance before any job. The prereg is written once; its sha goes into the ledger."""
    if a.driver not in DRIVERS or any(x in a.driver.lower() for x in FORBIDDEN_MODELS):
        raise SystemExit(f"driver must be one of {list(DRIVERS)}")
    os.makedirs(a.work, exist_ok=True)
    f = f"{a.work}/prereg.json"
    if os.path.exists(f):
        raise SystemExit(f"{f} exists")
    reg = json.load(open(a.registry))
    pool = read_jsonl([a.iid])
    n1 = sum(half(i) == 1 for i in pool)
    bench = collections.Counter(i.split("/")[0] for i in pool if half(i) == 1)
    tests = a.rounds * a.stage2
    alpha = a.alpha / tests
    se2 = sum((a.sd ** 2 / n) / len(bench) ** 2 for n in bench.values())
    from statistics import NormalDist
    z = NormalDist().inv_cdf
    mde_w = (z(1 - alpha) + z(.8)) * math.sqrt(se2)
    print(f"half 1: {n1} problems, {len(bench)} datasets; {tests} tests in this lineage, alpha {alpha:.4f} one-sided")
    print(f"whole pool, macro: smallest gain with 80 % power {100 * mde_w:.1f} points at per-problem sd {100 * a.sd:.1f}; pooled, not macro: {100 * mde(a.sd, n1, alpha):.1f}; minimum for acceptance {100 * a.min_whole:.0f} (noise floor about +2)")
    cats = collections.Counter(str(p.get("category")) for i, p in pool.items() if half(i) == 1)
    print("category, 8 draws, per-problem sd 30: problems for a gain of 20 points " + str(math.ceil(((z(1 - alpha) + z(.8)) * .30 / .20) ** 2)) + f"; categories with at least that many: {sum(v >= ((z(1 - alpha) + z(.8)) * .30 / .20) ** 2 for v in cats.values())} of {len(cats)}")
    if mde_w > a.max_mde and not a.force:
        raise SystemExit(f"power: {100 * mde_w:.1f} points is above {100 * a.max_mde:.1f}. Larger pool, fewer tests, or --force")
    label = f"{a.start}_{a.driver}" + ("_exp" if a.experience else "") + ("_fixedparent" if a.fixed_parent else "")
    pr = dict(label=label, written=NOW(), driver=a.driver, start=a.start, root=STARTS[a.start], stock=a.stock, models=a.models, cache=a.cache, iidfile=a.iid, registry=a.registry, ood_sha=reg["ood_sha"],
              experience=a.experience, fixed_parent=a.fixed_parent, notes_pool=a.notes_pool, seed=a.seed, rounds=a.rounds, slots=a.slots, stage2=a.stage2, screen=a.screen, dirlib=a.dirlib,
              acceptance=dict(alpha=a.alpha, tests=tests, min_whole=a.min_whole, min_category=a.min_category, rule="scope all: macro gain on half 1, lower bound at alpha/tests above 0, gain at least min_whole, no dataset significantly worse at 95 %. scope cat: category gain on half 1 over 8 draws, lower bound at alpha/tests above 0, gain at least min_category, rest of the pool not significantly worse at 95 %. Same-class transfer is reported, never blocks. Same rule for every operator."),
              power=dict(sd=a.sd, mde_whole=mde_w))
    json.dump(pr, open(f, "w"), indent=1, ensure_ascii=False)
    print(f"{label}: prereg sha {sha(open(f).read())[:12]}")


def cmd_finalize(a):
    """Once. Final child = the best accepted on half 0; scored with its start on the frozen lists, by class. Refuses a second reading or a changed registry."""
    pr = json.load(open(f"{a.work}/prereg.json"))
    st = json.load(open(f"{a.work}/state.json"))
    reg = json.load(open(a.registry))
    if st["round"] < pr["rounds"] and not a.stopped_early:
        raise SystemExit("the lineage has not finished")
    now = sha(canon({c: v["ood"] for c, v in reg["classes"].items()}) + canon(reg["general_ood"]))
    if now != pr["ood_sha"]:
        raise SystemExit("the frozen OOD lists changed")
    stamp = f"{a.work}/ood.stamp"
    if os.path.exists(stamp):
        print("already queued", open(stamp).read())
    root = st["root"]
    final = max(st["accepted"] + [dict(root, fit_acc=-1, gid=root["gid"])], key=lambda g: g.get("fit_acc", -1))
    ood = read_jsonl(reg["ood_files"])
    keep = [i for c, v in reg["classes"].items() if c not in reg["dropped"] and v["clean_test"] for i in v["ood"]] + reg["general_ood"]
    rows = f"{pr['cache']}/rows/ood.jsonl"
    if not os.path.exists(rows):
        with open(rows, "w") as o:
            for i in sorted(set(keep)):
                o.write(json.dumps(ood[i], ensure_ascii=False) + "\n")
    q = Queue()
    res = {}
    for name, g in (("start", root), ("final", final)):
        i = gid(g)
        d = f"{pr['cache']}/ood/{i}"
        os.makedirs(d, exist_ok=True)
        res[name] = f"{d}/a.oodx2.jsonl"
        arm = f"{pr['cache']}/arms/{i}.json"
        if not os.path.exists(arm):
            json.dump([{"label": "a", "effort": "medium", "engine": g["engine"]}], open(arm, "w"))
        model = pr["stock"] if not g["vision"] else f"{pr['models']}/{sha(canon(g['vision']))[:12]}"
        if not os.path.exists(res[name]):
            q.submit(f"vh_ood_{i[:10]}", f"python -u rsi/vharness.py job-eval {d} oodx2 {arm} --samples 2 --sets oodx2:{rows} --curve 8192 --model {model} --pool {rows} " + TAIL.format(lib=pr["dirlib"]))
    open(stamp, "w").write(json.dumps(dict(t=NOW(), final=final["gid"], start=root["gid"])))
    if all(os.path.exists(f) for f in res.values()):
        P, C = load_traces(res["start"]), load_traces(res["final"])
        tab = {}
        for c, v in reg["classes"].items():
            if c in reg["dropped"] or not v["clean_test"]:
                tab[c] = "dropped" if c in reg["dropped"] else "no clean test set"
                continue
            tab[c] = paired(P, C, [i for i in v["ood"] if i in P], ood, alpha=.025)
        tab["general"] = paired(P, C, [i for i in reg["general_ood"] if i in P], ood, alpha=.025)
        json.dump(tab, open(f"{a.work}/ood.json", "w"), indent=1)
        print(json.dumps(tab, indent=1))


def cmd_report(a):
    """The story: every accepted edit with operator and d, the three numbers kept apart, then the arms against each other. Delta_RSI needs finished OOD files of both arms on the same rows."""
    arms = {}
    for w in a.work:
        pr = json.load(open(f"{w}/prereg.json"))
        print(f"\n== {pr['label']} (prereg {pr['written']})")
        rows = [json.loads(l) for l in open(f"{w}/ledger.jsonl")] if os.path.exists(f"{w}/ledger.jsonl") else []
        for r in rows:
            if r["step"] == "decision":
                x = r.get("result") or {}
                pc = lambda k: (f"{100 * x[k]['gain']:+.1f} [{100 * x[k]['lo']:+.1f},{100 * x[k]['hi']:+.1f}]" if x.get(k) else "-")
                tok = x.get("behaviour")
                beh = f" tokens {tok['parent']['tokens']:.0f}>{tok['child']['tokens']:.0f} trunc {tok['parent']['trunc']:.3f}>{tok['child']['trunc']:.3f} fin {tok['parent']['fin_acc']:.3f}>{tok['child']['fin_acc']:.3f}" if tok else ""
                print(f"r{r['round']} {'ACC' if r['accepted'] else 'rej'} {r['op']} d={r['d']} {r['scope']}: category {pc('category')} whole {pc('whole')} transfer {pc('transfer_same_class_iid')}{beh}{' THROUGH TRUNCATION' if x.get('through_truncation') else ''}{' eps ' + str(round(100 * x['epsilon_AB']['epsilon'], 1)) if x.get('epsilon_AB') else ''} | {r['why'][:120]}")
        n = [r for r in rows if r["step"] == "decision"]
        print(f"proposals valid {sum(r.get('valid', 0) for r in rows if r['step'] == 'proposals')}, tested {sum(1 for r in n if r.get('result'))}, accepted {sum(r['accepted'] for r in n)}; accepted d: {[r['d'] for r in n if r['accepted']]}")
        if os.path.exists(f"{w}/ood.json"):
            arms[pr["label"]] = (pr, json.load(open(f"{w}/ood.json")))
    for x, (pa, ta) in arms.items():
        for y, (pb, tb) in arms.items():
            if x < y and pa["start"] == pb["start"]:
                print(f"Delta_RSI {x} minus {y}, final minus start, by class: " + ", ".join(f"{c} {100 * (ta[c]['gain'] - tb[c]['gain']):+.1f}" for c in ta if isinstance(ta[c], dict) and isinstance(tb.get(c), dict)) + " (unpaired difference of paired gains, the same seed and rows; give it no interval until the two final files are paired)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "job-propose":
        return cmd_job_propose(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "job-eval":
        return cmd_job_eval(sys.argv[2:])
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("register"); p.add_argument("registry"); p.add_argument("--iid-files", nargs="+", required=True); p.add_argument("--ood-files", nargs="+", required=True)
    p = sp.add_parser("drop-class"); p.add_argument("registry"); p.add_argument("cls"); p.add_argument("--reason", required=True)
    p = sp.add_parser("plan"); p.add_argument("work"); p.add_argument("--registry", required=True); p.add_argument("--iid", required=True); p.add_argument("--start", required=True, choices=list(STARTS)); p.add_argument("--driver", required=True)
    p.add_argument("--experience"); p.add_argument("--fixed-parent", action="store_true"); p.add_argument("--notes-pool"); p.add_argument("--seed", type=int, default=0); p.add_argument("--rounds", type=int, default=3); p.add_argument("--slots", type=int, default=4); p.add_argument("--stage2", type=int, default=2); p.add_argument("--screen", type=int, default=600)
    p.add_argument("--stock", default=STOCK); p.add_argument("--models", default=f"{D}/models/vh"); p.add_argument("--cache", default=f"{W}/vh/cache"); p.add_argument("--dirlib", default=DIRLIB)
    p.add_argument("--alpha", type=float, default=.05); p.add_argument("--min-whole", type=float, default=.02); p.add_argument("--min-category", type=float, default=.10); p.add_argument("--sd", type=float, default=.277); p.add_argument("--max-mde", type=float, default=.03); p.add_argument("--force", action="store_true")
    p = sp.add_parser("run"); p.add_argument("work")
    p = sp.add_parser("relay"); p.add_argument("work", nargs="+"); p.add_argument("--once", action="store_true")
    p = sp.add_parser("finalize"); p.add_argument("work"); p.add_argument("--registry", required=True); p.add_argument("--i-have-frozen-the-lineage", action="store_true", required=True); p.add_argument("--stopped-early", action="store_true")
    p = sp.add_parser("report"); p.add_argument("work", nargs="+")
    a = ap.parse_args()
    if a.cmd == "plan" and a.experience and a.driver != "model":
        pass
    {"register": cmd_register, "drop-class": cmd_drop_class, "plan": cmd_plan, "relay": cmd_relay, "finalize": cmd_finalize, "report": cmd_report, "run": lambda x: Lineage(x.work).run()}[a.cmd](a)


def cmd_job_eval(argv):
    """GPU job: probe_base writes <label>.<set>.jsonl; a crash leaves .failed for the lineage instead of a wait."""
    d, name, arm = argv[0], argv[1], argv[2]
    out = f"{d}/a.{name}.jsonl"
    code = subprocess.run([sys.executable, "-u", os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_base.py"), d, "--probes", arm] + argv[3:]).returncode
    if code or not os.path.exists(out):
        open(out + ".failed", "w").write(f"probe_base exit {code}\n")
        sys.exit(code or 1)


if __name__ == "__main__":
    main()
