"""The optimal-control read of the model: depth as time, each writer's scale as a control input, x_{l+1} = x_l + u_l f_l(x_l). The derivative of an answer margin to u_l is the switching function, the inner product of the adjoint at that layer with what the writer wrote, and one backward pass gives it for every writer at once.
This tests that law against the +/- dose scan, with no fitting: the same read and the same items as rsi/eye_exam.py, the scan's gain per writer against the adjoint's. Then along an edit: the gradient at dose x gives H times the edit, so the peak dose is predicted before any arm is run, and the gain of each writer after the edit (g' = g + H d) is predicted before any rescan.
Also per output channel: the same pass splits each writer's derivative by channel, on two halves of the items, for the reliability and size of the finer control the scan cannot reach.

Usage: python -u rsi/mm_adjoint.py --model M --dir EXAM --out OUT [--tasks cell,trace,ring] [--arms JSON] [--doses .5,1] [--device cuda:0] [--traces TRACES --max-tokens 800] [--at DOSES.json] [--draws 2 --radius .1] [--no-channels]
--traces reads the margin at the \\boxed{ the model reached after its own reasoning instead of with thinking off. --at keeps the writers at those scales, for a next round. --draws adds antithetic random-sign reads of every writer: their mean is the gain smoothed over the radius, their difference the diagonal curvature.
Writes OUT.base.npz (per item derivatives and margins, writer norms, channel derivatives by half), OUT.arms.jsonl (per arm, per dose: margin and gradient per task), OUT.txt.
"""
import argparse, json, re, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tasks", default="cell,trace,ring")
ap.add_argument("--arms")
ap.add_argument("--doses", default=".5,1")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--traces")
ap.add_argument("--max-tokens", type=int, default=800)
ap.add_argument("--at")
ap.add_argument("--draws", type=int, default=0)
ap.add_argument("--radius", type=float, default=.1)
ap.add_argument("--no-channels", action="store_true")
ap.add_argument("--max-patches", type=int, default=16384)
args = ap.parse_args()

from transformers import AutoProcessor
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
proc = AutoProcessor.from_pretrained(args.model)
tk = proc.tokenizer
model = M.Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
model.to(args.device).eval().requires_grad_(False)
if args.no_channels:                # real images are large: keep only one layer's activations at a time; train() only reaches the checkpoint switch, this model has no dropout
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
ENDS = "ABCDEFGH"


def writers():
    """Label to module, for every last linear before a residual add, named as in the scan."""
    out = {}
    for L, layer in enumerate(model.model.language_model.layers):
        attn = layer.self_attn.o_proj if hasattr(layer, "self_attn") and layer.self_attn is not None else layer.linear_attn.out_proj
        out[f"t{L}.attn"] = attn
        if hasattr(layer.mlp, "down_proj"):
            out[f"t{L}.mlp"] = layer.mlp.down_proj
    for N, block in enumerate(model.model.visual.blocks):
        out[f"v{N}.attn"] = block.attn.proj
        out[f"v{N}.mlp"] = block.mlp.linear_fc2
    return out


W = writers()
NAMES = list(W)
print(f"{len(NAMES)} writers", flush=True)
u = {k: torch.ones((), device=args.device, dtype=torch.float32, requires_grad=True) for k in NAMES}
keep = {}
track = {"on": False}


def hook(k):
    def fn(mod, inp, out):
        y = out * u[k].to(out.dtype)
        if track["on"]:
            y.retain_grad()
            keep[k] = y
        return y
    return fn


for k, mod in W.items():
    mod.register_forward_hook(hook(k))
items = [json.loads(l) for l in open(args.dir + "/items.jsonl")]
tasks = args.tasks.split(",") if args.tasks != "all" else sorted({p["task"] for p in items})
items = [p for p in items if p["task"] in tasks]
own = {}
if args.traces:
    for l in open(args.traces):
        t = json.loads(l)
        if t["id"] not in own and t["finish"] == "stop" and t["tokens"] <= args.max_tokens and "\\boxed{" in (t["content"] or ""):
            own[t["id"]] = t
    items = [p for p in items if p["id"] in own]
prep = []
for p in items:
    opts = list(ENDS[:int(p["level"])]) if p["task"] == "trace" else sorted(set(re.findall(r"\(([A-Z])\)", p["prompt"]) or re.findall(r"(?:^|\s)([A-Z])\.\s", p["prompt"])) | {p["answer"]})
    ids = [tk("\\boxed{" + x, add_special_tokens=False).input_ids for x in opts]
    n = next(i for i in range(len(ids[0]) + 1) if i == len(ids[0]) or any(i >= len(y) or y[i] != ids[0][i] for y in ids))
    if len(opts) < 2 or any(len(y) <= n for y in ids):
        continue
    mode = dict(reasoning_effort="medium") if args.traces else dict(enable_thinking=False)
    inp = proc.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": x} for x in p["image_paths"]] + [{"type": "text", "text": p["prompt"]}]}], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", **mode)
    said = []
    if args.traces:
        t = own[p["id"]]
        said = tk(t["reasoning"] + "\n</think>\n\n" + t["content"][:t["content"].rindex("\\boxed{")], add_special_tokens=False).input_ids
    if inp["pixel_values"].shape[0] > args.max_patches:
        continue
    tail = torch.tensor(said + ids[0][:n], dtype=torch.long)[None]
    feed = {"input_ids": torch.cat([inp["input_ids"], tail], 1), "attention_mask": torch.ones((1, inp["input_ids"].shape[1] + tail.shape[1]), dtype=torch.long)}
    if "mm_token_type_ids" in inp:
        feed["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros_like(tail)], 1)
    feed.update(pixel_values=inp["pixel_values"], image_grid_thw=inp["image_grid_thw"])
    prep.append(dict(id=p["id"], task=p["task"], level=p["level"], cand=[y[n] for y in ids], right=opts.index(p["answer"]), feed=feed))
print(f"{len(prep)} items", flush=True)
TASK = np.array([p["task"] for p in prep])


def forward(p):
    feed = {a: b.to(args.device) for a, b in p["feed"].items()}
    feed["pixel_values"] = feed["pixel_values"].to(torch.bfloat16)
    with torch.enable_grad():
        out = model(**feed, use_cache=False, logits_to_keep=1)
        lp = out.logits[0, -1].float().log_softmax(-1)[torch.tensor(p["cand"], device=args.device)]
        rest = torch.cat([lp[:p["right"]], lp[p["right"] + 1:]])
        margin = lp[p["right"]] - rest.logsumexp(0)
    return margin


AT = json.load(open(args.at)) if args.at else {}


def set_u(dose=None):
    """Scales on top of the round's starting point."""
    with torch.no_grad():
        for k in NAMES:
            u[k].fill_(AT.get(k, 1.0))
        for k, a in (dose or {}).items():
            u[k].fill_(AT.get(k, 1.0) * a)


def grads(margin):
    for k in NAMES:
        u[k].grad = None
    margin.backward()
    return np.array([float(u[k].grad) for k in NAMES])


def arm_dose(arm):
    """Writer to scale, from an engine arm; vision scales weight and bias together, which is the same as scaling the output."""
    out = {}
    for e in arm["engine"]:
        name = re.sub(r"\.(weight|bias)$", "", e["param"])
        hit = re.search(r"visual\.blocks\.(\d+)\.(attn|mlp)", name) or re.search(r"layers\.(\d+)\.(linear_attn|self_attn)", name)
        if hit:
            key = f"v{hit[1]}.{hit[2]}" if "visual" in name else f"t{hit[1]}.attn"
            out[key] = e["alpha"]
    return out


lines = []


def say(s):
    print(s, flush=True)
    lines.append(s)
    open(args.out + ".txt", "w").write("\n".join(lines) + "\n")


t0 = time.time()
G = np.zeros((len(prep), len(NAMES)))
MARG = np.zeros(len(prep))
NF = np.zeros(len(NAMES))
NP = np.zeros(len(NAMES))
CH = {}
for t in tasks:
    CH[t] = [{k: None for k in NAMES}, {k: None for k in NAMES}]
counts = {t: [0, 0] for t in tasks}
set_u()
track["on"] = not args.no_channels
skipped = []
for i, p in enumerate(prep):
    keep.clear()
    try:
        m = forward(p)
        G[i] = grads(m)
    except RuntimeError as e:
        if "out of memory" not in str(e) and "CUDA error" not in str(e):
            raise
        skipped.append(i)
        keep.clear()
        for k in NAMES:
            u[k].grad = None
        torch.cuda.empty_cache()
        print(f"item {i} too large, skipped", flush=True)
        continue
    MARG[i] = float(m)
    half = i % 2
    if args.no_channels:
        if i in (0, 9, 99) or i % 200 == 199:
            print(f"item {i + 1}: {time.time() - t0:.0f}s, mem {torch.cuda.max_memory_allocated() / 2 ** 30:.0f} GiB", flush=True)
        continue
    for j, k in enumerate(NAMES):
        y = keep[k]
        pa = y.grad.float()                 # the adjoint at the writer's output
        f = y.detach().float() / float(u[k])
        NF[j] += float((f ** 2).sum())
        NP[j] += float((pa ** 2).sum())
        s = (pa * f).sum(tuple(range(f.dim() - 1))).cpu().numpy()     # switching function by output channel
        cur = CH[p["task"]][half][k]
        CH[p["task"]][half][k] = s if cur is None else cur + s
    counts[p["task"]][half] += 1
    keep.clear()
    if i in (0, 9, 99) or i % 200 == 199:
        print(f"item {i + 1}: {time.time() - t0:.0f}s, mem {torch.cuda.max_memory_allocated() / 2 ** 30:.0f} GiB", flush=True)
track["on"] = False
NF /= len(prep)
NP /= len(prep)
cz = {f"{t}/{h}/{k}": CH[t][h][k] / max(1, counts[t][h]) for t in tasks for h in (0, 1) for k in NAMES if CH[t][h][k] is not None}
SMOOTH, DIAG = np.zeros_like(G), np.zeros_like(G)
rng = np.random.default_rng(0)
for draw in range(args.draws):
    v = dict(zip(NAMES, rng.choice([-args.radius, args.radius], len(NAMES))))
    vec = np.array([v[k] for k in NAMES])
    GP, GM = np.zeros_like(G), np.zeros_like(G)
    for sign, out in ((1, GP), (-1, GM)):
        set_u({k: 1 + sign * x for k, x in v.items()})
        for i, p in enumerate(prep):
            if i not in skipped:
                try:
                    out[i] = grads(forward(p))
                except RuntimeError as e:
                    if "out of memory" not in str(e) and "CUDA error" not in str(e):
                        raise
                    skipped.append(i)
                    for k in NAMES:
                        u[k].grad = None
                    torch.cuda.empty_cache()
    set_u()
    SMOOTH += (GP + GM) / 2 / args.draws
    DIAG += (GP - GM) / (2 * vec) / args.draws                # E[(g(+v) - g(-v)) / 2 v_i] = H_ii, the rest of the row averages out over draws
    print(f"draw {draw + 1}: {time.time() - t0:.0f}s", flush=True)
np.savez_compressed(args.out + ".base.npz", smooth=SMOOTH, diag=DIAG, skipped=np.array(skipped, dtype=int), G=G, margin=MARG, task=TASK, ids=np.array([p["id"] for p in prep]), level=np.array([p["level"] for p in prep]), names=np.array(NAMES), nf=NF, np=NP, **{("ch:" + a): b.astype(np.float32) for a, b in cz.items()})
say(f"base: {len(prep)} items, " + " ".join(f"{t} margin {MARG[TASK == t].mean():+.3f}" for t in tasks) + f" ({time.time() - t0:.0f}s)")
for arm in (json.load(open(args.arms)) if args.arms else []):
    dose = arm_dose(arm)
    if not dose:
        continue
    for x in [float(v) for v in args.doses.split(",")]:
        set_u({k: 1 + x * (a - 1) for k, a in dose.items()})
        Gx = np.zeros_like(G)
        Mx = np.zeros(len(prep))
        for i, p in enumerate(prep):
            m = forward(p)
            Gx[i] = grads(m)
            Mx[i] = float(m)
        set_u()
        d = np.array([dose.get(k, 1.0) - 1 for k in NAMES])
        row = {"label": arm["label"], "x": x}
        msg = []
        for t in tasks:
            s = TASK == t
            g0, gx = G[s].mean(0), Gx[s].mean(0)
            row[t] = dict(gain=float((Mx - MARG)[s].mean()), slope0=float(g0 @ d), slopex=float(gx @ d), gx=[round(float(v), 4) for v in gx])
            peak = (g0 @ d) / ((g0 @ d) - (gx @ d)) * x if (g0 @ d) > (gx @ d) else float("inf")
            msg.append(f"{t} gain {row[t]['gain']:+.3f} slope {g0 @ d:+.3f}->{gx @ d:+.3f} peak {peak:.2f}")
        with open(args.out + ".arms.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        say(f"{arm['label']} x{x}: " + "; ".join(msg) + f" ({time.time() - t0:.0f}s)")
say("done")
