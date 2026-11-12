"""score_worker.py (selfedit Track A1, fable_dir 2026-11-11): per-trace scores s = grad log p(trace | prompt) of cached tip traces.

  python score_worker.py --traces gset_t005r_traces.json --rows mixed --shard 0/2 --out OUTDIR --model /path/Qwen3.8-27B

--rows: "mixed" (rows whose 8 samples have mixed outcomes; the only rows that carry signal), "all", or a file with one row id per line.
For every trace of the selected rows (in sorted row order, then sample order, dealt round-robin to shards): rebuild the tip's prompt
(images resized client-side to >= 4096 image tokens, max 16384, Qwen2VL smart_resize factor 32 with the processor's resample; the row's
prefix + "\\n\\n" + prompt when it has one; chat template, reasoning_effort medium, thinking on), append the generated tokens
(reasoning + "\\n</think>\\n\\n" + content + <|im_end|> when finish == stop) and run ONE teacher-forced forward + backward of
log p(generated tokens | prompt) with frozen weights (bf16, vision tower + merger in float32, gradient checkpointing).
Saves OUTDIR/<row id with / -> __>__s<k>.npz per trace (skips existing ones, so a restart resumes):
  row, sample, ok, tokens (server count), gtok / ptok (our token counts) and tokens_ref / ptok_ref (the trace's), logp, sec, peak_gib
  dT        d logp_T / dT at T = 1 = sum_t (E_p[z_t] - z_t[y_t])  (analytic; top-k / top-p truncation ignored)
  g_attn    [64]  d logp / d u, u x (attention or linear-attention output) of each block, at u = 1
  g_mlp     [64]  same for the MLP output
  g_qtemp   [16]  same for u x query after q_norm in the full-attention blocks (3, 7, ..., 63) = attention-logit temperature 1/alpha
  g_merger  [1]   same for u x vision-merger output
  bias_rp   [3, 64, 64]  d logp / d b for an additive vector b after each block (residual stream), summed over generated / image-token /
                         prompt-text positions, projected by P [5120, 64] = standard normal (seed --proj-seed, default 1111) / sqrt(64)
  bias_full [3, 3, 5120] the unprojected vectors for layers 16, 32, 48 (order gen / img / txt x layers)
  proj_seed
"""
import argparse, json, os, time
import numpy as np, torch
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument("--traces", required=True); ap.add_argument("--rows", default="mixed"); ap.add_argument("--shard", default="0/1")
ap.add_argument("--out", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--chunk", type=int, default=1024); ap.add_argument("--max-len", type=int, default=32768)
ap.add_argument("--proj-seed", type=int, default=1111); ap.add_argument("--full-layers", default="16,32,48")
ap.add_argument("--path-from", default=""); ap.add_argument("--path-to", default="")
ap.add_argument("--img-cache", default=""); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--reverse", action="store_true", help="process this shard's traces from the end (a second site filling in from the other side)")
a = ap.parse_args()
si, sn = (int(x) for x in a.shard.split("/"))
os.makedirs(a.out, exist_ok=True)
CACHE = a.img_cache or os.path.join(a.out, "_img4096"); os.makedirs(CACHE, exist_ok=True)
from transformers import AutoProcessor
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from torch.utils.checkpoint import checkpoint

D = json.load(open(a.traces))
if a.rows == "mixed":
    rows = [r for r in sorted(D) if 0 < sum(bool(s["ok"]) for s in D[r]["samples"].values()) < len(D[r]["samples"])]
elif a.rows == "all":
    rows = sorted(D)
else:
    rows = [l.strip() for l in open(a.rows) if l.strip()]
items = [(r, int(k)) for r in rows for k in sorted(D[r]["samples"], key=int)][si::sn]
fname = lambda r, k: os.path.join(a.out, f"{r.replace('/', '__')}__s{k}.npz")
todo = [x for x in items if not os.path.exists(fname(*x))]
if a.reverse:
    todo = todo[::-1]
if a.limit:
    todo = todo[: a.limit]
print(f"rows {len(rows)} ({a.rows}); shard {si}/{sn}: {len(items)} traces, {len(items) - len(todo)} already saved, {len(todo)} to do", flush=True)
if not todo:
    raise SystemExit(0)

dev = a.device
proc = AutoProcessor.from_pretrained(a.model); tk = proc.tokenizer; RES = getattr(proc.image_processor, "resample", 3)
model = M.Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
model.eval().requires_grad_(False)
model.model.visual.to(torch.float32)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.train()   # no dropout in this family
layers = model.model.language_model.layers; NL = len(layers); H = model.config.text_config.hidden_size
FULL = [l for l in range(NL) if layers[l].block_type == "full_attention"]
IMG = model.config.image_token_id; END = tk.convert_tokens_to_ids("<|im_end|>")
FL = [int(x) for x in a.full_layers.split(",")]
PROJ = torch.tensor(np.random.default_rng(a.proj_seed).standard_normal((H, 64)) / 8.0, dtype=torch.float32, device=dev)
f32 = dict(device=dev, dtype=torch.float32)
B = torch.zeros(3, NL, H, requires_grad=True, **f32)
U_att = torch.ones(NL, requires_grad=True, **f32); U_mlp = torch.ones(NL, requires_grad=True, **f32)
U_tmp = torch.ones(len(FULL), requires_grad=True, **f32); U_mrg = torch.ones(1, requires_grad=True, **f32)
MASK = {}


def scale_out(u):
    def f(m, i, o):
        if isinstance(o, tuple):
            return ((o[0].float() * u).to(o[0].dtype),) + tuple(o[1:])
        return (o.float() * u).to(o.dtype)
    return f


for l in range(NL):
    att = layers[l].linear_attn if layers[l].block_type == "linear_attention" else layers[l].self_attn
    att.register_forward_hook(scale_out(U_att[l])); layers[l].mlp.register_forward_hook(scale_out(U_mlp[l]))
    def add_bias(m, i, o, l=l):
        h = o[0] if isinstance(o, tuple) else o
        h = h + (MASK["m"] @ B[:, l, :]).to(h.dtype)[None]
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
    layers[l].register_forward_hook(add_bias)
for j, l in enumerate(FULL):
    layers[l].self_attn.q_norm.register_forward_hook(scale_out(U_tmp[j]))
model.model.visual.merger.register_forward_hook(scale_out(U_mrg[0]))


def image(p):
    q = p.replace(a.path_from, a.path_to) if a.path_from else p
    c = os.path.join(CACHE, q.strip("/").replace("/", "__") + ".png")
    if not os.path.exists(c):
        im = Image.open(q).convert("RGB")
        h, w = smart_resize(im.height, im.width, factor=32, min_pixels=4096 * 32 * 32, max_pixels=16384 * 32 * 32)
        im.resize((w, h), resample=RES).save(c)
    return Image.open(c).convert("RGB")


def build(v, s):
    text = (v["prefix"] + "\n\n" + v["prompt"]) if v.get("prefix") else v["prompt"]
    ims = [image(p) for p in v["image_paths"]]; parts = text.split("<image>")
    if len(parts) - 1 == len(ims) and ims:
        content = []
        for k, t in enumerate(parts):
            if t.strip():
                content.append({"type": "text", "text": t.strip()})
            if k < len(ims):
                content.append({"type": "image", "image": ims[k]})
    else:
        content = [{"type": "image", "image": q} for q in ims] + [{"type": "text", "text": text}]
    inp = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt", reasoning_effort="medium", enable_thinking=True)
    gen = tk((s["reasoning"] or "") + "\n</think>\n\n" + (s["content"] or ""), add_special_tokens=False).input_ids
    if s.get("finish") == "stop":
        gen = gen + [END]
    return inp, gen[: a.max_len]


TOP_K, TOP_P = 20, 0.95        # the server's sampling (engine_v4/common.py SAMPLING): temperature 1.0, top_k 20, top_p 0.95


def keep_mask(z, y):
    """The behaviour policy's support at each position: top-k, then the nucleus on the renormalised top-k (a token is kept while the
    mass before it is < top_p), always containing the taken token. The score must be that of the policy that produced the trace:
    log p(y) = z_y - logsumexp over the kept set, not over the vocabulary (coordinator, 11-11)."""
    zt, it = z.topk(TOP_K, -1)
    p = torch.softmax(zt, -1)
    kept = (p.cumsum(-1) - p) < TOP_P
    m = torch.zeros_like(z, dtype=torch.bool).scatter_(-1, it, kept)
    inside = m.gather(-1, y[:, None])[:, 0]
    m.scatter_(-1, y[:, None], True)
    return m, inside


def lp_chunk(h, y):
    z = model.lm_head(h).float()
    m, _ = keep_mask(z.detach(), y)
    return (z.gather(-1, y[:, None])[:, 0] - torch.logsumexp(z.masked_fill(~m, float("-inf")), -1)).sum()



# ---- fd.py: finite-difference check of the scores on one trace (env FD_ROW, FD_S, FD_REF = dir of the other implementation's records)
import glob
ROW, SMP, REF = os.environ["FD_ROW"], int(os.environ["FD_S"]), os.environ["FD_REF"]
v = D[ROW]; s = v["samples"][str(SMP)]
inp, gen = build(v, s); P = inp["input_ids"].shape[1]
ids = torch.cat([inp["input_ids"][0], torch.tensor(gen, dtype=inp["input_ids"].dtype)])[None].to(dev); T = ids.shape[1]
m = torch.zeros(T, 3, **f32); m[P:, 0] = 1
isimg = ids[0, :P] == IMG; m[:P, 1] = isimg.float(); m[:P, 2] = (~isimg).float(); MASK["m"] = m
feed = dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=inp["pixel_values"].to(dev, torch.float32), image_grid_thw=inp["image_grid_thw"].to(dev))
if "mm_token_type_ids" in inp:
    feed["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"][0], torch.zeros(T - P, dtype=inp["mm_token_type_ids"].dtype)])[None].to(dev)
y = ids[0, P:]


def total_logp():
    with torch.no_grad():
        hs = model.model(**feed, use_cache=False).last_hidden_state[0]; hx = hs[P - 1: T - 1]; lp = 0.0
        for c0 in range(0, len(y), a.chunk):
            lp += float(lp_chunk(hx[c0: c0 + a.chunk], y[c0: c0 + a.chunk]))
        return lp


own = np.load(os.path.join(a.out, f"{ROW.replace('/', '__')}__s{SMP}.npz")); oth = np.load(os.path.join(REF, f"{ROW.replace('/', '__')}__s{SMP}.npz"))
base = total_logp(); print(f"FD {ROW}#{SMP} base logp {base:.3f} (record {float(own['logp']):.3f})", flush=True)
tests = [("g_attn", U_att, None), ("g_mlp", U_mlp, None), ("g_merger", U_mrg, None)]
for name, U, _ in tests:
    for tag, rec in (("own", own), ("other", oth)):
        g = torch.tensor(rec[name], **f32); d = g / g.norm()
        for e in (0.01, 0.03):
            with torch.no_grad(): U += e * d
            lp_p = total_logp()
            with torch.no_grad(): U -= 2 * e * d
            lp_m = total_logp()
            with torch.no_grad(): U += e * d
            fd = (lp_p - lp_m) / (2 * e)
            print(f"FD {name:8s} dir={tag:5s} e={e}: finite diff {fd:9.3f} | own grad.d {float(torch.tensor(own[name], **f32) @ d):9.3f} | other grad.d {float(torch.tensor(oth[name], **f32) @ d):9.3f}", flush=True)
# bias at layer 32 on generated positions, unprojected
L = 32; j = list(FL).index(L)
for tag, rec in (("own", own), ("other", oth)):
    g = torch.tensor(rec["bias_full"][0, j], **f32); d = g / g.norm()
    for e in (0.01, 0.03):
        with torch.no_grad(): B[0, L] += e * d
        lp_p = total_logp()
        with torch.no_grad(): B[0, L] -= 2 * e * d
        lp_m = total_logp()
        with torch.no_grad(): B[0, L] += e * d
        fd = (lp_p - lp_m) / (2 * e)
        print(f"FD bias32g dir={tag:5s} e={e}: finite diff {fd:9.3f} | own grad.d {float(torch.tensor(own['bias_full'][0, j], **f32) @ d):9.3f} | other grad.d {float(torch.tensor(oth['bias_full'][0, j], **f32) @ d):9.3f}", flush=True)
