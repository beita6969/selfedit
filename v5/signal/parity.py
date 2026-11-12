"""One-row autograd parity / cost check (coordinator 01:2x): teacher-forced forward + backward of the executor on one cached trace,
gradient checkpointing, vision tower float32, weights frozen; probes at three layers: residual bias b_L (zeros, d = hidden) on the
decoder layer output, scalar gates on the MLP output and on the attention / linear-attention output. Writes loss, gradients, seconds,
peak memory to OUT.npz/.json. Run once in the signal venv (fla kernels) and once in the vLLM venv (torch fallback); compare.py diffs.
usage: python parity.py OUT [--row ID --sample S] [--layers 8 31 54] [--tower bf16]"""
import argparse, base64, io, json, sys, time
import numpy as np, torch
sys.path.insert(0, "/workspace/fudan/mingdazhang/selfedit-v4/engine_v4")
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--row"); ap.add_argument("--sample")
ap.add_argument("--layers", type=int, nargs="+", default=[8, 31, 54])
ap.add_argument("--tower", default="fp32")
ap.add_argument("--target", type=int, default=9000)
ap.add_argument("--no-fla", action="store_true", help="block the fla package: torch fallback kernels")
a = ap.parse_args()
if a.no_fla:
    sys.modules["fla"] = None          # import fla -> ImportError: transformers keeps its torch implementation
M = "/data/fd/mingdazhang/selfedit/models/Qwen3.8-27B"
TR = "/data/fd/mingdazhang/selfedit/v4/tmp_coord_gset_t005r_traces.json"
import transformers
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as MQ
from PIL import Image
import common as C


def impl_of(f):
    cl = dict(zip(f.__code__.co_freevars, [c.cell_contents for c in (f.__closure__ or [])]))
    i = cl.get("implementation")
    return f"{getattr(i, '__module__', '?')}.{getattr(i, '__name__', '?')}" if i is not None else "?"


info = dict(torch=torch.__version__, transformers=transformers.__version__, python=sys.executable,
            chunk_impl=impl_of(MQ.torch_chunk_gated_delta_rule), conv_impl=impl_of(MQ.causal_conv1d_fn))
print(info, flush=True)
T = json.load(open(TR))
if a.row:
    row, s = a.row, a.sample
else:      # show / monitor row whose sample is closest to the target length (finished, with content)
    best = None
    for r, d in T.items():
        if d.get("role") not in ("show", "monitor"):
            continue
        for s_, x in (d.get("samples") or {}).items():
            if x.get("finish", "stop") != "stop" or not x.get("content"):
                continue
            g = abs(int(x.get("tokens") or 0) - a.target)
            if best is None or g < best[0]:
                best = (g, r, s_)
    _, row, s = best
d = T[row]; x = d["samples"][str(s)]
st = d.get("settings") or {}
print("trace", row, "sample", s, "role", d.get("role"), "tokens", x.get("tokens"), "ok", x.get("ok"), flush=True)
content, nt = C.content_for(d["prompt"], d["image_paths"], st.get("min_image_tokens"), st.get("max_image_tokens"), d.get("prefix"))
imgs, parts = [], []
for c in content:
    if c["type"] == "image_url":
        imgs.append(Image.open(io.BytesIO(base64.b64decode(c["image_url"]["url"].split(",", 1)[1]))).convert("RGB"))
        parts.append({"type": "image"})
    else:
        parts.append({"type": "text", "text": c["text"]})
proc = AutoProcessor.from_pretrained(M)
prompt = proc.apply_chat_template([{"role": "user", "content": parts}], add_generation_prompt=True, tokenize=False, enable_thinking=True)
if not prompt.rstrip().endswith("<think>"):
    prompt = prompt + "<think>\n"
ans = (x.get("reasoning") or "").strip() + "\n</think>\n\n" + (x.get("content") or "").strip() + "<|im_end|>"
enc = proc(text=[prompt], images=imgs, return_tensors="pt")
ans_ids = proc.tokenizer(ans, add_special_tokens=False, return_tensors="pt")["input_ids"]
ids = torch.cat([enc["input_ids"], ans_ids], 1)
n_prompt, n_ans = enc["input_ids"].shape[1], ans_ids.shape[1]
print(f"prompt tokens {n_prompt} (image tokens {sum(nt)}), answer tokens {n_ans}, total {ids.shape[1]}", flush=True)

t0 = time.time()
dev = torch.device("cuda:0")
try:
    model = Qwen3_5ForConditionalGeneration.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda:0")
except Exception as e:
    print("device_map load failed, fallback:", repr(e)[:200], flush=True)
    with torch.device("cuda:0"):
        model = Qwen3_5ForConditionalGeneration.from_pretrained(M, dtype=torch.bfloat16)
model.requires_grad_(False)
load_s = time.time() - t0
lt = model.config.text_config.layer_types
layers = model.model.language_model.layers
vis = model.model.visual
if a.tower == "fp32":
    vis.float()


def cast(o, dt):
    if torch.is_tensor(o):
        return o.to(dt) if o.is_floating_point() else o
    if isinstance(o, (tuple, list)):
        return type(o)(cast(v, dt) for v in o)
    if hasattr(o, "__dataclass_fields__") or hasattr(o, "keys"):
        for k in list(o.keys()):
            o[k] = cast(o[k], dt)
        return o
    return o


if a.tower == "fp32":
    vis.register_forward_pre_hook(lambda m, args, kw: (cast(args, torch.float32), cast(kw, torch.float32)), with_kwargs=True)
    vis.register_forward_hook(lambda m, i, o: cast(o, torch.bfloat16))
H = model.config.text_config.hidden_size
P = {}
for L in a.layers:
    P[f"b{L}"] = torch.zeros(H, device=dev, dtype=torch.float32, requires_grad=True)
    P[f"gm{L}"] = torch.ones((), device=dev, dtype=torch.float32, requires_grad=True)
    P[f"ga{L}"] = torch.ones((), device=dev, dtype=torch.float32, requires_grad=True)
    lay = layers[L]
    att = lay.linear_attn if lt[L] == "linear_attention" else lay.self_attn

    def first(o, f):
        return f(o) if torch.is_tensor(o) else (f(o[0]),) + tuple(o[1:])
    lay.register_forward_hook(lambda m, i, o, L=L: first(o, lambda t: t + P[f"b{L}"].to(t.dtype)))
    lay.mlp.register_forward_hook(lambda m, i, o, L=L: first(o, lambda t: t * P[f"gm{L}"].to(t.dtype)))
    att.register_forward_hook(lambda m, i, o, L=L: first(o, lambda t: t * P[f"ga{L}"].to(t.dtype)))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()
inp = {k: v.to(dev) for k, v in enc.items() if k != "input_ids"}
inp["input_ids"] = ids.to(dev)
inp.pop("attention_mask", None)
if "mm_token_type_ids" in inp:     # extend over the answer tokens (text = 0), else get_rope_index sees a length mismatch
    mt = inp["mm_token_type_ids"]
    inp["mm_token_type_ids"] = torch.cat([mt, torch.zeros(1, n_ans, dtype=mt.dtype, device=mt.device)], 1)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
base_mem = torch.cuda.memory_allocated()
t1 = time.time()
out = model.model(**inp, use_cache=False)
h = out.last_hidden_state[0]
tgt = ids[0].to(dev)
lab_pos = torch.arange(n_prompt - 1, ids.shape[1] - 1, device=dev)     # position p predicts token p + 1


def chunk_loss(hc, yc):
    return torch.nn.functional.cross_entropy(model.lm_head(hc).float(), yc, reduction="sum")


loss = 0.0
for i in range(0, len(lab_pos), 1024):
    p = lab_pos[i:i + 1024]
    loss = loss + torch.utils.checkpoint.checkpoint(chunk_loss, h[p], tgt[p + 1], use_reentrant=False)
loss = loss / len(lab_pos)
torch.cuda.synchronize(); t_fwd = time.time() - t1
loss.backward()
torch.cuda.synchronize(); t_all = time.time() - t1
peak = torch.cuda.max_memory_allocated()
res = dict(info, row=row, sample=s, layers=a.layers, layer_types=[lt[L] for L in a.layers], tower=a.tower, n_prompt=n_prompt, n_answer=n_ans,
           image_tokens=sum(nt), loss=float(loss), load_s=round(load_s, 1), fwd_s=round(t_fwd, 1), fwd_bwd_s=round(t_all, 1),
           weights_gb=round(base_mem / 2**30, 2), peak_gb=round(peak / 2**30, 2), reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
           card_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
print(json.dumps(res), flush=True)
np.savez(a.out + ".npz", **{k: v.grad.detach().float().cpu().numpy() for k, v in P.items()})
json.dump(res, open(a.out + ".json", "w"), indent=1)
