"""fable_int (internals agent, 2026-11-11): teacher-forced HF forwards of Qwen3.8-27B with a manual layer loop, capture / patch hooks.
Prompt building = score_worker_v41.build (tip t005r view: rows_xs.jsonl holds prefix + prompt and the 4096-floor image cache paths).
Sites: res (residual after block l), mix (mixer output), mlp (MLP output), head (input of o_proj / out_proj = per-head outputs),
neu (input of mlp.down_proj = neuron activations), qk (post-rope q, k of full-attention blocks), qproj / vproj (full-attention blocks).
"""
import contextlib, json, os, time
import numpy as np, torch
from PIL import Image

MODEL = os.environ.get("FI_MODEL", "/dev/shm/selfedit/models/Qwen3.8-27B")
TRACES = os.environ.get("FI_TRACES", "/dev/shm/fable_dir/gset_t005r_traces.json")
ROWS = os.environ.get("FI_ROWS", "/home/bedicloud/sharestore2/zxc/selfedit/fable_dir/xs/rows_xs.jsonl")      # "" = build the prompt from the traces file (Venus: v4 engine layout)
FACTOR = 32


def _smart_resize(h, w, factor=FACTOR, min_pixels=65536, max_pixels=16777216):       # v4/engine_v4/common.py smart_resize
    import math
    hb = round(h / factor) * factor; wb = round(w / factor) * factor
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels); hb = max(factor, math.floor(h / beta / factor) * factor); wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w)); hb = math.ceil(h * beta / factor) * factor; wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def view_image(path, min_tokens=None, max_tokens=None):
    """v4/engine_v4/common.py image_url_for without the PNG round trip (lossless): RGB (RGBA on white), bicubic resize to the view's smart_resize size."""
    im = Image.open(path); im.load()
    if im.mode != "RGB":
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA"); bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3]); im = bg
        else:
            im = im.convert("RGB")
    if max_tokens and max_tokens * FACTOR ** 2 >= 16777216:
        max_tokens = None
    if min_tokens and min_tokens * FACTOR ** 2 <= 65536:
        min_tokens = None
    if not min_tokens and not max_tokens:
        return im
    mx = max_tokens * FACTOR ** 2 if max_tokens else 16777216; mn = min_tokens * FACTOR ** 2 if min_tokens else min(65536, mx)
    w, h = im.size; hb, wb = _smart_resize(h, w, min_pixels=mn, max_pixels=mx)
    return im.resize((wb, hb), Image.BICUBIC) if (hb, wb) != (h, w) else im
THINK_END = "</think>"


class StopFwd(Exception):
    pass


class Ctx:
    pass


class Lab:
    def __init__(self, device="cuda:0", f32=False, traces=TRACES, rows=ROWS, model=MODEL):
        from transformers import AutoProcessor
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M
        from torch.nn.utils.stateless import _reparametrize_module
        self.M = M
        self.dev, self.f32 = device, f32
        self.proc = AutoProcessor.from_pretrained(model); self.tk = self.proc.tokenizer
        t0 = time.time()
        if os.environ.get("GPU_MEM_FRACTION"):          # Vesta: stay inside the card quota given by gpu bg
            torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["GPU_MEM_FRACTION"])))
        self.model = M.Qwen3_5ForConditionalGeneration.from_pretrained(model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval().requires_grad_(False)
        self.lm = self.model.model.language_model
        self.layers = self.lm.layers; self.NL = len(self.layers); self.H = self.model.config.text_config.hidden_size
        self.FULL = [l for l in range(self.NL) if self.layers[l].block_type == "full_attention"]
        self.IMG = self.model.config.image_token_id
        self.END = self.tk.convert_tokens_to_ids("<|im_end|>")
        self.TEND = self.tk.convert_tokens_to_ids(THINK_END)
        self.D = json.load(open(traces)) if traces else {}
        self.R = {p["id"]: p for p in map(json.loads, open(rows))} if rows else {}
        pf, pt = os.environ.get("FI_PATH_FROM", ""), os.environ.get("FI_PATH_TO", "")      # image-path prefix of the traces file -> this machine (node0015: /data/fd/mingdazhang/selfedit/ -> /home/bedicloud/sharestore2/zxc/selfedit/)
        if pf:
            for v in self.D.values():
                if isinstance(v, dict) and isinstance(v.get("image_paths"), list):
                    v["image_paths"] = [q.replace(pf, pt) for q in v["image_paths"]]
        self.ST = dict(cap={}, out={}, patch={}, l=-1, kw={})
        if True:
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            self.model.model.visual.to(torch.float32)
            self.lm.embed_tokens.register_forward_hook(lambda m, i, o: o.float() if self.f32 else None)

            def conv(x):
                if torch.is_tensor(x):
                    return x.float() if x.is_floating_point() else x
                if isinstance(x, (tuple, list)):
                    return type(x)(conv(v) for v in x)
                return x

            def wrap(mod):
                orig = mod.forward

                def fwd(*args, **kw):
                    if not self.f32:
                        return orig(*args, **kw)
                    args = conv(args); kw = {k: conv(v) for k, v in kw.items()}
                    with _reparametrize_module(mod, {n: p.float() for n, p in mod.named_parameters()}):
                        return orig(*args, **kw)
                mod.forward = fwd
            for blk in self.layers:
                wrap(blk)
            wrap(self.lm.norm)
            self.Wlm = None
        self._hooks()
        print(f"[lab] model on {device} in {time.time() - t0:.0f}s, f32={f32}, {self.NL} layers, full {self.FULL}", flush=True)

    # ------------------------------------------------------------------ hooks
    def _hooks(self):
        ST, M = self.ST, self.M

        def out_hook(kind, l):
            def f(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                fn = ST["patch"].get((kind, l))
                if fn is not None:
                    o = fn(o)
                c = ST["cap"].get(kind)
                if c is not None:
                    r = c(l, o)
                    if r is not None:
                        ST["out"][(kind, l)] = r
                if fn is not None:
                    return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
            return f

        def pre_hook(kind, l):
            def f(mod, args):
                x = args[0]
                fn = ST["patch"].get((kind, l))
                if fn is not None:
                    x = fn(x)
                c = ST["cap"].get(kind)
                if c is not None:
                    r = c(l, x)
                    if r is not None:
                        ST["out"][(kind, l)] = r
                if fn is not None:
                    return (x,) + tuple(args[1:])
            return f

        def kw_hook(l):
            def f(mod, args, kwargs):
                if ST.get("grab"):
                    ST["kw"][l] = (args, dict(kwargs))
                    if l == 3:
                        raise StopFwd()
            return f
        for l, lay in enumerate(self.layers):
            lin = lay.block_type == "linear_attention"
            mixer = lay.linear_attn if lin else lay.self_attn
            mixer.register_forward_hook(out_hook("mix", l))
            lay.mlp.register_forward_hook(out_hook("mlp", l))
            lay.mlp.down_proj.register_forward_pre_hook(pre_hook("neu", l))
            (mixer.out_proj if lin else mixer.o_proj).register_forward_pre_hook(pre_hook("head", l))
            lay.post_attention_layernorm.register_forward_hook(out_hook("mlpin", l))
            if not lin:
                mixer.q_proj.register_forward_hook(out_hook("qproj", l))
                mixer.v_proj.register_forward_hook(out_hook("vproj", l))
            if l in (0, 3):
                lay.register_forward_pre_hook(kw_hook(l), with_kwargs=True)
        _rot0 = M.apply_rotary_pos_emb

        def _rot(q, k, cos, sin, *a, **kw):
            qe, ke = _rot0(q, k, cos, sin, *a, **kw)
            c = ST["cap"].get("qk")
            if c is not None:
                r = c(ST["l"], (qe, ke))
                if r is not None:
                    ST["out"][("qk", ST["l"])] = r
            return qe, ke
        M.apply_rotary_pos_emb = _rot

    # ------------------------------------------------------------------ inputs
    def gen_ids(self, rid, k, max_reason=0):
        """token ids of reasoning + '\\n</think>\\n\\n' + content (+ <|im_end|>), and n_reason = number of reasoning tokens before the
        '\\n</think>' part (prediction of </think> happens at the position of the token before it)."""
        s = self.D[rid]["samples"][str(k)]
        max_reason = max_reason or int(os.environ.get("FI_MAX_REASON", 0))       # teacher-force only the first FI_MAX_REASON thinking tokens of long reads
        gen = self.tk((s["reasoning"] or "") + "\n</think>\n\n" + (s["content"] or ""), add_special_tokens=False).input_ids
        if s.get("finish") == "stop":
            gen = gen + [self.END]
        te = gen.index(self.TEND) if self.TEND in gen else len(gen)
        if max_reason and te > max_reason:
            gen = gen[:max_reason]; te = len(gen)           # truncated inside the reasoning: no </think>
        return gen, te

    def prep(self, rid, effort, gen, system=None):
        """-> Ctx with embeddings (image features scattered), layer kwargs for both block types, P (prompt length), T, ids."""
        if self.R:
            p = self.R[rid]
            ims = [Image.open(q).convert("RGB") for q in p["image_paths"]]
            text = p["prompt"]; parts = text.split("<image>")
            if len(parts) - 1 == len(ims) and ims:
                content = []
                for j, t in enumerate(parts):
                    if t.strip():
                        content.append({"type": "text", "text": t.strip()})
                    if j < len(ims):
                        content.append({"type": "image", "image": ims[j]})
            else:
                content = [{"type": "image", "image": q} for q in ims] + [{"type": "text", "text": text}]
        elif self.D[rid].get("parts"):                  # message parts as sent (t007c export): image parts with their own min / max image tokens, text parts verbatim; the routed prefix is its own text part
            v = self.D[rid]
            drop = effort == "noproc" or (effort == "medium" and bool(os.environ.get("FI_BASE_NOPREFIX")))
            if effort in ("proc", "noproc"):
                effort = "medium"
            content = []
            for j, p in enumerate(v["parts"]):
                if p["type"] == "image":
                    content.append({"type": "image", "image": view_image(p["path"], p.get("min_image_tokens"), p.get("max_image_tokens"))})
                elif not (drop and j == v.get("prefix_part")):
                    content.append({"type": "text", "text": p["text"]})
        else:                                           # v4 engine layout (common.content_for): prefix as its own text part right before the prompt text
            v = self.D[rid]; st = v.get("settings") or {}
            ims = [view_image(q, st.get("min_image_tokens"), st.get("max_image_tokens")) for q in v["image_paths"]]
            # modes for the routed-procedure contrast: "proc" = the prompt as read (medium effort, routed prefix kept); "noproc", or "medium" under FI_BASE_NOPREFIX=1 = the same prompt with the prefix text removed
            drop = effort == "noproc" or (effort == "medium" and bool(os.environ.get("FI_BASE_NOPREFIX")))
            if effort in ("proc", "noproc"):
                effort = "medium"
            addpre = None
            if effort == "hind":                        # hindsight: one mechanical system line stating the row's gold final answer (a token-level teacher on labelled rows; never exists at deployment)
                system = os.environ.get("FI_HIND", "The correct final answer to this question is: {gold}.").format(gold=str(v["answer"]).strip()); effort = "medium"
            elif effort.startswith("sys@"):               # the base prompt plus a system message read from a file (interface notes)
                system = open(effort[4:]).read().strip(); effort = "medium"
            elif effort.startswith("pre@"):             # the base prompt plus an always-on prefix text read from a file (placed like a routed prefix)
                addpre = open(effort[4:]).read().strip(); effort = "medium"
            pre = [{"type": "text", "text": v["prefix"].rstrip() + "\n\n"}] if (v.get("prefix") and not drop) else []
            if addpre:
                pre = [{"type": "text", "text": addpre + "\n\n"}] + pre
            parts = v["prompt"].split("<image>")
            if len(parts) - 1 == len(ims) and ims:
                content = list(pre)
                for j, t in enumerate(parts):
                    if t.strip():
                        content.append({"type": "text", "text": t.strip()})
                    if j < len(ims):
                        content.append({"type": "image", "image": ims[j]})
            else:
                content = [{"type": "image", "image": q} for q in ims] + pre + [{"type": "text", "text": v["prompt"]}]
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt", reasoning_effort=effort, enable_thinking=True)
        P = inp["input_ids"].shape[1]
        ids = torch.cat([inp["input_ids"][0], torch.tensor(gen, dtype=inp["input_ids"].dtype)])[None].to(self.dev); T = ids.shape[1]
        feed = dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=inp["pixel_values"].to(self.dev, torch.float32),
                    image_grid_thw=inp["image_grid_thw"].to(self.dev))
        if "mm_token_type_ids" in inp:
            feed["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"][0], torch.zeros(T - P, dtype=inp["mm_token_type_ids"].dtype)])[None].to(self.dev)
        ST = self.ST; ST["grab"] = True; ST["kw"] = {}
        try:
            with torch.inference_mode():
                self.model.model(**feed, use_cache=False)
        except StopFwd:
            pass
        ST["grab"] = False
        c = Ctx()
        c.rid, c.effort, c.P, c.T, c.ids = rid, effort, P, T, ids[0]
        c.h0 = ST["kw"][0][0][0]
        c.kw_lin, c.kw_full = ST["kw"][0][1], ST["kw"][3][1]
        ST["kw"] = {}
        c.isimg = (c.ids[:P] == self.IMG)
        return c

    # ------------------------------------------------------------------ forward
    @torch.inference_mode()
    def run(self, c, start=0, h=None, stop=None, res_patch=None, res_cap=None, full_mask=None):
        """manual layer loop from block `start` (h = residual entering it; default embeddings) to `stop` (exclusive; default all).
        res_patch {l: fn(h) -> h} applied to the residual AFTER block l; res_cap fn(l, h) called after block l (after patching).
        full_mask {l: attention_mask} replaces the mask of full-attention block l. Returns the residual after the last block run."""
        ST = self.ST
        h = c.h0 if h is None else h
        stop = self.NL if stop is None else stop
        for l in range(start, stop):
            lay = self.layers[l]
            kw = c.kw_full if lay.block_type == "full_attention" else c.kw_lin
            if full_mask is not None and l in full_mask:
                kw = dict(kw, attention_mask=full_mask[l])
            ST["l"] = l
            h = lay(h, **kw)
            if res_patch is not None and l in res_patch:
                h = res_patch[l](h)
            if res_cap is not None:
                res_cap(l, h)
        return h

    @torch.inference_mode()
    def logp(self, hfin, chunk=256):
        """hfin [n, H] residual after the last block -> log-softmax over the vocabulary, float32 [n, V]"""
        out = []
        W = self.model.lm_head.weight
        for i in range(0, hfin.shape[0], chunk):
            x = self.lm.norm(hfin[i: i + chunk][None])[0]
            z = (x.float() @ W.float().T) if self.f32 else self.model.lm_head(x).float()
            out.append(torch.log_softmax(z, -1))
        return torch.cat(out)

    def clear(self):
        self.ST["cap"] = {}; self.ST["patch"] = {}; self.ST["out"] = {}

    # ------------------------------------------------------------------ helpers
    def align(self, cm, cx):
        """medium ctx cm and xhigh ctx cx of the same row / tokens: xhigh = medium with one inserted block [a, a + S).
        -> a, S; medium position p >= a maps to xhigh position p + S."""
        im, ix = cm.ids, cx.ids
        S = cx.T - cm.T
        n = min(cm.T, cx.T)
        neq = (im[:n] != ix[:n]).nonzero()
        a = int(neq[0]) if len(neq) else n
        assert torch.equal(im[a:], ix[a + S:]), "templates differ by more than one inserted block"
        return a, S

    def classes(self, c, a=None, S=0):
        """position classes of ctx c (xhigh: pass a, S): dict name -> LongTensor of positions.
        sys (inserted block), img, txt (other prompt tokens before the tail), tail (last 5 prompt tokens: <|im_end|>\\n<|im_start|>assistant\\n<think>\\n ...), gen."""
        P, T = c.P, c.T
        pos = torch.arange(T, device=c.ids.device)
        cls = torch.full((T,), 2, device=c.ids.device)       # txt
        cls[:P][c.isimg] = 1
        cls[P - 5: P] = 3
        cls[P:] = 4
        if a is not None and S:
            cls[a: a + S] = 0
        return {n: pos[cls == j] for j, n in enumerate(["sys", "img", "txt", "tail", "gen"])}, cls


def kl(lp_ref, lp):
    """KL(ref || cond) per position, float32 [n]"""
    return (lp_ref.exp() * (lp_ref - lp)).sum(-1)


def pick_rows(D, per_bench=8, seed=1111, gen_lo=300, gen_hi=3000, prompt_hi=6500, roles=("show", "monitor"), want_finish=True):
    """-> list of (row id, sample) : per benchmark `per_bench` rows (sorted ids, seeded shuffle) with a sample inside the length window."""
    rng = np.random.default_rng(seed)
    by = {}
    for r in sorted(D):
        v = D[r]
        if v["role"] not in roles:
            continue
        ks = [int(k) for k, s in v["samples"].items() if gen_lo <= int(s["tokens"]) <= gen_hi and int(s["prompt_tokens"]) <= prompt_hi and s.get("reasoning") and (s.get("finish") == "stop" or not want_finish)]
        if ks:
            by.setdefault(v["bench"], []).append((r, sorted(ks)))
    out = []
    for b in sorted(by):
        L = by[b]; idx = rng.permutation(len(L))[:per_bench]
        for i in sorted(idx):
            r, ks = L[i]
            out.append((r, int(ks[int(rng.integers(len(ks)))])))
    return out


def pick_sample(D, rid, gen_lo=300, gen_hi=3000, prompt_hi=6500, seed=0):
    """a sample of row rid inside the length window (seeded choice), or None"""
    ks = sorted(int(k) for k, s in D[rid]["samples"].items() if gen_lo <= int(s["tokens"]) <= gen_hi and int(s.get("prompt_tokens") or 0) <= prompt_hi and s.get("reasoning") and s.get("finish") == "stop")
    if not ks:
        return None
    import zlib
    return ks[(zlib.crc32(rid.encode()) + seed) % len(ks)]


def items_from_ids(D, path, **kw):
    out = []
    for l in open(path):
        r = l.strip()
        if r and r in D:
            k = pick_sample(D, r, **kw)
            if k is not None:
                out.append((r, k))
    return out
