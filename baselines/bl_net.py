"""Model placement for the baselines: FSDP2 data-parallel over the training cards. Full fp32 master weights, sharded across cards; forward in bf16 through FSDP2 mixed precision;
gradients reduced in fp32; AdamW moments kept in CPU RAM, moved through the card one parameter at a time at the optimizer step. Works on the actor (lm_head) and on the critic (value head)."""
import datetime, gc, glob, json, math, os, time
import torch, torch.nn as nn, torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import checkpoint
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoImageProcessor
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from bl_common import *
from surgery import load_image

IMG = 248056


class Net(nn.Module):
    def __init__(self, hf, critic=False, chunk=2048):
        super().__init__()
        self.hf, self.critic, self.chunk = hf, critic, chunk
        if critic:
            self.vhead = nn.Linear(hf.config.text_config.hidden_size, 1, bias=True)   # small init and bias .5: our adaptation, see protocol
            nn.init.normal_(self.vhead.weight, std=1e-3)
            nn.init.constant_(self.vhead.bias, .5)

    def forward(self, input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw, resp_mask):
        h = self.hf.model(input_ids=input_ids, attention_mask=attention_mask, mm_token_type_ids=mm_token_type_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw, use_cache=False)[0]
        sel = resp_mask[:, 1:]
        hs = h[:, :-1][sel]
        if self.critic:
            return self.vhead(hs).squeeze(-1).float()
        lab = input_ids[:, 1:][sel]

        def f(x, y):
            return torch.log_softmax(self.hf.lm_head(x).float(), -1).gather(-1, y[..., None]).squeeze(-1)
        return torch.cat([checkpoint(f, hs[i:i + self.chunk], lab[i:i + self.chunk], use_reentrant=False) for i in range(0, hs.shape[0], self.chunk)])


def setup_dist():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=6))   # other ranks wait in a collective during long rollouts and evaluations
    r = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return r, dist.get_world_size(), init_device_mesh("cuda", (dist.get_world_size(),))


def build(mesh, critic=False, offload=False, freeze_vision=False, log=print):
    """Load the stock weights on CPU, fp32 master, shard block by block, so one rank peaks near one bf16 copy in RAM."""
    t0 = time.time()
    hf = Qwen3_5ForConditionalGeneration.from_pretrained(STOCK, dtype=torch.bfloat16)
    log("hf load s", round(time.time() - t0, 1), "params B", round(sum(p.numel() for p in hf.parameters()) / 1e9, 3), "visual B", round(sum(p.numel() for p in hf.model.visual.parameters()) / 1e9, 3))
    hf.config.use_cache = False
    hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
    if critic:
        del hf.lm_head
        hf.lm_head = None
    net = Net(hf, critic=critic)
    net.train()
    if freeze_vision:
        for p in hf.model.visual.parameters():
            p.requires_grad_(False)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True)
    kw = dict(mesh=mesh, mp_policy=mp)
    if offload:
        kw["offload_policy"] = CPUOffloadPolicy(pin_memory=os.environ.get("BL_PIN", "1") == "1")   # each round moves every shard to the card about three times
    for b in list(hf.model.language_model.layers) + list(hf.model.visual.blocks):
        b.to(torch.float32)
        fully_shard(b, **kw)
    net.to(torch.float32) if False else None
    for n, p in list(net.named_parameters()):
        if p.dtype == torch.bfloat16:
            p.data = p.data.float()
    fully_shard(net, **kw)
    gc.collect()
    torch.cuda.empty_cache()
    log("fsdp ready s", round(time.time() - t0, 1), "gpu GB", round(torch.cuda.memory_allocated() / 2**30, 1))
    return net


class OffloadAdamW:
    """AdamW on the local shards. Moments live in CPU RAM as fp32. Same update as torch.optim.AdamW (decoupled weight decay, bias correction)."""
    def __init__(self, params, betas=(0.9, 0.95), eps=1e-8, wd=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.b1, self.b2, self.eps, self.wd = betas[0], betas[1], eps, wd
        self.m = [None] * len(self.params); self.v = [None] * len(self.params); self.t = 0

    def grad_norm(self):
        s = torch.zeros((), device="cuda", dtype=torch.float64)
        for p in self.params:
            if p.grad is not None:
                s += p.grad._local_tensor.float().pow(2).sum().double()
        dist.all_reduce(s)
        return float(s.sqrt())

    @torch.no_grad()
    def step(self, lr, clip=1.0):
        gn = self.grad_norm()
        scale = min(1.0, clip / (gn + 1e-6)) if clip else 1.0
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad._local_tensor.float()
            if scale != 1.0:
                g = g * scale
            x = p._local_tensor
            if self.m[i] is None:
                m, v = torch.zeros_like(g), torch.zeros_like(g)
            else:
                m, v = self.m[i].to(g.device, non_blocking=True), self.v[i].to(g.device, non_blocking=True)
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            if self.wd:
                x.mul_(1 - lr * self.wd)
            x.addcdiv_(m / bc1, (v / bc2).sqrt_().add_(self.eps), value=-lr)
            if self.m[i] is None:
                self.m[i], self.v[i] = (m.cpu().pin_memory(), v.cpu().pin_memory()) if m.is_cuda else (m, v)   # pinned host buffers: DMA both ways
            elif m.is_cuda:
                self.m[i].copy_(m); self.v[i].copy_(v)
            del m, v, g
        for p in self.params:
            p.grad = None
        return gn


# ---------------------------------------------------------------- batches
_proc = None


def image_tensors(paths):
    global _proc
    if _proc is None:
        _proc = AutoImageProcessor.from_pretrained(STOCK)
    return _proc(images=[load_image(p) for p in paths], return_tensors="pt")


def collate(samples, device="cuda"):
    """samples: dict(prompt_token_ids, token_ids, image_paths). Right padding. resp_mask marks response tokens."""
    L = max(len(s["prompt_token_ids"]) + len(s["token_ids"]) for s in samples)
    n = len(samples)
    ids = torch.zeros(n, L, dtype=torch.long)
    am = torch.zeros(n, L, dtype=torch.long)
    rm = torch.zeros(n, L, dtype=torch.bool)
    pv, gr = [], []
    for i, s in enumerate(samples):
        a, b = s["prompt_token_ids"], s["token_ids"]
        ids[i, :len(a) + len(b)] = torch.tensor(a + b)
        am[i, :len(a) + len(b)] = 1
        rm[i, len(a):len(a) + len(b)] = True
        it = image_tensors(s["image_paths"])
        pv.append(it["pixel_values"]); gr.append(it["image_grid_thw"])
        assert int((it["image_grid_thw"].prod(1) // 4).sum()) == a.count(IMG), (s.get("id"), it["image_grid_thw"].tolist(), a.count(IMG))
    return dict(input_ids=ids.to(device), attention_mask=am.to(device), mm_token_type_ids=(ids == IMG).long().to(device), pixel_values=torch.cat(pv).to(device), image_grid_thw=torch.cat(gr).to(device), resp_mask=rm.to(device))


def rounds(samples, world, max_tok):
    """Micro-batches of similar lengths (first-fit decreasing on padded tokens), grouped into rounds of one per rank, heaviest together: FSDP ranks move in step layer by layer, so a round costs what its heaviest rank costs.
    A batch too small to fill a round is split so that every rank gets real work. All ranks must run the same number of forward and backward passes; a slot still empty gets a copy of the smallest sample at weight 0."""
    ln = lambda i: len(samples[i]["prompt_token_ids"]) + len(samples[i]["token_ids"])
    bins = []
    for i in sorted(range(len(samples)), key=lambda i: -ln(i)):
        for b in bins:
            if b["max"] * (len(b["i"]) + 1) <= max_tok:
                b["i"].append(i)
                break
        else:
            bins.append(dict(max=ln(i), i=[i]))
    while len(bins) < world and any(len(b["i"]) > 1 for b in bins):
        b = max((b for b in bins if len(b["i"]) > 1), key=lambda b: b["max"] * len(b["i"]))
        k = len(b["i"]) // 2
        bins.append(dict(max=ln(b["i"][k]), i=b["i"][k:])); b["i"] = b["i"][:k]
    bins.sort(key=lambda b: -b["max"] * len(b["i"]))
    out = []
    for k in range(0, len(bins), world):
        r = [b["i"] for b in bins[k:k + world]]
        out.append(r + [[] for _ in range(world - len(r))])
    return out


def logp_targets(batch_samples, key):
    return torch.cat([torch.as_tensor(s[key], dtype=torch.float32) for s in batch_samples]).to("cuda")


def smi():
    import subprocess
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.replace("\n", " | ")
    except Exception as e:
        return repr(e)


# ---------------------------------------------------------------- export + checkpoint
def export_bf16(net, out_dir, rank, log=print):
    """Gather full tensors and write a checkpoint with the stock shard layout, names and per-tensor dtypes. Tensors the HF model does not hold (mtp.*) are copied from stock."""
    t0 = time.time()
    idx = json.load(open(f"{STOCK}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    tmp = out_dir + ".tmp"
    if rank == 0:
        for d in (tmp, out_dir):     # engines are asleep here and read nothing; free the space first so no moment holds two copies
            if os.path.exists(d):
                shutil.rmtree(d)
        os.makedirs(tmp)
    named = {n.removeprefix("hf."): p for n, p in net.named_parameters()}
    by = {}
    for k, f in wm.items():
        by.setdefault(f, []).append(k)
    for f in sorted(by):
        tensors = {}
        src = safe_open(f"{STOCK}/{f}", "pt") if rank == 0 else None
        for k in by[f]:
            if k in named:
                t = named[k].to("cuda").full_tensor()
                if rank == 0:
                    tensors[k] = t.to(src.get_slice(k).get_dtype() and _st_dtype(src.get_slice(k).get_dtype())).cpu()
                del t
            elif rank == 0:
                tensors[k] = src.get_tensor(k)
        if rank == 0:
            save_file(tensors, f"{tmp}/{f}", metadata=dict(format="pt"))
    if rank == 0:
        for g in os.listdir(STOCK):
            if not g.endswith(".safetensors") and g not in ("README.md", "LICENSE", "crc32.txt"):
                shutil.copy(f"{STOCK}/{g}", f"{tmp}/{g}")
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.rename(tmp, out_dir)
    dist.barrier()
    log("export s", round(time.time() - t0, 1))


def _st_dtype(s):
    return dict(bf16=torch.bfloat16, BF16=torch.bfloat16, f32=torch.float32, F32=torch.float32, f16=torch.float16, F16=torch.float16, float32=torch.float32, bfloat16=torch.bfloat16)[str(s)]


def save_ckpt(nets, opts, d, meta, rank, log=print, adam=True):
    """Per rank: local fp32 shards and AdamW moments, one file per rank per net. Same card count needed for resume. adam=False keeps weights only; Adam restarts from zero on resume."""
    t0 = time.time()
    os.makedirs(d, exist_ok=True)
    for name, net in nets.items():
        o = opts[name]
        blob = dict(w={n: p._local_tensor.cpu() for n, p in net.named_parameters()}, t=o.t if adam else 0,
                    m=[x if adam else None for x in o.m], v=[x if adam else None for x in o.v])
        torch.save(blob, f"{d}/{name}.rank{rank}.tmp")
        os.rename(f"{d}/{name}.rank{rank}.tmp", f"{d}/{name}.rank{rank}.pt")
        del blob
    if rank == 0:
        json.dump(meta, open(f"{d}/meta.json", "w"))
    dist.barrier()
    log("ckpt saved s", round(time.time() - t0, 1), "dir GB", round(sum(os.path.getsize(f) for f in glob.glob(f"{d}/*")) / 2**30, 1))


def _local(p):
    return getattr(p, "_local_tensor", p)


def ckpt_bytes(nets, adam):
    """This rank's checkpoint size in bytes: fp32 weights, plus two fp32 moments with adam."""
    return sum(_local(p).numel() * 4 for n in nets.values() for p in n.parameters()) * (3 if adam else 1)


def save_ckpt_safe(nets, opts, d, meta, rank, log=print, adam=True, room=None):
    """Write new, verify, then swap; the old checkpoint is removed only after the new one is complete and verified (coordinator 11-08).
    1. Each rank writes {d}.new/{name}.rank{r}.pt (tmp + rename). 2. Each rank reloads its file with mmap and compares every tensor bit for bit with
    its live shard, and the Adam step; rank 0 writes meta.json and reads it back. 3. All ranks agree (all_reduce), then rank 0 renames d -> d.old,
    d.new -> d, and only then removes d.old. A failed verify removes d.new and keeps d. room(need_gb) -> None or the reason there is no room; with
    no room the checkpoint is skipped and the old one stays. Returns dict(ok, why, s, gb)."""
    t0 = time.time(); nd, od = d + ".new", d + ".old"
    if rank == 0:
        if not os.path.isdir(d) and os.path.isdir(od):       # a crash between the two renames: the old checkpoint is still whole
            os.rename(od, d); log("ckpt: restored", od, "->", d)
        if os.path.isdir(nd):                                # an earlier attempt that never finished; never the only copy
            import shutil; shutil.rmtree(nd); log("ckpt: removed unfinished", nd)
    dist.barrier()
    need = torch.tensor(float(ckpt_bytes(nets, adam)) / 2**30, dtype=torch.float64); dist.all_reduce(need); need = float(need)
    o = [room(need) if (room and rank == 0) else None]; dist.broadcast_object_list(o, 0)
    if o[0]:
        return dict(ok=False, why="no room: " + o[0], s=round(time.time() - t0, 1), gb=round(need, 3))
    os.makedirs(nd, exist_ok=True); dist.barrier()
    for name, net in nets.items():
        o_ = opts[name]
        blob = dict(w={n: _local(p).detach().cpu() for n, p in net.named_parameters()}, t=o_.t if adam else 0,
                    m=[x if adam else None for x in o_.m], v=[x if adam else None for x in o_.v])
        torch.save(blob, f"{nd}/{name}.rank{rank}.tmp"); os.rename(f"{nd}/{name}.rank{rank}.tmp", f"{nd}/{name}.rank{rank}.pt")
        del blob
    if rank == 0:
        json.dump(meta, open(f"{nd}/meta.json.tmp", "w")); os.rename(f"{nd}/meta.json.tmp", f"{nd}/meta.json")
    dist.barrier(); t1 = time.time()
    bad = []
    for name, net in nets.items():
        o_ = opts[name]
        try:
            b = torch.load(f"{nd}/{name}.rank{rank}.pt", map_location="cpu", weights_only=True, mmap=True)
            live = dict(net.named_parameters())
            if set(b["w"]) != set(live):
                bad.append(f"{name}: parameter names differ")
            for n, p in live.items():
                x = _local(p).detach()
                if b["w"][n].shape != x.shape or b["w"][n].dtype != x.dtype or not torch.equal(b["w"][n], x.cpu()):
                    bad.append(f"{name}.{n}"); break
            if adam:
                if b["t"] != o_.t or len(b["m"]) != len(o_.m) or any((u is None) != (v is None) or (u is not None and not torch.equal(u, v.cpu())) for u, v in zip(b["m"] + b["v"], o_.m + o_.v)):
                    bad.append(f"{name}: adam state differs")
            del b
        except Exception as e:
            bad.append(f"{name}: {type(e).__name__} {e}")
    if rank == 0:
        try:
            if json.load(open(f"{nd}/meta.json")) != json.loads(json.dumps(meta)):
                bad.append("meta.json differs")
        except Exception as e:
            bad.append(f"meta.json: {e}")
    nbad = torch.tensor(float(len(bad))); dist.all_reduce(nbad)
    o = [bad]; dist.broadcast_object_list(o, 0)
    if float(nbad) > 0:
        if rank == 0:
            import shutil; shutil.rmtree(nd)
        dist.barrier()
        return dict(ok=False, why=f"verify failed on {int(float(nbad))} item(s), old checkpoint kept; rank 0: {bad[:3]}", s=round(time.time() - t0, 1), gb=round(need, 3))
    if rank == 0:
        if os.path.isdir(d):
            os.rename(d, od)
        os.rename(nd, d)
        if os.path.isdir(od):
            import shutil; shutil.rmtree(od)
    dist.barrier()
    gb = sum(os.path.getsize(f) for f in glob.glob(f"{d}/*")) / 2**30
    log("ckpt saved and verified s", round(time.time() - t0, 1), "verify s", round(time.time() - t1, 1), "dir GB", round(gb, 1))
    return dict(ok=True, why=None, s=round(time.time() - t0, 1), s_verify=round(time.time() - t1, 1), gb=round(gb, 3), need_gb=round(need, 3))


def load_ckpt(nets, opts, d, rank, log=print):
    t0 = time.time()
    if not os.path.isdir(d) and os.path.isdir(d + ".old"):     # save_ckpt_safe crashed between its two renames: the old checkpoint is whole
        log("ckpt: loading", d + ".old"); d = d + ".old"
    for name, net in nets.items():
        o = opts[name]
        blob = torch.load(f"{d}/{name}.rank{rank}.pt", map_location="cpu", weights_only=True, mmap=True)
        for n, p in net.named_parameters():
            p._local_tensor.copy_(blob["w"][n].to(p._local_tensor.device))
        o.t, o.m, o.v = blob["t"], [None if x is None else x.clone().pin_memory() for x in blob["m"]], [None if x is None else x.clone().pin_memory() for x in blob["v"]]
        del blob
    dist.barrier()
    log("ckpt loaded s", round(time.time() - t0, 1))
    return json.load(open(f"{d}/meta.json"))


@torch.no_grad()
def checksum(net):
    s = torch.zeros((), device="cuda", dtype=torch.float64)
    for p in net.parameters():
        s += p._local_tensor.double().abs().sum()
    dist.all_reduce(s)
    return float(s)
