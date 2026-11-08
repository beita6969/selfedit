"""Write a note into the weights with repurposed MLP units of the last layer, one per token. Each unit: silu(b w.x) * (w.x) with w fitted to +1 at its key position and -1 elsewhere, so the gate gives a near-hard threshold and noise under it is squared away, where a plain linear memory (mm_recite.py) fires a little everywhere and hijacks the reasoning. down_proj column = alpha u. Units are the least used of the layer. Half 0 fits, half 1 reads.
Writes ARMS/<tag>.json (set on gate_proj, up_proj rows and down_proj columns), OUT.<tag>.txt.
"""
import argparse, hashlib, json, os, time, zlib
import numpy as np
import torch

ap = argparse.ArgumentParser()
for k in ("model", "traces", "others", "pool", "notes", "out", "arms"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--plain", default="")
ap.add_argument("--strict-half0", action="store_true")     # half 1 of --others reaches only the printed check, not the choice of units
ap.add_argument("--cat", default="")          # or take the problems of one category from the pool (~ = space)
ap.add_argument("--engine", default="")       # this generation's edits: applied first, their units are not reused
ap.add_argument("--variants", default="")     # problems reworded: half 0 records join the fit, so the key is the situation and not one wording
ap.add_argument("--read-variants", default="")  # reworded problems kept out of the fit, all halves: name<TAB>file,...
ap.add_argument("--neg-start", default="")    # prompts where the note must not start: half 0 join the fit at the first key, half 1 are read
ap.add_argument("--skip-units", type=int, default=0)      # rules written side by side in one generation must not take the same units
ap.add_argument("--hard", default="")
ap.add_argument("--hard-per", type=int, default=4)
ap.add_argument("--hard-max", type=int, default=40000)
ap.add_argument("--mine", type=int, default=3)
ap.add_argument("--layer", type=int, default=63)
ap.add_argument("--beta", type=float, default=8.)
ap.add_argument("--gap", type=float, default=12.)
ap.add_argument("--lam", type=float, default=.01)
ap.add_argument("--n-others", type=int, default=160)
ap.add_argument("--ctrl", type=int, default=12)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--effort", default="medium")     # 128 (11-08): the chat template's reasoning effort, the run's prereg gen effort; medium as before
ap.add_argument("--trace-cap", type=int, default=3000)      # 128 (11-08): prereg gate_trace_cap; traces are cut at this many tokens and others longer than it are skipped; 3000 as before
a = ap.parse_args()
a.cat = a.cat.strip("'")       # the node starts commands without a shell: a quote stays in the word

from transformers import AutoProcessor
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
L = a.layer
proc = AutoProcessor.from_pretrained(a.model)
tk = proc.tokenizer
model = M.Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
model.to(a.device).eval().requires_grad_(False)
lang = model.model.language_model
lay = lang.layers[L]
mlp = lay.mlp
norm, lm = lang.norm, model.lm_head
taken = []
if a.engine:      # the parent's edits, in order, same arithmetic as engine/surgery.py: its gates, scales and directions are in force while this one is fit
    for e in json.load(open(a.engine))[0]["engine"]:
        W = lang.get_submodule(e["param"]).weight.data
        if e["op"] == "scale":
            W *= float(e["alpha"])
        elif e["op"] == "amplify" and e.get("side", "out") == "out" and e.get("space", "param") == "param":
            u = torch.tensor(e["u"], device=a.device, dtype=torch.float32)
            u = u / u.norm()
            W += ((float(e["alpha"]) * u)[:, None] * (u @ W.float())[None, :]).to(W.dtype)
        elif e["op"] == "set" and "neurons" in e.get("sel", {}):
            idx = torch.tensor(e["sel"]["neurons"], device=a.device)
            val = torch.tensor(e["value"], device=a.device, dtype=W.dtype)
            if e["param"].endswith("down_proj"):
                W[:, idx] = val
            else:
                W[idx] = val
            if e["param"].startswith(f"layers.{L}.mlp."):
                taken += e["sel"]["neurons"]
        else:
            raise SystemExit(f"parent edit this writer cannot apply: {e['op']} {e['param']}")
G0, U0, D0 = mlp.gate_proj.weight.data.clone(), mlp.up_proj.weight.data.clone(), mlp.down_proj.weight.data.clone()
pool = {p["id"]: p for p in map(json.loads, open(a.pool))}
if a.cat:
    same = (lambda p: f"{p.get('bench') or p['id'].split('/')[0]}|{p['category']}" == a.cat.replace("~", " ")) if "|" in a.cat else (lambda p: p["category"] == a.cat.replace("~", " "))    # bench|category: the key hboost scopes by
    plain = {i: p for i, p in pool.items() if same(p) and p.get("image_paths")}
else:
    plain = {p["id"]: p for p in map(json.loads, open(a.plain))}
half = lambda i: zlib.crc32(i.encode()) % 2
rng = np.random.default_rng(0)
cap, want = {}, {}
lay.register_forward_hook(lambda m, i, o: cap.__setitem__("r", (o[0] if isinstance(o, tuple) else o)[0][want["idx"]].float()))
mlp.gate_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__("x", i[0][0][want["idx"]].float()))
mlp.down_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__("h", i[0][0][want["idx"]].float()))


def run(p, said, at):
    inp = proc.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": x} for x in p["image_paths"]] + [{"type": "text", "text": p["prompt"]}]}], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", reasoning_effort=a.effort)
    tail = torch.tensor(said, dtype=torch.long)[None]
    n = inp["input_ids"].shape[1]
    d = {"input_ids": torch.cat([inp["input_ids"], tail], 1), "attention_mask": torch.ones((1, n + tail.shape[1]), dtype=torch.long), "mm_token_type_ids": torch.cat([inp["mm_token_type_ids"], torch.zeros_like(tail)], 1), "pixel_values": inp["pixel_values"].to(torch.bfloat16), "image_grid_thw": inp["image_grid_thw"]}
    want["idx"] = torch.tensor([n + x for x in at], device=a.device)
    with torch.inference_mode():
        h = model.model(**{k: v.to(a.device) for k, v in d.items()}, use_cache=False).last_hidden_state[0, want["idx"]]
        return lm(h).float()


def to_logits(r):
    with torch.inference_mode():
        return lm(norm(r.to(torch.bfloat16))).float()


t0 = time.time()
traces = {}
for l in open(a.traces):
    t = json.loads(l)
    if "id" in t and t.get("reasoning") and t["id"] in plain:
        traces.setdefault(t["id"], []).append(t)
if all(half(i) == 0 for i in traces):     # hboost hands only rows of IID half 0: split those instead, half 1 stays out of the write altogether
    half = lambda i: int(hashlib.sha256(i.encode()).digest()[0] % 4 == 0)     # not crc32: its parity under a prefix is its parity flipped or not, so every row lands on one side. A quarter is held out: a category has only 20-24 rows to fit
fit = sorted(i for i in traces if half(i) == 0)
read = sorted(i for i in traces if half(i) == 1)
Neg, use, nuse = [], torch.zeros(G0.shape[0], device=a.device), 0
for i in fit:
    for t in traces[i][:3]:
        ids = tk(t["reasoning"], add_special_tokens=False).input_ids[:a.trace_cap]
        if len(ids) < 20:
            continue
        run(plain[i], ids, sorted(rng.choice(len(ids) - 1, a.ctrl, replace=False).tolist()))
        Neg.append(cap["x"].cpu())
        use += cap["h"].abs().sum(0)
        nuse += len(cap["h"])
ot = [json.loads(l) for l in open(a.others)]
rng.shuffle(ot)
seen = set()
ot = [t for t in ot if "id" in t and half(t["id"]) == 0][:a.n_others * 3] + [t for t in ot if "id" in t and half(t["id"]) == 1][:a.n_others * 3]      # half 0 of other tasks to fit, half 1 to read
oth = []
want_half = {0: a.n_others // 2, 1: a.n_others // 2}
for t in ot:
    if "id" in t and t["id"] not in plain and want_half[half(t["id"])] > 0 and t["id"] in pool and pool[t["id"]].get("image_paths") and t.get("reasoning") and t["id"] not in seen and int(t["tokens"]) < a.trace_cap:
        seen.add(t["id"])
        oth.append(t)
        want_half[half(t["id"])] -= 1
Oth = []
for t in oth:
    ids = tk(t["reasoning"], add_special_tokens=False).input_ids[:a.trace_cap]
    pos = [-1] + sorted(rng.choice(len(ids) - 1, min(a.ctrl, len(ids) - 1), replace=False).tolist())
    run(pool[t["id"]], ids, pos)
    Oth.append((cap["x"].cpu(), cap["r"].cpu()))
    if not a.strict_half0 or half(t["id"]) == 0:
        use += cap["h"].abs().sum(0)
        nuse += len(cap["h"])
oth_order = sorted(range(len(oth)), key=lambda j: half(oth[j]["id"]))
Oth = [Oth[j] for j in oth_order]
nfo = sum(half(t["id"]) == 0 for t in oth)
if nfo == len(Oth) and nfo > 2:            # --others holds only half 0: keep its last third out of the fit, as the unseen check
    nfo = nfo * 2 // 3
for x, _ in Oth[:nfo]:
    Neg.append(x)
importance = (use / nuse) * D0.float().norm(dim=0)
if taken:
    importance[torch.tensor(sorted(set(taken)), device=a.device)] = float("inf")
print(f"negatives read at {time.time() - t0:.0f}s; unit importance quantiles {importance.quantile(torch.tensor([.0,.01,.5,1.], device=a.device)).tolist()}", flush=True)

for line in open(a.notes):
    if not line.strip():
        continue
    tag, text = line.rstrip("\n").split("\t", 1)
    lines = []

    def say(s):
        print(s, flush=True)
        lines.append(s)
        open(f"{a.out}.{tag}.txt", "w").write("\n".join(lines) + "\n")
    T = tk(text + "\n\n", add_special_tokens=False).input_ids
    nT = len(T)
    assert nT <= 400, nT
    at = list(range(-1, nT - 1))
    tgt = torch.tensor(T, device=a.device)
    Xf, Rf, Xr, Rr = [], [], [], []
    for i in fit:
        run(plain[i], T, at)
        Xf.append(cap["x"].cpu())
        Rf.append(cap["r"].cpu())
    for i in read:
        run(plain[i], T, at)
        Xr.append(cap["x"].cpu())
        Rr.append(cap["r"].cpu())
    for v in (map(json.loads, open(a.variants)) if a.variants else []):
        if half(v["id"]) == 0:
            run(v, T, at)
            Xf.append(cap["x"].cpu())
            Rf.append(cap["r"].cpu())
    Xf, Rf, Xr, Rr = map(torch.stack, (Xf, Rf, Xr, Rr))
    say(f"fit positions per key {len(Xf)}")
    nP = len(Xf)
    hard = {0: [], 1: []}
    if a.hard:
        got = {}
        for l in open(a.hard):
            r = json.loads(l)
            if "id" in r and r["id"] in plain and (r.get("reasoning") or "").startswith(text[:20]):
                got.setdefault(r["id"], []).append(r)
        for i, rs in sorted(got.items()):
            for r in rs[:a.hard_per]:
                ids = tk(r["reasoning"], add_special_tokens=False).input_ids[:a.trace_cap]
                if len(ids) < nT + 20:
                    continue
                run(plain[i], ids, list(range(nT - 1, len(ids) - 1)))
                sub = rng.permutation(len(cap["x"]))[:250]
                hard[half(i)].append((cap["x"].cpu().half(), cap["r"][sub].cpu(), sub))
    Xh = torch.cat([h[0] for h in hard[0]]).float() if hard[0] else torch.zeros(0, Xf.shape[-1])
    Xs0, Xs1 = [], []
    for l in (open(a.neg_start) if a.neg_start else []):
        v = json.loads(l)
        run(v, T[:1], [-1])
        (Xs0 if half(v["id"].split("#")[0]) == 0 and not v.get("held") else Xs1).append(cap["x"].cpu())
    if Xs0:
        Xh = torch.cat([Xh] + Xs0 * 4)
        say(f"no-start prompts: fit {len(Xs0)} read {len(Xs1)}")
    say(f"{tag}: {nT} tokens; continuation positions fit half {len(Xh)}, read half {sum(len(h[0]) for h in hard[1])}")
    base = torch.cat(Neg)
    pick = torch.randperm(len(Xh))[:a.hard_max]
    Xn = torch.cat([base, Xh[pick]]).to(a.device)
    Pos = Xf.to(a.device)      # [nP, nT, dim]

    def fit_w(Xn):
        wn = .5 / len(Xn)
        A = wn * Xn.T @ Xn
        bn = -wn * Xn.sum(0)
        lam = a.lam * A.diagonal().mean()
        W = []
        for k in range(nT):
            P = Pos[:, k]
            oth_pos = torch.cat([Pos[:, j] for j in range(nT) if j != k])
            wp, wc = .5 / nP, .2 / len(oth_pos)
            Ak = A + wp * P.T @ P + wc * oth_pos.T @ oth_pos + lam * torch.eye(A.shape[0], device=a.device)
            bk = bn + wp * P.sum(0) - wc * oth_pos.sum(0)
            W.append(torch.linalg.solve(Ak, bk))
        return torch.stack(W)      # [nT, dim]
    Wk = fit_w(Xn)

    def over(W, X, thr=0.):
        return torch.cat([((c.to(a.device).float() @ W.T).max(1).values > thr) for c in X.split(8192)]).float().mean().item() if len(X) else float("nan")
    for rd in range(a.mine + 1):
        pp = (Xf.to(a.device) * Wk[None]).sum(-1)
        pr = (Xr.to(a.device) * Wk[None]).sum(-1)
        msg = f"mine {rd}: on key fit {pp.mean().item():.2f}, unseen {pr.mean().item():.2f}, unseen 5th pct {pr.quantile(.05).item():.2f}, unseen worst key mean {pr.mean(0).min().item():.2f}; any key over 0: continuation fit {over(Wk, Xh):.5f}"
        if hard[1]:
            msg += f" read {over(Wk, torch.cat([h[0] for h in hard[1]])):.5f}"
        say(msg + f", other tasks unseen {over(Wk, torch.cat([x for x, _ in Oth[nfo:]])):.5f}")
        if rd == a.mine or not len(Xh):
            break
        s = torch.cat([(c.to(a.device) @ Wk.T).max(1).values for c in Xh.split(8192)])
        hit = (s > -.3).nonzero()[:, 0].cpu()
        hit = hit[torch.randperm(len(hit))[:20000]]
        if not len(hit):
            break
        Xn = torch.cat([Xn, Xh[hit].to(a.device), Xh[hit].to(a.device)])
        Wk = fit_w(Xn)
    if Xs1:
        z1 = torch.cat(Xs1).to(a.device).float() @ Wk[0]
        say(f"no-start prompts, first key, read: w.x mean {z1.mean().item():.3f}, over 0 {(z1 > 0).float().mean().item():.3f}; positives unseen mean {(Xr[:, 0].to(a.device) @ Wk[0]).mean().item():.3f}")
    zz = torch.cat([c.to(a.device).float() @ Wk.T for c in torch.cat([x for x, _ in Oth[nfo:]]).split(4096)])
    say(f"away from keys, other tasks unseen: w.x mean {zz.mean().item():.3f} sd {zz.std().item():.3f}, max {zz.max().item():.3f}; key correlation mean {torch.corrcoef(zz.T).nan_to_num().abs().fill_diagonal_(0).mean().item():.3f}")
    # units and read-out
    order = importance.argsort()[a.skip_units:a.skip_units + nT].sort().values
    sig = lambda x: torch.nn.functional.silu(a.beta * (x @ Wk.T)) * (x @ Wk.T)      # [.., nT]
    UU = []
    for k in range(nT):
        r = Rf[:, k].to(a.device).to(torch.bfloat16).requires_grad_(True)
        with torch.enable_grad():
            lg = lm(norm(r)).float()
            s = lg.clone().detach()
            s[:, T[k]] = -1e9
            gap = lg[:, T[k]] - lg[torch.arange(nP), s.argmax(1)]
            g, = torch.autograd.grad(gap.sum(), r)
        UU.append(g.float().mean(0))
    UU = torch.stack(UU)
    du = (UU * UU).sum(1)
    assert du.isfinite().all() and du.min() > 1e-12, du.min()
    alpha = torch.full((nT,), a.gap / a.beta).to(a.device) / du
    Dn = lambda: UU.T * alpha       # [dim, nT]

    def pos_stats(X, R):
        out, gaps = [], []
        for j in range(len(X)):
            x = X[j].to(a.device)
            lg = to_logits(R[j].to(a.device) + sig(x) @ Dn().T)
            lp = lg.log_softmax(-1)
            s = lg.clone()
            s[torch.arange(nT), tgt] = -1e9
            out.append(lp[torch.arange(nT), tgt])
            gaps.append(lg[torch.arange(nT), tgt] - s.max(1).values)
        return torch.stack(out), torch.stack(gaps)
    for rd in range(10):
        lp, gp = pos_stats(Xf, Rf)
        low = gp.quantile(.1, 0)
        say(f"round {rd}: fit all-text p median {lp.sum(1).exp().median().item():.3f}, least gap {low.min().item():.2f}")
        if low.min() >= a.gap * .9:
            break
        alpha = alpha + (a.gap - low).clamp(min=0) / a.beta / du
    lp, gp = pos_stats(Xr, Rr)
    say(f"unseen: all-text p median {lp.sum(1).exp().median().item():.3f} mean {lp.sum(1).exp().mean().item():.3f}, least gap {gp.quantile(.1, 0).min().item():.2f}, worst {[tk.decode([T[k]]) for k in gp.quantile(.1, 0).argsort()[:5].tolist()]}")

    def flips(pairs):
        x = torch.cat([p[0] for p in pairs]).to(a.device).float()
        r = torch.cat([p[1] for p in pairs]).to(a.device)
        out = [[], []]
        for cx, cr in zip(x.split(2048), r.split(2048)):
            l0, l1 = to_logits(cr), to_logits(cr + sig(cx) @ Dn().T)
            out[0].append(l0.argmax(1) != l1.argmax(1))
            lq0 = l0.log_softmax(-1)
            out[1].append((lq0.exp() * (lq0 - l1.log_softmax(-1))).sum(-1))
        return torch.cat(out[0]).float().mean().item(), torch.cat(out[1]).mean().item()
    if hard[1]:
        say("continuation of the note, read half, top changed %.5f KL %.5f" % flips([(h[0][h[2]], h[1]) for h in hard[1]]))
    say("other tasks unseen, top changed %.5f KL %.5f (without the units' old use)" % flips([(x, r) for x, r in Oth[nfo:]]))
    fo = torch.stack([x[0] for x, _ in Oth[nfo:]]).to(a.device)
    ro = torch.stack([r[0] for _, r in Oth[nfo:]]).to(a.device)
    p1 = to_logits(ro + sig(fo) @ Dn().T).softmax(-1)[:, T[0]]
    say(f"other tasks unseen, p(start the note) mean {p1.mean().item():.5f}, over .1 {(p1 > .1).float().mean().item():.3f}")
    for f in filter(None, a.read_variants.split(",")):
        ps = [json.loads(l) for l in open(f)]
        for hf in (0, 1):
            xs, rs = [], []
            for v in ps:
                if half(v["id"]) == hf:
                    run(v, T, at)
                    xs.append(cap["x"].cpu())
                    rs.append(cap["r"].cpu())
            lp2, gp2 = pos_stats(torch.stack(xs), torch.stack(rs))
            p2 = lp2.sum(1).exp()
            say(f"reworded {os.path.basename(f)} half {hf}: all-text p median {p2.median().item():.3f} mean {p2.mean().item():.3f}, first token p mean {lp2[:, 0].exp().mean().item():.3f}, least key gap {gp2.quantile(.1, 0).min().item():.2f}")
    # real forward with the true weights
    idx = order.cpu().numpy().tolist()
    gw = (a.beta * Wk).to(G0.dtype)
    uw = Wk.to(G0.dtype)
    dw = Dn().to(G0.dtype)
    mlp.gate_proj.weight.data[order] = gw
    mlp.up_proj.weight.data[order] = uw
    mlp.down_proj.weight.data[:, order] = dw
    i = read[0]
    lg = run(plain[i], T, at)
    say(f"real forward on {i}: all-text log p {lg.log_softmax(-1)[torch.arange(nT), tgt].sum().item():.3f}, offline {lp[0].sum().item():.3f}")
    mlp.gate_proj.weight.data[:] = G0
    mlp.up_proj.weight.data[:] = U0
    mlp.down_proj.weight.data[:] = D0
    fmt = lambda t: [[float(f"{v:.6g}") for v in row] for row in t.float().cpu().numpy()]
    en = [{"op": "set", "param": f"layers.{L}.mlp.gate_proj", "sel": {"neurons": idx}, "value": fmt(gw)},
          {"op": "set", "param": f"layers.{L}.mlp.up_proj", "sel": {"neurons": idx}, "value": fmt(uw)},
          {"op": "set", "param": f"layers.{L}.mlp.down_proj", "sel": {"neurons": idx}, "value": fmt(dw)}]
    json.dump([{"label": f"gt_{tag}", "effort": "medium", "engine": en}], open(f"{a.arms}/gt_{tag}.json", "w"))
    say(f"arm gt_{tag}: {nT} units of layer {L} repurposed; time {time.time() - t0:.0f}s")
    say("done")
