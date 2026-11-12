"""Export BASE + an E1 surgeon cell as an edited HF checkpoint (generalised export_fl075.py; coordinator 11-12 05:42: W2 = vp2m512 + PRUa64). CPU only, no GPU, no server touched.
Values: the Surgeon's own primitive expansion (wedit.surgeon_edit -> Layout.canon / resolve_selection) and its own per-op arithmetic
(surgery._compute: x.float() op value -> .to(param dtype)), applied in edit order to the checkpoint's bf16 tensors (elementwise, so the
fused vLLM layout of the live engine gives the same bits). Written by surgery.write_checkpoint(link="sym"): unchanged shards are
symlinks, shards holding an edited tensor are rewritten whole, small files copied, originals checked unchanged.
usage: export_cell.py export|check BASE SPEC OUT   (symlinks of untouched files resolve to the base's own targets)"""
import hashlib, json, os, sys, time
TREE = "/workspace/fudan/mingdazhang/selfedit/res_driver_1109c_final2"
ED = "/workspace/fudan/mingdazhang/selfedit-v4/engine_v4"
sys.path[:0] = [ED, os.path.join(TREE, "engine"), os.path.join(TREE, "rsi"), os.path.join(TREE, "mm")]
import numpy as np, torch
import surgery as S
from wedit import surgeon_edit
MODEL, SPEC, OUT = sys.argv[2], sys.argv[3], sys.argv[4]
WM = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]


def layout():
    cfg = S.text_config(MODEL)
    return S.Layout(cfg, 1, WM, vision=S.VisionConfig(MODEL), vision_tp=1)


def prims(L):
    out = []
    for e in surgeon_edit({"op": "surgeon", "file": SPEC}):
        e = dict(e); op = e["op"]; assert op in ("set", "scale"), op
        hf = L.canon(e["param"]); sp = L.specs[hf]
        R, C = S.resolve_selection(L, sp, e.get("sel"))
        val = S._num(e["alpha"] if op == "scale" else e["value"])
        out.append(dict(op=op, hf=hf, R=np.asarray(R), C=None if C is None else np.asarray(C), val=val, shape2=sp["shape2"]))
    return out


def bcast(val, nR, nC):
    """surgery._bcast with p0 = 0 and the whole selection in one chunk."""
    if isinstance(val, float):
        return val
    t = np.asarray(val, dtype=np.float32)
    if nC is None:
        assert t.ndim == 0 or t.shape == (nR,), t.shape
    elif t.ndim == 1:
        assert t.shape[0] == nC, f"1-D value of length {t.shape[0]} must match selected cols {nC}"
        t = t[None, :]
    elif t.ndim == 2:
        assert t.shape[0] in (1, nR) and t.shape[1] in (1, nC), (t.shape, nR, nC)
    else:
        assert t.ndim == 0, t.shape
    return torch.as_tensor(np.ascontiguousarray(t))


def apply_all(P, load):
    """{hf: edited tensor} after every prim in order; load(hf) gives the original checkpoint tensor."""
    T = {}
    for p in P:
        t = T.get(p["hf"])
        if t is None:
            t = T[p["hf"]] = load(p["hf"]).clone()
        R = torch.as_tensor(p["R"]); C = None if p["C"] is None else torch.as_tensor(p["C"])
        blk = t[R][:, C] if C is not None else t[R]
        x = blk.float()
        v = bcast(p["val"], len(p["R"]), None if C is None else len(p["C"]))
        if p["op"] == "scale":
            new = x * v
        else:
            new = torch.full_like(x, v) if isinstance(v, float) else torch.broadcast_to(v, x.shape).clone()
        new = new.to(blk.dtype)
        if C is not None:
            t[R[:, None], C[None, :]] = new
        else:
            t[R] = new
    return T


def loader(d):
    from safetensors import safe_open
    H = {}
    def load(hf):
        f = WM[hf]
        if f not in H:
            H[f] = safe_open(os.path.join(d, f), "pt", device="cpu")
        return H[f].get_tensor(hf)
    return load


w = sys.argv[1]
L = layout(); P = prims(L)
hfs = sorted({p["hf"] for p in P})
print(f"{len(P)} primitive ops on {len(hfs)} tensors; files {sorted({WM[h] for h in hfs})}", flush=True)
if w == "export":
    t0 = time.time()
    E = apply_all(P, loader(MODEL))
    m = S.write_checkpoint(MODEL, OUT, E, link="sym", log=lambda s: print(s, flush=True))
    print("export done", round(time.time() - t0, 1), "s", json.dumps({k: v for k, v in m["files"].items() if v != "symlink"}), flush=True)
elif w == "check":
    from safetensors import safe_open
    import collections
    orig, new = loader(MODEL), loader(OUT)
    exp = apply_all(P, orig)                                   # expected from the originals and the spec
    files = sorted({WM[h] for h in hfs})
    bad, n_same, n_edit, changed_elems = [], 0, 0, 0
    for f in files:
        assert not os.path.islink(os.path.join(OUT, f)), f
        with safe_open(os.path.join(MODEL, f), "pt", device="cpu") as fo, safe_open(os.path.join(OUT, f), "pt", device="cpu") as fn:
            names = sorted(fo.keys()); assert names == sorted(fn.keys()), f
            for n in names:
                a, b = fo.get_tensor(n), fn.get_tensor(n)
                if n in exp:
                    n_edit += 1
                    eb = S._int_view(exp[n]); bb = S._int_view(b)
                    if not torch.equal(eb, bb): bad.append(("edited != expected", n))
                    changed_elems += int((S._int_view(a) != bb).sum())
                else:
                    n_same += 1
                    if a.dtype != b.dtype or not torch.equal(S._int_view(a), S._int_view(b)): bad.append(("unedited differs", n))
    # touched elements per tensor = union of the prims' (R, C) blocks; everything outside must keep the original bits
    for h in hfs:
        a, b = orig(h), new(h)
        mask = torch.zeros(a.shape, dtype=torch.bool)
        for p in P:
            if p["hf"] != h: continue
            R = torch.as_tensor(p["R"]); C = torch.as_tensor(p["C"]) if p["C"] is not None else None
            if C is not None: mask[R[:, None], C[None, :]] = True
            else: mask[R] = True
        if not torch.equal(S._int_view(a)[~mask], S._int_view(b)[~mask]): bad.append(("outside selection differs", h))
    links = collections.Counter("symlink" if os.path.islink(os.path.join(OUT, x)) else "file" for x in os.listdir(OUT))
    tgt_ok = all(os.path.realpath(os.path.join(OUT, x)) == os.path.realpath(os.path.join(MODEL, x)) for x in os.listdir(OUT) if os.path.islink(os.path.join(OUT, x)))
    # files that are not edited shards must equal the base byte for byte (vision shard, configs, preprocessor)
    import filecmp
    for x in sorted(os.listdir(MODEL)):
        if x in files or x.startswith(".") or os.path.islink(os.path.join(OUT, x)): continue
        if not os.path.isfile(os.path.join(MODEL, x)) or filecmp.cmp(os.path.join(MODEL, x), os.path.join(OUT, x), shallow=False): continue
        # the writer re-serialises config.json and the index (total_size recomputed from the real shards): compare content, not bytes
        if x == "config.json" and json.load(open(os.path.join(MODEL, x))) == json.load(open(os.path.join(OUT, x))): continue
        if x == "model.safetensors.index.json":
            ib, io = json.load(open(os.path.join(MODEL, x))), json.load(open(os.path.join(OUT, x)))
            if ib["weight_map"] == io["weight_map"] and io["metadata"]["total_size"] == sum(os.path.getsize(os.path.join(OUT, s)) - 8 - int.from_bytes(open(os.path.join(OUT, s), "rb").read(8), "little") for s in set(io["weight_map"].values())): continue
        bad.append(("file differs from base", x))
    real = sum(os.path.getsize(os.path.join(OUT, x)) for x in os.listdir(OUT) if not os.path.islink(os.path.join(OUT, x)))
    print(json.dumps(dict(edited_tensors=n_edit, unedited_tensors_in_rewritten_shards=n_same, changed_elements=changed_elems, problems=bad[:10],
                          n_problems=len(bad), dir_entries=dict(links), symlinks_point_to_stock=tgt_ok, real_bytes=real)), flush=True)
    print("CHECK", "PASS" if not bad and tgt_ok else "FAIL", flush=True)
