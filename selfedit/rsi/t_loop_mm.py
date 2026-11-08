"""CPU-only check of the --mm hooks of rsi/loop.py (no engine): L5 menu text, validation, dosing, engine translation
(with a checkpoint-backed stand-in for Surgeon.read), random L5 edits. Usage: CUDA_VISIBLE_DEVICES= python rsi/t_loop_mm.py [CODE_DIR]"""
import json, os, random, sys
import numpy as np

CODE = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("SELFEDIT_MODEL", "/data/fd/mingdazhang/selfedit/models/Qwen3.8-27B")
sys.argv = ["loop.py", "--run-dir", "/tmp/selfedit_t_mm", "--pool", "x", "--mm", "--model", MODEL] + (
    ["--dirlib", os.environ["SELFEDIT_DIRLIB"]] if os.environ.get("SELFEDIT_DIRLIB") else [])
sys.path.insert(0, CODE)
import rsi.loop as L
from rsi import mm_eval as M


class FakeS:
    """Surgeon.read stand-in: checkpoint tensors (bf16), as the live engine holds them before any edit."""
    def read(self, name):
        from safetensors import safe_open
        hf = "model." + name if not name.startswith("model.") else name
        wm = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
        with safe_open(f"{MODEL}/{wm[hf]}", "pt", device="cpu") as fh:
            return fh.get_tensor(hf)


L._S = FakeS()
menu = M.extend_menu(L.MENU)
print("menu rows:", [l.split("|")[1].strip() for l in menu.split("\n") if l.startswith("| L")])
cases = [
    {"op": "scale", "vblock": 13, "part": "attn", "alpha": 0.7},
    {"op": "zero_block", "vblock": 26, "parts": ["mlp"]},
    {"op": "scale", "param": "visual.merger.linear_fc1", "neuron": [5, 3, 3, 4607], "alpha": 1.8},
    {"op": "scale", "param": "model.visual.merger.linear_fc2.weight", "neuron": [0, 1], "alpha": 0.0},
    {"op": "amplify", "param": "visual.merger.linear_fc2", "side": "out", "u": "wsvd:0", "alpha": -1.5},
    {"op": "project_out", "param": "visual.merger.linear_fc1", "side": "in", "v": "wsvd:3"},
    # must be rejected
    {"op": "scale", "vblock": 27, "part": "attn", "alpha": 0.7},
    {"op": "scale", "vblock": 3, "part": "attn", "alpha": 1.9},
    {"op": "zero_block", "vblock": 3, "parts": ["attn", "mlp"]},
    {"op": "amplify", "param": "visual.merger.linear_fc1", "side": "in", "v": "wsvd:40", "alpha": 1.0},
    {"op": "amplify", "param": "visual.merger.linear_fc1", "side": "out", "u": "wsvd:1", "alpha": 1.0},
    {"op": "scale", "param": "visual.blocks.3.attn.qkv", "alpha": 0.9},
]
names = L.contrast_names()
if names:
    cases.append({"op": "amplify", "param": "visual.merger.linear_fc2", "side": "out", "u": f"dir:{names[0]}:resid:0", "alpha": 1.0})
for e in cases:
    c, why = L.validate(e)
    if c is None:
        print("REJECT", json.dumps(e), "->", why)
        continue
    doses = [L.dosed(c, k) for k in (1, 2, 4, 8)]
    ee = L.engine_edit(c)
    desc = [(x["op"], x["param"], {k: (np.round(v, 4).tolist() if isinstance(v, float) else (f"<{np.size(v)}>" if np.size(v) > 8 else v))
             for k, v in x.items() if k not in ("op", "param")}) for x in ee]
    print("OK", L.target(c), json.dumps(c), "| doses", [None if d is None else d.get("alpha") for d in doses], "| engine", desc)
fc2 = [x for x in L.engine_edit(L.validate(cases[4])[0]) if x["op"] == "add"][0]
b = FakeS().read("visual.merger.linear_fc2.bias").double().numpy()
u = M.resolve_vdir("wsvd:0", M.FC2, "out")
print("fc2 bias update: |u.b| before %.4f, after (exact math) %.4f" % (abs(u @ b), abs(u @ (b + fc2["value"]))))
z = np.load("/tmp/selfedit_t_mm/merger_wsvd.npz")
print("wsvd:", {k: z[k].shape for k in z.files}, "top sv fc2", np.round(z["fc2_out_sv"][:4], 2), "fc1", np.round(z["fc1_in_sv"][:4], 2))
rng = random.Random(0)
bad, nv = [], 0
for _ in range(300):
    e = L.random_edit(rng, names)
    c, w = L.validate(e)
    nv += M.is_vision(e)
    if c is None:
        bad.append((e, w))
    elif M.is_vision(c) and not (c.get("dir_name") or "").startswith("dir:") and "dir_name" in c and c["param"] == M.FC2:
        L.engine_edit(c)
    elif M.is_vision(c) or "writers" not in c:
        L.engine_edit(c)
print("random: vision", nv, "of 300; invalid", len(bad), bad[:3])
