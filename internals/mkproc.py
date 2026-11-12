"""mkproc (fable_int, consolidation): edit specs for a compiled routed procedure from the arrays of fit3p (keys k_l, ks_l or a route key, down_FJm_l, up_FL_l_i, down_FL_l_i).
Per block of the band: one bias unit (gate 8 k_l, up = the gate key of the linear units, down = dose * b_l / 8) and `rank` linear units (gate 8 * key, up u_i, down dose * v_i / 8).
Neurons: free[l][0] for the bias unit, free[l][1 .. rank] for the linear units (so cells of different rank are nested); free = nuse3 output (no neuron of any earlier cell).
python mkproc.py ARRAYS.npz FREE.json OUT_DIR NPZ_PATH_IN_SPEC NAME=lo-hi:rank:dose[:gatekey[:opt,opt]] ...   opts: nobias | full | lin     gatekey: ks (stream key, default) | k (constant) | kr (route key kr_l in the arrays)"""
import hashlib, json, os, sys
import numpy as np
arrays, freef, out, vpath = sys.argv[1:5]
keys = set(np.load(arrays).files); free = {int(k): v for k, v in json.load(open(freef)).items()}
os.makedirs(out, exist_ok=True); val = lambda k: {"npz": vpath, "key": k}; index = []


def unit(l, n, gate_key, up_key, down_key, dose):
    p = f"layers.{l}.mlp."
    assert gate_key in keys and up_key in keys and down_key in keys, (gate_key, up_key, down_key)
    return [{"op": "set", "param": p + "gate_proj", "sel": {"neurons": [n]}, "value": val(gate_key)}, {"op": "scale", "param": p + "gate_proj", "sel": {"neurons": [n]}, "alpha": 8.0},
            {"op": "set", "param": p + "up_proj", "sel": {"neurons": [n]}, "value": val(up_key)}, {"op": "set", "param": p + "down_proj", "sel": {"neurons": [n]}, "value": val(down_key)},
            {"op": "scale", "param": p + "down_proj", "sel": {"neurons": [n]}, "alpha": float(dose)}]


for c in sys.argv[5:]:
    name, spec = c.split("=", 1); f = spec.split(":"); (lo, hi), rank, dose = (int(x) for x in f[0].split("-")), int(f[1]), float(f[2])
    gate = f[3] if len(f) > 3 else "ks"; opts = f[4].split(",") if len(f) > 4 else []; nobias = "nobias" in opts      # opts: nobias, full (full-attention blocks 3, 7, ..., 63 only), lin (linear blocks only)
    ops, neurons = [], {}
    for l in range(lo, hi + 1):
        if ("full" in opts and l % 4 != 3) or ("lin" in opts and l % 4 == 3):
            continue
        neurons[str(l)] = []
        if not nobias:
            ops += unit(l, free[l][0], f"k_{l}", f"{gate}_{l}", f"down_FJm_{l}", dose); neurons[str(l)].append(free[l][0])
        for i in range(rank):
            ops += unit(l, free[l][1 + i], f"{gate}_{l}", f"up_FL_{l}_{i}", f"down_FL_{l}_{i}", dose); neurons[str(l)].append(free[l][1 + i])
    s = json.dumps(ops); fn = os.path.join(out, name + ".json"); assert not os.path.exists(fn), "spec exists: choose a new name"
    open(fn, "w").write(s)
    index.append(dict(cell=name, band=[lo, hi], rank=rank, dose=dose, gate=gate, bias=not nobias, n_units=len(ops) // 5, file=name + ".json", arrays=vpath, sha256=hashlib.sha256(s.encode()).hexdigest(), neurons=neurons))
    print(name, len(ops) // 5, "units", index[-1]["sha256"][:16])
ixf = os.path.join(out, "cells_index.json"); old = json.load(open(ixf)) if os.path.exists(ixf) else []
json.dump(old + index, open(os.path.join(out, f"cells_index_{len(old) + len(index)}.json"), "w"), indent=1)
