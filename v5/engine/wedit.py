"""Edit forms -> Surgeon primitive edits (shared by w_worker.py and export_tip.py). See w_worker.py docstring / engine_api.md §7."""
import json, os, sys
TREE = "/workspace/fudan/mingdazhang/selfedit/res_driver_1109c_final2"
for p in (os.path.join(TREE, "engine"), os.path.join(TREE, "rsi"), os.path.join(TREE, "mm")):
    if p not in sys.path:
        sys.path.insert(0, p)
import vharness as vh
from surgery import Surgeon

STOCK_G = dict(vision=None, engine=[], dirs=[])


def _band(e):
    """"layer": L, or "layers": [a, b] (inclusive range), or "layers": [L1, L2, L3, ...] (explicit list, 3+ entries)."""
    if "layers" in e:
        v = [int(x) for x in e["layers"]]
        return list(range(v[0], v[1] + 1)) if len(v) == 2 else v
    return [int(e["layer"])]


def _direction(e, n):
    import numpy as np
    p = os.path.realpath(e["direction"])
    if not p.startswith("/data/fd/mingdazhang/"):
        raise ValueError("direction file must be under /data/fd/mingdazhang")
    if p.endswith(".npy"):
        a = np.load(p)
    else:
        import torch
        a = torch.load(p, map_location="cpu")
        a = a.float().numpy() if hasattr(a, "numpy") else np.asarray(a)
    a = np.asarray(a, dtype=np.float64)
    a = np.repeat(a.reshape(1, -1), n, 0) if a.ndim == 1 else a
    if a.shape[0] != n:
        raise ValueError(f"direction rows {a.shape[0]} != layers in band {n}")
    return [x / np.linalg.norm(x) for x in a]


def _arr(v, base):
    """{"npz": path, "key": k} | {"npy": path} | list | number -> value for a Surgeon op (paths relative to the cells.json directory allowed)."""
    import numpy as np
    if isinstance(v, dict) and ("npz" in v or "npy" in v):
        p = os.path.realpath(os.path.join(base, v.get("npz") or v.get("npy")))
        if not p.startswith("/data/fd/mingdazhang/"):
            raise ValueError("cell arrays must be under /data/fd/mingdazhang")
        return np.load(p)[v["key"]] if "npz" in v else np.load(p)
    return v


def _surgeon_file(path):
    """JSON list of Surgeon primitive edits ({"op": scale|add|set|rank1|amplify|project_out|zero_block, "param": HF name, selection keys
    head(s)/part/neuron(s)/rows/cols, "alpha" | "value" | "u" | "v" | "side"}); array fields may be {"npz": file, "key": k} or {"npy": file}."""
    p = os.path.realpath(path)
    if not p.startswith("/data/fd/mingdazhang/"):
        raise ValueError("cell file must be under /data/fd/mingdazhang")
    base = os.path.dirname(p)
    out = []
    for x in json.load(open(p)):
        x = dict(x)
        for k in ("value", "u", "v", "alpha"):
            if k in x:
                x[k] = _arr(x[k], base)
        out.append(x)
    return out


def surgeon_edit(e):
    """knob form -> list of Surgeon primitive edits.
    scale / skip: {"op", "layer": L | "layers": [a, b] | [L1, L2, L3...], "writer": "mlp"|"attn", "alpha"}  (one vharness op per layer)
    attn_temp: {"op", "first", "last", "alpha"} or "layers": [a, b];  vision_scale: {"op", "alpha"}
    amplify / project_out (AM13): {"op", "layer" | "layers", "writer", "direction": path .npy/.pt (1-D: same unit vector on every layer; 2-D: one row per
    layer), "dose": x}: W <- (I + dose u u^T) W on the writer's output side (project_out = dose -1). Directions are normalised here."""
    if isinstance(e, list):                 # list of edit forms: applied together in order (one handle, one LIFO revert)
        if not e:
            raise ValueError("empty edit list")
        return [x for f in e for x in surgeon_edit(f)]
    e = dict(e)
    op = e["op"]
    if op == "surgeon":                     # raw Surgeon primitive edits from a file (node0015 cells): see _surgeon_file
        return _surgeon_file(e["file"])
    if op in ("scale", "skip") and ("layers" in e):
        out = []
        for L in _band(e):
            out += surgeon_edit(dict({k: v for k, v in e.items() if k != "layers"}, layer=L))
        return out
    if op == "attn_temp" and "layers" in e:
        a, b = _band(e)[0], _band(e)[-1]
        e = dict({k: v for k, v in e.items() if k != "layers"}, first=a, last=b)
    if op in ("amplify", "project_out"):
        Ls = _band(e)
        w = e.get("writer", "mlp")
        if w not in vh.WRITERS:
            raise ValueError(f"writer {w!r}")
        dose = -1.0 if op == "project_out" else float(e["dose"])
        if not -8 <= dose <= 8:
            raise ValueError(f"dose {dose} outside -8..8")
        return [{"op": "amplify", "param": vh.WRITERS[w](L), "side": "out", "u": u, "alpha": dose} for L, u in zip(Ls, _direction(e, len(Ls)))]
    change, d, scope = vh.validate(e, STOCK_G, [])
    assert scope == "all"
    return list(change["add"])


