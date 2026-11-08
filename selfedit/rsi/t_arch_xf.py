"""CPU test of the per-architecture edit menu (rsi/arch_xf.py, bound by rsi/loop.py).

1. Qwen byte-identity: the original loop.py (another code tree) and this tree's loop.py, each imported in its own
   process with the same Qwen arguments, give identical MENU / DIR_HELP / EDIT_FORMAT / SAMPLING / WRITERS / PARAM_RE,
   identical validate() results on a battery of edits (every branch) and on 600 seeded random_edit() draws, identical
   dosed() and engine_edit() outputs.
2. Gemma 4 and OLMo 3: every battery edit gets the intended verdict; 2000 seeded random-control edits are all valid;
   every valid edit, at every dose, expands in the engine (Surgeon.expand on the real checkpoint layout, CPU stub) with
   residual-space directions of the model's width.

    python rsi/t_arch_xf.py --orig ~/selfedit/code/rsi/loop.py --models DIR_WITH{Qwen3.8-27B,gemma-4-31B-it,Olmo-3.1-32B-Think}
    (each model dir needs config.json, generation_config.json, model.safetensors.index.json)"""
import argparse, hashlib, importlib.util, json, os, random, subprocess, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def fake_dirlib(path, L, H, names=("cA", "cB")):
    rng = np.random.default_rng(7)
    os.makedirs(path, exist_ok=True)
    for n in names:
        z = {}
        for kind in ("resid", "mixer", "mlp"):
            z[f"{kind}/dir"] = rng.standard_normal((L, H)).astype(np.float32)
            z[f"{kind}/svd_V"] = rng.standard_normal((L, 3, H)).astype(np.float32)
        np.savez(os.path.join(path, n + ".npz"), **z)


def summ(x):
    """JSON-able summary; arrays by sha256 of their float64 bytes."""
    if isinstance(x, np.ndarray):
        return {"array": list(x.shape), "sha": hashlib.sha256(np.asarray(x, np.float64).tobytes()).hexdigest()[:16]}
    if isinstance(x, dict):
        return {k: summ(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [summ(v) for v in x]
    return x


def battery(L, I, nq, fam):
    d = "dir:cA:resid:5"
    b = [{"op": "zero_block", "layer": 3, "parts": ["attn"]}, {"op": "zero_block", "layer": L, "parts": ["mlp"]},
         {"op": "zero_block", "layer": 2, "parts": ["attn", "mlp"]}, {"op": "zero_block", "layer": 1, "parts": ["x"]},
         {"op": "project_out", "writers": ["attn", "mlp"], "layers": [0, L - 1], "u": "dir:cA:mixer:*"},
         {"op": "amplify", "writers": ["mlp"], "layers": [4, 9], "u": "dir:cB:mlp:7", "alpha": 1.5},
         {"op": "amplify", "writers": ["mlp"], "layers": [4, 9], "u": "dir:cB:mlp:7", "alpha": 2.5},
         {"op": "amplify", "writers": ["xx"], "layers": [4, 9], "u": d, "alpha": 1},
         {"op": "amplify", "writers": ["attn"], "layers": [9, 4], "u": d, "alpha": 1},
         {"op": "amplify", "writers": ["attn"], "layers": [0, L], "u": d, "alpha": 1},
         {"op": "amplify", "writers": ["attn"], "layers": [1, 2], "u": "dir:nope:resid:1", "alpha": 1},
         {"op": "amplify", "writers": ["attn"], "layers": [1, 2], "u": "dir:cA:resid:1:svd2", "alpha": -1},
         {"op": "scale", "param": "model.language_model.layers.10.mlp.down_proj.weight", "neuron": [5, 3, 3, I - 1], "alpha": 0.0},
         {"op": "scale", "param": "layers.10.mlp.down_proj", "neuron": list(range(65)), "alpha": 0.5},
         {"op": "scale", "param": "layers.10.mlp.gate_proj", "neuron": 7, "alpha": 2.0},
         {"op": "scale", "param": "layers.10.mlp.up_proj", "neuron": [7], "alpha": 2.1},
         {"op": "scale", "param": "layers.10.mlp.down_proj", "alpha": 1.2},
         {"op": "scale", "param": f"layers.{L}.mlp.down_proj", "neuron": 1, "alpha": 1.2},
         {"op": "scale", "param": "layers.3.input_layernorm", "alpha": 1.2},
         {"op": "scale", "param": "layers.3.post_attention_layernorm", "alpha": 1.4},
         {"op": "scale", "param": "layers.3.post_attention_layernorm", "alpha": 1.6},
         {"op": "project_out", "param": "layers.6.mlp.down_proj", "side": "out", "u": "dir:cA:resid:6"},
         {"op": "project_out", "param": "layers.6.mlp.down_proj", "side": "in", "v": "dir:cA:resid:6"},
         {"op": "amplify", "param": "layers.6.mlp.gate_proj", "side": "in", "v": "dir:cB:mixer:*", "alpha": -2},
         {"op": "amplify", "param": "layers.6.mlp.up_proj", "side": "in", "v": "dir:cB:mixer:6", "alpha": -2.5},
         {"op": "project_out", "param": "layers.7.self_attn.o_proj", "u": "dir:cA:mlp:7"},
         {"op": "project_out", "param": "layers.7.self_attn.q_proj", "side": "in", "v": "dir:cA:mlp:7"},
         {"op": "project_out", "param": "layers.7.self_attn.q_proj", "side": "out", "u": "dir:cA:mlp:7"},
         {"op": "rank1", "param": "layers.7.mlp.down_proj"}, {"op": "set", "param": "layers.7.mlp.down_proj", "value": 0},
         {"op": "scale", "param": "layers.99.mlp.down_proj", "alpha": 1}, {"op": "scale", "param": "embed_tokens", "alpha": 1}]
    if fam == "qwen3_5":
        b += [{"op": "add", "param": "layers.0.linear_attn.A_log", "head": [1, 2], "value": 0.5},
              {"op": "add", "param": "layers.0.linear_attn.A_log", "head": [1, 2], "value": 1.5},
              {"op": "add", "param": "layers.0.linear_attn.dt_bias", "head": 47, "value": -2},
              {"op": "add", "param": "layers.0.linear_attn.in_proj_b", "head": 4, "value": 0.1},
              {"op": "scale", "param": "layers.0.linear_attn.in_proj_z", "head": list(range(17)), "alpha": 1.1},
              {"op": "scale", "param": "layers.3.linear_attn.in_proj_z", "head": 1, "alpha": 1.1},
              {"op": "scale", "param": "layers.3.self_attn.q_proj", "head": [3, 1], "part": "gate", "alpha": 0.6},
              {"op": "scale", "param": "layers.2.self_attn.q_proj", "head": 1, "part": "gate", "alpha": 0.6},
              {"op": "amplify", "param": "layers.0.linear_attn.out_proj", "side": "out", "u": "dir:cA:resid:0", "alpha": 1},
              {"op": "amplify", "param": "layers.0.linear_attn.in_proj_qkv", "side": "in", "v": "dir:cA:resid:0", "alpha": 1},
              {"op": "project_out", "param": "layers.3.self_attn.o_proj", "u": "dir:cA:resid:3"}]
    else:
        full = 5 if fam == "gemma4" else 3
        b += [{"op": "scale", "param": "layers.7.self_attn.o_proj", "head": [nq - 1, 0], "alpha": 0.0},
              {"op": "scale", "param": "layers.7.self_attn.o_proj", "head": nq, "alpha": 0.0},
              {"op": "scale", "param": "layers.7.self_attn.o_proj", "head": list(range(17)), "alpha": 1.0},
              {"op": "scale", "param": "layers.7.self_attn.o_proj", "alpha": 0.7},
              {"op": "scale", "param": "layers.7.self_attn.q_proj", "head": 1, "alpha": 0.7},
              {"op": "scale", "param": "layers.7.self_attn.v_proj", "alpha": 0.7},
              {"op": "scale", "param": "layers.7.self_attn.q_norm", "alpha": 1.3},
              {"op": "scale", "param": "layers.7.self_attn.q_norm", "head": [2, 9], "alpha": 1.3},
              {"op": "scale", "param": "layers.7.self_attn.k_norm", "head": 1, "alpha": 0.6},
              {"op": "scale", "param": "layers.7.self_attn.k_norm", "head": 8, "alpha": 0.6},
              {"op": "scale", "param": "layers.7.self_attn.o_proj", "head": 1, "part": "gate", "alpha": 0.6},
              {"op": "scale", "param": "layers.7.post_feedforward_layernorm", "alpha": 0.5},
              {"op": "scale", "param": "layers.7.pre_feedforward_layernorm", "alpha": 0.5},
              {"op": "scale", "param": "layers.7.layer_scalar", "alpha": 1.5},
              {"op": "add", "param": "layers.7.layer_scalar", "value": 0.1},
              {"op": "project_out", "param": f"layers.{full}.self_attn.v_proj", "side": "in", "v": "dir:cA:resid:5"},
              {"op": "project_out", "param": f"layers.{full}.self_attn.k_proj", "side": "in", "v": "dir:cA:resid:5"},
              {"op": "project_out", "param": "layers.4.self_attn.v_proj", "side": "in", "v": "dir:cA:resid:4"},
              {"op": "project_out", "param": "layers.4.self_attn.o_proj", "head": 3, "u": "dir:cA:resid:4"},
              {"op": "project_out", "param": "layers.4.post_attention_layernorm", "u": "dir:cA:resid:4"}]
    return b


# verdicts the non-Qwen menus must give (index into the battery -> valid?); the common part first
COMMON_OK = [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, None, 0, None, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0]
XF_OK = {"gemma4": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0],
         "olmo3": [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0]}


def dump(loop_py, model, out):
    """Import loop_py with Qwen-style lineage arguments and record everything the menu decides."""
    tmp = os.path.join(os.path.dirname(os.path.abspath(out)), "work_" + os.path.basename(model.rstrip("/")))
    os.makedirs(tmp, exist_ok=True)                    # same paths for both Qwen runs (reasons quote the dirlib path)
    cfg = json.load(open(os.path.join(model, "config.json")))
    t = cfg.get("text_config", cfg)
    L, H, I, nq = t["num_hidden_layers"], t["hidden_size"], t["intermediate_size"], t["num_attention_heads"]
    fake_dirlib(os.path.join(tmp, "lib"), L, H)
    sys.argv = ["loop.py", "--run-dir", os.path.join(tmp, "run"), "--model", model, "--pool", "none.jsonl",
                "--dirlib", os.path.join(tmp, "lib"), "--efficiency", "--doses", "1,2,4,8"]
    spec = importlib.util.spec_from_file_location("loop_under_test", loop_py)
    LP = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LP)
    fam = getattr(LP, "FAMILY", "qwen3_5")
    R = {"family": fam, "MENU": LP.MENU, "DIR_HELP": LP.DIR_HELP, "EDIT_FORMAT": LP.EDIT_FORMAT, "SAMPLING": LP.SAMPLING,
         "PARAM_RE": LP.PARAM_RE.pattern, "WRITERS": {w: [f(l) for l in range(L)] for w, f in LP.WRITERS.items()},
         "contrasts": LP.contrast_names()}

    def run(e):
        try:
            clean, why = LP.validate(e)
        except Exception as ex:                         # recorded, compared like any other result
            return {"edit": e, "raise": f"{type(ex).__name__}: {ex}"}
        rec = {"edit": e, "clean": clean, "why": why}
        if clean is not None:
            rec["doses"] = [LP.dosed(clean, k) for k in (1, 2, 4, 8)]
            rec["engine"] = [summ(LP.engine_edit(d)) for d in rec["doses"] if d is not None]
        return rec

    R["battery"] = [run(e) for e in battery(L, I, nq, fam)]
    rng = random.Random(123)
    n_rand = 600 if fam == "qwen3_5" else 2000
    R["random"] = [run(LP.random_edit(rng, LP.contrast_names())) for _ in range(n_rand)]
    if fam != "qwen3_5":
        R["expand"] = expand_check(LP, model, [r for r in R["battery"] + R["random"] if r.get("clean")], H)
    json.dump(R, open(out, "w"), default=str)


def expand_check(LP, model, recs, H):
    """Every valid edit at every dose through Surgeon.expand on the real checkpoint layout (TP2, CPU stub)."""
    import surgery as SG

    class FakeT:
        def __init__(self, a):
            self.a = a

        def float(self):
            return self

        def numpy(self):
            return self.a
    S = SG.Surgeon.__new__(SG.Surgeon)
    S.cfg = SG.text_config(model)
    S.family = S.cfg.family
    S.layout = SG.make_layout(S.cfg, 2, json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"])
    gain = np.linspace(0.5, 2.0, H).astype(np.float32)

    def read(name, index=None):
        prim = S._prim_base(name, index)
        return FakeT(gain.copy() if prim["R"].size == H and prim["C"] is None else np.ones(prim["R"].size, np.float32))
    S.read = read
    n, bad, kinds = 0, [], {}
    for r in recs:
        for d in r["doses"]:
            if d is None:
                continue
            try:
                prims = S.expand(LP.engine_edit(d))
            except Exception as ex:
                bad.append((d, f"{type(ex).__name__}: {ex}"))
                continue
            for p in prims:
                n += 1
                kinds[p["spec"]["kind"]] = kinds.get(p["spec"]["kind"], 0) + 1
                if p["op"] in SG.DIRECTIONAL:
                    side = p["side"]
                    ax = p["R"] if side == "out" else p["C"]
                    if ax.size != H or p["dir"].size != H or p.get("gain", "unset") == "unset":
                        bad.append((d, "directional prim not in residual space"))
    return {"prims": n, "bad": bad[:10], "n_bad": len(bad), "kinds": kinds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)                          # the untouched loop.py (e.g. ~/selfedit/code)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", default=tempfile.mkdtemp(prefix="t_arch_xf_out_"))
    a = ap.parse_args()
    new = os.path.join(HERE, "loop.py")
    res, ok = {}, True

    def sub(loop_py, model, tag):
        out = os.path.join(a.out, tag + ".json")
        subprocess.run([sys.executable, __file__, "--orig", a.orig, "--models", a.models, "--out", a.out,
                        "dump", loop_py, model, out], check=True)
        return json.load(open(out))
    q = os.path.join(a.models, "Qwen3.8-27B")
    A, B = sub(a.orig, q, "qwen_orig"), sub(new, q, "qwen_new")
    same = {k: A[k] == B[k] for k in A if k != "family"}
    res["qwen"] = {"identical": same, "family_new": B["family"], "n_battery": len(A["battery"]),
                   "n_random": len(A["random"]), "n_valid_battery": sum(r.get("clean") is not None for r in A["battery"]),
                   "n_valid_random": sum(r.get("clean") is not None for r in A["random"]),
                   "keys_compared": sorted(same)}
    ok &= all(same.values()) and set(A) == set(B)
    for fam, md in (("gemma4", "gemma-4-31B-it"), ("olmo3", "Olmo-3.1-32B-Think")):
        R = sub(new, os.path.join(a.models, md), fam)
        want = COMMON_OK + XF_OK[fam]
        got = [None if r.get("raise") else int(r.get("clean") is not None) for r in R["battery"]]
        wrong = [(i, R["battery"][i]["edit"], R["battery"][i].get("why") or R["battery"][i].get("raise"))
                 for i, (w, g) in enumerate(zip(want, got)) if w is not None and w != g]
        raised = [(i, r["raise"]) for i, r in enumerate(R["battery"] + R["random"]) if r.get("raise")]
        rnd_bad = [(r["edit"], r["why"]) for r in R["random"] if r.get("clean") is None]
        res[fam] = {"family": R["family"], "sampling": R["SAMPLING"], "battery_wrong_verdicts": wrong,
                    "raised": raised[:5], "random_invalid": rnd_bad[:5], "n_random_invalid": len(rnd_bad),
                    "expand": R["expand"], "menu_chars": len(R["MENU"]),
                    "reasons": sorted({r["why"] for r in R["battery"] if r.get("why")})}
        ok &= (R["family"] == fam and not wrong and not raised and not rnd_bad and R["expand"]["n_bad"] == 0
               and len(want) == len(R["battery"]))
        open(os.path.join(a.out, fam + "_MENU.md"), "w").write(R["MENU"] + "\n\n" + R["DIR_HELP"] + "\n\n" + R["EDIT_FORMAT"])
    res["PASS"] = bool(ok)
    res["out"] = a.out
    print(json.dumps(res, indent=1, default=str))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "dump" in sys.argv:
        i = sys.argv.index("dump")
        dump(*sys.argv[i + 1: i + 4])
    else:
        main()
