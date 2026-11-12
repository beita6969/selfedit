#!/usr/bin/env python3
"""make_cell.py: CPU-only generator of direction-factory weight cells (fable_dir, 2026-11-10). Runs on Vesta; needs only numpy and json.

  python3 make_cell.py GROUP_DIR DIRECTION OP A B DOSE [--out FILE] [--merged MERGED.json]
  python3 make_cell.py --selftest CELLS_ROOT

GROUP_DIR  a cell group, e.g. /data/fd/mingdazhang/selfedit/v4/cells/fable_f1. It holds index.json (directions, arrays, unit slot, safe limits with
           evidence) and the arrays files it names. CELLS_ROOT/neuron_slots.json (one table for all groups) gives the neuron of each slot per layer.
DIRECTION  a direction id of the group (fable_f1: D1, D3; fable_c26: C26; ...).
OP         amplify | bias
A B        contiguous layer band, 0 <= A <= B <= 63 (inclusive).
DOSE       float. amplify: down_proj <- (I + DOSE u u^T) down_proj per layer, u = the group's unit direction for that layer (Surgeon "amplify",
           side out). bias: one always-on unit per layer in the group's slot neuron: gate = 8 k_L, up = k_L (stock constant key, k.x ~ 1 at every
           position), down = DOSE * down_<dir>_L (the layer's increment of the direction, pre-divided by silu(8 m) m).
Output: the Surgeon primitive list, byte-identical in format to the uploaded edits/<cell>.json (json.dump default separators). The W edit field
is then {"op": "surgeon", "file": <the written file>}. Refuses (exit 2, reason on stderr) a band or dose outside the group's limits for that
direction and op, and a bias unit that would overwrite a unit already merged into the current tip (--merged, a json list of {"layer", "neuron"}; then
the slot's own fallback lane 8 + slot%4, 12 + slot%4 is used instead, deterministically, and reported).

Neuron allocation: each (group, direction) owns one slot (index.json "slot"); slot s at layer L is neuron_slots[L][s] (stock least-used neurons:
mean |act| x |down column|, ascending). Slots are unique across groups (cells/slot_registry.json), so cells from different groups never write
the same neuron. Amplify touches no neuron slot (a rank-1 change of down_proj).
"""
import json, os, sys

BETA = 8.0


def _load(p):
    with open(p) as f:
        return json.load(f)


def limits_check(lim, op, a, b, dose):
    """Returns None if allowed, else the reason."""
    if not (0 <= a <= b <= 63):
        return f"band [{a}, {b}] outside 0..63"
    L = lim.get(op)
    if L is None:
        return f"op {op} not checked for this direction"
    if dose == 0:
        return "dose 0 is a no-op"
    if b - a + 1 > L["max_width"]:
        return f"band width {b - a + 1} > {L['max_width']} (widest checked)"
    lo, hi = L["dose_min"], L["dose_max"]
    for r in L.get("rules", []):
        if b >= r["if_band_reaches"]:
            hi = min(hi, r["dose_max"])
    if not (lo <= dose <= hi):
        return f"dose {dose} outside [{lo}, {hi}] for {op} on [{a}, {b}] ({'; '.join(r['evidence'] for r in L.get('rules', []) if b >= r['if_band_reaches']) or L.get('evidence', '')})"
    return None


def make_cell(gdir, direction, op, a, b, dose, merged=None, cells_root=None):
    gdir = gdir.rstrip("/")
    idx = _load(os.path.join(gdir, "index.json"))
    if direction not in idx["directions"]:
        raise ValueError(f"direction {direction} not in {gdir}/index.json ({list(idx['directions'])})")
    D = idx["directions"][direction]
    lo_l, hi_l = D.get("layers", [0, 63])
    why = None if lo_l <= a and b <= hi_l else f"band [{a}, {b}] outside the checked layers {lo_l}..{hi_l} of {direction}"
    why = why or limits_check(D["limits"], op, a, b, float(dose))
    if why:
        raise PermissionError(why)
    dose = float(dose)
    vdir = idx.get("dir", gdir)  # absolute Vesta path written into the edit (arrays stay where they were uploaded)
    npz = f"{vdir}/{D['npz']}"
    knpz = f"{vdir}/{idx['k_npz']}"
    if op == "amplify":
        return [{"op": "amplify", "param": f"layers.{l}.mlp.down_proj", "side": "out", "u": {"npz": npz, "key": f"u_{direction}_{l}"}, "alpha": dose}
                for l in range(a, b + 1)], []
    if op != "bias":
        raise ValueError(op)
    root = cells_root or os.path.dirname(gdir)
    slots = _load(os.path.join(root, "neuron_slots.json"))["slots"]
    reg = _load(os.path.join(root, "slot_registry.json"))["slots"]
    taken = set(reg.values())
    occ = {}
    for m in (merged or []):
        occ.setdefault(int(m["layer"]), set()).add(int(m["neuron"]))
    out, notes = [], []
    for l in range(a, b + 1):
        n = slots[l][D["slot"]]
        if n in occ.get(l, ()):
            alt = next(s for s in range(8 + D["slot"] % 4, len(slots[l]), 4) if s not in taken and slots[l][s] not in occ.get(l, ()))  # per-slot fallback lane
            notes.append(f"layer {l}: slot {D['slot']} neuron {n} holds a merged unit; using slot {alt} neuron {slots[l][alt]}")
            n = slots[l][alt]
        out += [{"op": "set", "param": f"layers.{l}.mlp.gate_proj", "sel": {"neurons": [n]}, "value": {"npz": knpz, "key": f"k_{l}"}},
                {"op": "scale", "param": f"layers.{l}.mlp.gate_proj", "sel": {"neurons": [n]}, "alpha": BETA},
                {"op": "set", "param": f"layers.{l}.mlp.up_proj", "sel": {"neurons": [n]}, "value": {"npz": knpz, "key": f"k_{l}"}},
                {"op": "set", "param": f"layers.{l}.mlp.down_proj", "sel": {"neurons": [n]}, "value": {"npz": npz, "key": f"down_{direction}_{l}"}},
                {"op": "scale", "param": f"layers.{l}.mlp.down_proj", "sel": {"neurons": [n]}, "alpha": dose}]
    return out, notes


def selftest(root):
    root = root.rstrip("/")
    cases = [("fable_f1", "D1", "amplify", 21, 42, 1.0, "D1_amp_mid_+1"), ("fable_f1", "D1", "bias", 43, 63, 1.0, "D1_bias_late_+1"),
             ("fable_f1", "D1", "bias", 21, 42, 3.0, "D1_bias_mid_+3"), ("fable_c26", "C26", "bias", 21, 42, 2.0, "C26_bias_mid_+2"),
             ("fable_c26", "C26", "amplify", 21, 42, 1.0, "C26_amp_mid_+1")]
    ok = True
    for g, d, op, a, b, dose, name in cases:
        ref = os.path.join(root, g, "edits", name + ".json")
        if not os.path.exists(ref):
            print(f"SKIP {name}: {ref} missing"); continue
        E, _ = make_cell(os.path.join(root, g), d, op, a, b, dose, cells_root=root)
        same = json.dumps(E) == open(ref).read()
        ok &= same
        print(f"{'OK  ' if same else 'DIFF'} {name}: regenerated {'byte-identical' if same else 'DIFFERENT'} ({len(E)} primitives)")
    for args, expect in ((("fable_f1", "D1", "amplify", 43, 63, 3.0), "late amplify +3"), (("fable_f1", "D1", "bias", 0, 63, 1.0), "width 64"),
                         (("fable_c26", "C26", "bias", 0, 20, 1.0), "C26 early unchecked"), (("fable_f1", "D1", "bias", 21, 42, 4.0), "bias +4")):
        try:
            make_cell(os.path.join(root, args[0]), *args[1:], cells_root=root); print(f"FAIL refusal ({expect}) not raised"); ok = False
        except PermissionError as e:
            print(f"OK   refused ({expect}): {e}")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    av = sys.argv[1:]
    if av and av[0] == "--selftest":
        sys.exit(0 if selftest(av[1] if len(av) > 1 else os.path.dirname(os.path.abspath(__file__))) else 1)
    out = merged = None
    if "--out" in av:
        i = av.index("--out"); out = av[i + 1]; del av[i:i + 2]
    if "--merged" in av:
        i = av.index("--merged"); merged = _load(av[i + 1]); del av[i:i + 2]
    if len(av) != 6:
        print(__doc__); sys.exit(1)
    g, d, op, a, b, dose = av
    try:
        E, notes = make_cell(g, d, op, int(a), int(b), float(dose), merged=merged)
    except PermissionError as e:
        print(f"REFUSED: {e}", file=sys.stderr); sys.exit(2)
    for n in notes:
        print(n, file=sys.stderr)
    s = json.dumps(E)
    if out:
        with open(out, "w") as f:
            f.write(s)
        print(out)
    else:
        print(s)
