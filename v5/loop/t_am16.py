"""AM16 checks (coordinator 11-10): derived L1 floor, equal-compute rounding, arm-B ranges (settings and weight cells), capacity
measurement from stats.jsonl and W results, and the measured slot sizing on a finished fake run. usage: t_am16.py [E2E_ROOT]"""
import json, math, os, random, sys, tempfile, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
import numpy as np  # noqa: E402
import ecomp as ECOMP  # noqa: E402
import gen  # noqa: E402
import versions as V  # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("ok   " if cond else "FAIL ") + name)
    ok, fail = ok + bool(cond), fail + (not cond)


# 2. derived floor
check("l1_floor(.10) = 3, (.05) = 3, (.01) = 5", (V.l1_floor(.10), V.l1_floor(.05), V.l1_floor(.01)) == (3, 3, 5))
check("FLOOR_P is the derived floor at ALPHA", V.FLOOR_P == V.l1_floor(V.ALPHA) == 3)
# 3. rounding
check("ref_m_round: 0.6, 1.49 -> None; 1.5 -> 2; 2.3 -> 2; 2.5 -> 3; 3.49 -> 3",
      [ECOMP.ref_m_round(c) for c in (0.6, 1.49, 1.5, 2.3, 2.5, 3.49)] == [None, None, 2, 2, 3, 3])
check("ref_m (old, ceil) unchanged: 2.3 -> 3", ECOMP.ref_m(2.3) == 3)

# 4. arm-B ranges on a stand-in self
knobs = json.load(open(os.path.join(HERE, "knobs.json")))
S = types.SimpleNamespace(knobs=knobs, g=6, a=types.SimpleNamespace(seed=0), stock={k: knobs["settings"][k]["stock"] for k in knobs["settings"]},
                          state={"tip": {"entries": [dict(cond="always", settings={"min_image_tokens": 4096})]}})
S.settings_of = lambda es: {**S.stock, **{k: v for e in es for k, v in (e.get("settings") or {}).items()}}
S.tip_cands = lambda: [dict(author="B", entry=dict(settings={"max_image_tokens": 8192}))]
for f in ("b_ranges", "pow2", "b_cells_am16", "b_setting_problems", "band_cell"):
    setattr(S, f, types.MethodType(getattr(gen.Run, f), S))
cells = S.b_cells_am16()
by = {}
for k, v in cells:
    by.setdefault(k, []).append(v)
check(f"b_cells: min above and below 4096 {by.get('min_image_tokens')}", sorted(v > 4096 for v in by.get("min_image_tokens", [])) == [False, True])
check(f"b_cells: max only below the engine limit 16384, never the tried 8192 {by.get('max_image_tokens')}",
      len(by.get("max_image_tokens", [])) == 1 and by["max_image_tokens"][0] < 16384 and by["max_image_tokens"][0] != 8192)
check(f"b_cells: temperature one above and one below 1.0 inside [.5, 1.5] {by.get('temperature')}",
      sorted(v > 1.0 for v in by.get("temperature", [])) == [False, True] and all(.5 <= v <= 1.5 for v in by["temperature"]))
check("b_cells: deterministic per (generation, seed)", S.b_cells_am16() == cells)
check("b_cells: powers of two only", all(v & (v - 1) == 0 for k, v in cells if k != "temperature"))
check("b_setting_problems: 256 ok, 300 not, temperature 2.0 not, unknown knob not",
      (S.b_setting_problems(dict(settings={"min_image_tokens": 256})), bool(S.b_setting_problems(dict(settings={"min_image_tokens": 300}))),
       bool(S.b_setting_problems(dict(settings={"temperature": 2.0}))), bool(S.b_setting_problems(dict(settings={"top_p": .9})))) == ([], True, True, True))
rng = random.Random(1)
bc = [S.band_cell(rng) for _ in range(4000)]
ops = {e["op"] for _, e in bc}
check(f"band_cell: ops {sorted(ops)}", ops == {"scale", "attn_temp", "vision_scale"})
al = [e["alpha"] for _, e in bc]
check("band_cell: |alpha - 1| in [.05, .3], both signs", all(.049 <= abs(a - 1) <= .301 for a in al) and min(al) < 1 < max(al))
lay = [e.get("layers") or [e.get("first"), e.get("last")] for _, e in bc if e["op"] != "vision_scale"]
check("band_cell: contiguous bands inside 0-63, widths from 1 to 64", all(0 <= a <= b <= 63 for a, b in lay)
      and min(b - a for a, b in lay) == 0 and max(b - a for a, b in lay) >= 50)
lg = np.log(np.abs(np.array(al) - 1))
check("band_cell: |dose| log-uniform (mean of log |dose| near the interval's log midpoint)", abs(lg.mean() - (math.log(.05) + math.log(.3)) / 2) < .05)
check("P0 menu untouched: knobs['settings'] has no range keys", all("values" in v for v in knobs["settings"].values()))

# 1. capacity measurement
d = tempfile.mkdtemp(prefix="t_am16_")
with open(os.path.join(d, "stats.jsonl"), "w") as f:
    for t in range(100, 200, 5):
        f.write(json.dumps(dict(t=t, server="a", role="E", cap=30, state="up", inflight=30 if t < 150 else 10, gen_tok_s=800 if t < 150 else 300)) + "\n")
        f.write(json.dumps(dict(t=t, server="b", role="E", cap=30, state="up", inflight=5, gen_tok_s=400 + t % 3)) + "\n")
        f.write(json.dumps(dict(t=t, server="c", role="E", cap=30, state="drain", inflight=30, gen_tok_s=900)) + "\n")
        f.write(json.dumps(dict(t=t, server="w", role="W", cap=1, state="up", inflight=1, gen_tok_s=999)) + "\n")
    f.write(json.dumps(dict(t=500, server="a", role="E", cap=30, state="up", inflight=30, gen_tok_s=5000)) + "\n")
R = types.SimpleNamespace(a=types.SimpleNamespace(root=d))
cap = gen.Run.e_capacity(R, 100, 200)
check(f"e_capacity: backlogged mean of a (800) + peak of never-backlogged b (402); drained c and W and out-of-window left out -> {cap}", cap == 800 + 402)
os.makedirs(os.path.join(d, "w"))
for n in ("W3", "W7"):
    open(os.path.join(d, "w", f"{n}.up.json"), "w").write("{}")
res = {f"k{i}": dict(server="W3", t_done=1000 + 2 * i) for i in range(101)}
res.update({f"j{i}": dict(server="W7", t_done=1000 + 4 * i) for i in range(51)})
R.io = types.SimpleNamespace(res=res)
r, cards = gen.Run.w_capacity(R, 0, 10 ** 6)
check(f"w_capacity: W3 .5 reads/s, W7 .25 -> mean .375 x 2 cards = .75 -> {r}, cards {cards}", abs(r - .75) < 1e-9 and cards == 2)

# slots_am16 on a finished fake run (t_versions e2e root, if given)
if len(sys.argv) > 1:
    root = sys.argv[1]
    Rr = gen.Run(gen.args(["--root", root, "--rows", os.path.join(root, "rows", "learn.jsonl"), "--run", os.path.join(root, "run")]))
    Rr.save = lambda: None
    g_last = max(int(k) for k in Rr.state["gens"])
    Rr.g = g_last + 1
    Rr.state["gens"][str(Rr.g)] = dict(t0=0, done=[], times={}, parent=dict(new=0, bg=100))
    sl = Rr.slots_am16()
    check(f"slots_am16 on the e2e run: {json.dumps(sl['am16'])[:300]}", sl["P0"] >= Rr.l1_floor_total() and sl["am16"]["per_slot_from"] == f"g{g_last}")
    props = []
    Rr.add_w_props_am16(props, dict(BW=6))
    check(f"add_w_props_am16: 6 weight cells, kinds {[p['cand']['cell_group'] for p in props]}", len(props) == 6 and all(p["cand"]["edit"] for p in props))
print(f"\n{ok} ok, {fail} failed")
sys.exit(1 if fail else 0)
