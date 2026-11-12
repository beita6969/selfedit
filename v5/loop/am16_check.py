"""Read-only AM16 check on a live run (never writes state): what the measured sizing would give for the next generation from the last
finished one (capacity while backlogged, per-slot tokens, pre-read wall, W rate), the arm-B setting cells drawn from the ranges, and a
draw of weight cells by kind. usage: am16_check.py RUN_DIR [ROOT]"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
import gen  # noqa: E402

RUN = sys.argv[1]
argv = ["--run", RUN] + (["--root", sys.argv[2], "--rows", os.path.join(sys.argv[2], "rows/learn.jsonl")] if len(sys.argv) > 2 else [])
R = gen.Run(gen.args(argv))
R.save = lambda: None
fin = [int(k) for k, G in R.state["gens"].items() if "report" in (G.get("done") or [])]
R.g = max(int(k) for k in R.state["gens"]) + (0 if any("report" not in (G.get("done") or []) for G in R.state["gens"].values()) else 1)
R.state["gens"].setdefault(str(R.g), dict(t0=0, done=[], times={}, parent=dict(new=0, bg=0)))
print("finished generations", sorted(fin), "| sizing as for generation", R.g)
for g in sorted(fin)[-2:]:
    G = R.state["gens"][str(g)]
    t0, t1 = R.gen_window(G)
    print(f"g{g}: window {t1 - t0:.0f} s; E capacity backlogged {R.e_capacity(t0, t1)} tok/s; W {R.w_capacity(t0, t1)}; costs {json.dumps(R.slot_costs(g))}")
sl = R.slots_am16()
print("slots_am16:", json.dumps(sl, default=str))
print("arm-B setting cells:", R.b_cells_am16())
props = []
R.add_w_props_am16(props, dict(BW=sl["BW"]))
for p in props:
    print("  weight cell:", p["cand"]["cell_group"], p["cand"]["cell"], json.dumps(p["cand"]["edit"]))
