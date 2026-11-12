"""Read-only check of a run before the generation-4 code set takes over (never writes state): imports, weight-cell menu from
<root>/cells, the stack candidate the next generation would build, version state as created lazily, and the grader path of ecomp.
usage: precheck_g4.py RUN_DIR [ROOT]"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
import gen  # noqa: E402
import ecomp  # noqa: E402

RUN = sys.argv[1]
argv = ["--run", RUN] + (["--root", sys.argv[2], "--rows", os.path.join(sys.argv[2], "rows/learn.jsonl")] if len(sys.argv) > 2 else [])
R = gen.Run(gen.args(argv))
R.save = lambda: None                      # read only
g_last = max(int(k) for k in R.state["gens"])
print("STEPS", gen.STEPS, "| last generation", g_last, "done", R.state["gens"][str(g_last)]["done"])
R.g = g_last + 1
m = R.w_menu()
groups = {}
for gname, name, ed in m:
    groups.setdefault(gname, []).append(name)
print("weight menu:", {k: len(v) for k, v in groups.items()})
for k, v in groups.items():
    if k != "band":
        print(" ", k, v)
        for gname, name, ed in m:
            if gname == k:
                print("   ", name, json.dumps(ed), "file exists" if ed.get("op") != "surgeon" or os.path.exists(ed.get("file", "")) else "FILE MISSING")
print("fresh for tip", R.state["tip"]["id"], ":", len(R.w_fresh()))
st = R.stack_entry()
print("stack for next generation:", None if st is None else [(x["cid"], x["condition"][:60], x.get("settings")) for x in st["parts"]])
vs = R.v_state()
print("versions:", list(vs["versions"]), "inc", vs["inc"], "challengers", vs["challengers"])
print("grader:", ecomp._load_grader().__file__ if hasattr(ecomp, "_load_grader") else "n/a")
print("slots (Stage 1 formula):", R.slots())
