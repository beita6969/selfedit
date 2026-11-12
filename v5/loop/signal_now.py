"""B1 by hand: the signal export of the run's current tip (traces.json + traces_meta.json) and the candidate reads of the current and the
previous generation (traces_cand.jsonl per tip), read only: state.json is never written (the live loop owns it); the loop's own export
later continues the same files (traces_cand.keys). usage: signal_now.py RUN_DIR [gen.py args, e.g. --root]"""
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
import gen  # noqa: E402

t0 = time.time()
a = gen.args(["--run", sys.argv[1]] + sys.argv[2:])
R = gen.Run(a)                                   # the IO reads every result line once; show + monitor reads get their line location
R.save = lambda: (_ for _ in ()).throw(RuntimeError("signal_now.py never writes state.json"))
R.g = max(int(k) for k in R.state["gens"])
print(f"loaded in {time.time() - t0:.0f}s: {len(R.io.res)} results, {len(R.io.loc)} show/monitor reads located; generation {R.g}, tip {R.state['tip']['id']}")
tips = [R.state["tip"]] + [t for t in R.state["tips"] if t["id"] in R.cand_tips()]
for t in tips:
    R.load_routes(t["entries"], R.signal_rows())
rec = R.export_traces("by hand (signal_now.py)", force=True, record=False)
print("traces:", rec)
for tid in R.cand_tips() or [R.state["tip"]["id"]]:
    print(f"traces_cand {tid}: {R.export_traces_cand('by hand (signal_now.py)', tid)} lines appended")
print(f"done in {time.time() - t0:.0f}s")
