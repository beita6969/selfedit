"""Read-only on the run: write the exact parent read requests of the current tip on the JUDGE rows, as the loop builds them in step_parent
(gen.Run.read_req: prefix, settings, ops, seed scheme), for an engine A/A test, with NEW sample indices, key prefix "pcAA|" and
engine "T". Also writes, per row, the keys of the tip's cached parent samples 0..7 (q/out files hold them by key). Nothing is
submitted. usage: pcaa_reqs.py RUN_DIR OUT_DIR [SAMPLES=900,901]"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
import gen  # noqa: E402

RUN, OUT = sys.argv[1], sys.argv[2]
SS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "900,901").split(",")]
R = gen.Run(gen.args(["--run", RUN]))
R.save = lambda: None
tip = R.state["tip"]
R.load_routes(tip["entries"], R.judge_rows)
os.makedirs(OUT, exist_ok=True)
fq, fr = os.path.join(OUT, f"pcAA_{tip['id']}_reqs.jsonl"), os.path.join(OUT, f"pcAA_{tip['id']}_refkeys.jsonl")
nq = nh = 0
with open(fq, "w") as a, open(fr, "w") as b:
    for r in R.judge_rows:
        for s in SS:
            q, _ = R.read_req(tip["entries"], r, s, 2)
            q = dict(q, key="pcAA|" + q["key"], engine="T")
            a.write(json.dumps(q, ensure_ascii=False) + "\n")
            nq += 1
        ks = [R.read_req(tip["entries"], r, s, 2)[0]["key"] for s in range(gen.P_FULL)]
        have = [k for k in ks if R.io.get(k) is not None]
        nh += len(have)
        b.write(json.dumps(dict(row=r, tip=tip["id"], keys=ks, cached=have,
                                ok=[bool(R.io.get(k).get("ok")) for k in have], tokens=[R.io.get(k).get("tokens") for k in have])) + "\n")
print(f"tip {tip['id']}: {nq} requests ({len(R.judge_rows)} JUDGE rows x samples {SS}) -> {fq}; cached parent samples {nh} "
      f"({nh / max(len(R.judge_rows), 1):.2f} per row) -> {fr}")
print("example request:", json.dumps(json.loads(open(fq).readline()), ensure_ascii=False)[:700])
