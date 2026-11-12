"""report.py sweep | cache : throughput per sweep step from stats.jsonl, or the parent cache summary."""
import collections, json, sys, glob
import numpy as np
V4 = "/data/fd/mingdazhang/selfedit/v4"
what = sys.argv[1]
if what == "sweep":
    steps = [json.loads(l) for l in open(f"{V4}/logs/sweep.jsonl")]
    st = [json.loads(l) for l in open(f"{V4}/stats.jsonl")]
    ends = [s["t"] for s in steps[1:]] + [1e18]
    for port in (8200, 8201):
        srv = f"127.0.0.1:{port}"
        for s, e in zip(steps, ends):
            cap = s["caps"][str(port)]
            w = [x for x in st if x["server"] == srv and s["t"] + 150 <= x["t"] <= e and x.get("gen_tok_s") is not None]
            if not w:
                continue
            g = [x["gen_tok_s"] for x in w]
            fin = sum(x.get("finished_window") or 0 for x in w)
            dur = w[-1]["t"] - w[0]["t"] + 30
            pre = (w[-1].get("preemptions") or 0) - (w[0].get("preemptions") or 0)
            print(json.dumps(dict(server=srv, cap=cap, minutes=round(dur / 60, 1), gen_tok_s=round(float(np.mean(g))), gen_tok_s_sd=round(float(np.std(g))),
                                  running_mean=round(float(np.mean([x["running"] or 0 for x in w])), 1), peak_running=max(x["running"] or 0 for x in w),
                                  waiting_max=max(x["waiting"] or 0 for x in w), preempt=pre, kv_max=round(max(x["kv_usage"] or 0 for x in w), 3),
                                  finished=fin, rows_h=round(fin * 3600 / dur))))
else:
    R = []
    for f in glob.glob(f"{V4}/q/out/p000_parent_*.jsonl"):
        R += [json.loads(l) for l in open(f)]
    R = [r for r in R if "error" not in r]
    errs = sum(1 for f in glob.glob(f"{V4}/q/out/p000_parent_*.jsonl") for l in open(f) if '"error"' in l)
    by = collections.defaultdict(list)
    for r in R:
        by[r["bench"]].append(r)
    acc = {}
    print("bench samples rows acc tok_mean tok_p90 unfinished loop")
    for b, rs in sorted(by.items()):
        a = np.mean([r["ok"] for r in rs]); acc[b] = a
        t = [r["tokens"] for r in rs]
        print(b, len(rs), len({r["row"] for r in rs}), round(100 * a, 1), round(np.mean(t)), int(np.percentile(t, 90)),
              round(100 * np.mean([r["finish"] != "stop" for r in rs]), 1), round(100 * np.mean([r["loop"] for r in rs]), 1))
    t = [r["tokens"] for r in R]
    print("ALL", len(R), "macro", round(100 * np.mean(list(acc.values())), 2), "micro", round(100 * np.mean([r["ok"] for r in R]), 2),
          "tok_mean", round(np.mean(t)), "unfinished %", round(100 * np.mean([r["finish"] != "stop" for r in R]), 2), "errors", errs,
          "grade_errors", sum(1 for r in R if r.get("grade_error")))
    if R:
        t0, t1 = min(r["t_submit"] for r in R), max(r["t_done"] for r in R)
        per = collections.Counter(r["server"] for r in R)
        print("wall min", round((t1 - t0) / 60, 1), "samples/h", round(len(R) * 3600 / (t1 - t0)), "per server", dict(per))
        lat = [r["t_done"] - r["t_submit"] for r in R]
        print("latency s p50 p95 max", round(np.median(lat)), round(np.percentile(lat, 95)), round(max(lat)))
