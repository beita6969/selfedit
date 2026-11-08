"""Digest of a stop-read scan (t_hazard's RUN/hazard.jsonl) for the surgeon's dossier: the strongest edits each way within the drift cap, per direction.
Usage: python rsi/stop_map.py RUN/hazard.jsonl [RUN2/hazard.jsonl ...] OUT.md [--cap 0.01] [--min 0.1] [--top 12]"""
import argparse, json, os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("scans", nargs="+"); ap.add_argument("out")
ap.add_argument("--cap", type=float, default=0.01); ap.add_argument("--min", type=float, default=0.1); ap.add_argument("--top", type=int, default=12)
a = ap.parse_args()
rows = [json.loads(l) for f in a.scans for l in open(f)]
null = [r for r in rows if "random" in r["edit"]]
best = {}
for r in rows:                                  # later scans read more kinds of point: keep the fullest row of each edit
    if "u" in r["edit"]:
        k = json.dumps(r["edit"], sort_keys=True)
        if k not in best or sum(abs(r.get(x, 0)) > 0 for x in ("d_ans", "d_post", "d_pre")) >= sum(abs(best[k].get(x, 0)) > 0 for x in ("d_ans", "d_post", "d_pre")):
            best[k] = r
real = list(best.values())


def line(r):
    e = r["edit"]
    opt = "".join(f"{name} {r[k]:+.2f}, " for k, name in (("d_post", "after the answer"), ("d_pre", "before it"), ("d_ans", "at lines stating it")) if abs(r.get(k, 0)) > 0)
    return (f"- {e['param']} {e['u']} alpha {e['alpha']:g}: stop {r['d_cut']:+.2f}+-{r['se_cut']:.2f}, {opt}natural end {r.get('d_end', 0):+.2f}, own tokens {r['d_all']:+.4f}, revisit words {r['d_rev']:+.3f}")


ok = [r for r in real if -r["d_all"] <= a.cap]
up = sorted([r for r in ok if r["d_cut"] >= a.min], key=lambda r: -r["d_cut"])[: a.top]
down = sorted([r for r in ok if r["d_cut"] <= -a.min], key=lambda r: r["d_cut"])[: a.top // 2]
by = defaultdict(list)
for r in ok:
    by[r["edit"]["u"].split(":")[1]].append(r)
md = [f"# Stop map: {len(real)} single-layer edits measured on your own reasoning traces, without generating",
      f"stop: change in log-probability of closing your thinking at late points of a trace; after/before the answer: the same at points after or before the correct answer first appears in the trace. "
      f"Random directions at the same strength: stop {' '.join(f'{r['d_cut']:+.2f}' for r in null) or '(none yet)'}. Edits that move your own tokens by more than {a.cap} per token are left out.",
      "\nMost raises stop:"] + [line(r) for r in up] + ["\nMost lowers stop:"] + [line(r) for r in down]
md += ["\nMost selective (after the answer minus before it):"] + [line(r) for r in sorted(ok, key=lambda r: -(r.get("d_post", 0) - r.get("d_pre", 0)))[: a.top]]
md += ["\nMost raises the close at lines stating the answer:"] + [line(r) for r in sorted(ok, key=lambda r: -r.get("d_ans", 0))[: a.top // 2]]
md += ["\nBest layer per direction:"]
for d, rs in sorted(by.items()):
    r = max(rs, key=lambda r: r["d_cut"])
    md.append(f"- {d}: {r['edit']['param']} alpha {r['edit']['alpha']:g}, stop {r['d_cut']:+.2f}")
tmp = a.out + ".tmp"
open(tmp, "w").write("\n".join(md) + "\n")
os.replace(tmp, a.out)
print(f"{len(rows)} rows -> {a.out}")
