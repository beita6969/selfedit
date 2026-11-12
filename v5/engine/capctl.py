"""Adaptive in-flight cap per E server (AIMD), client side only: reads the dispatcher's stats.jsonl, writes "cap" in engines.json
(the dispatcher re-reads it every 5 s; slots above the cap pause after their current request). No server or dispatcher restart.
Per stats interval and server (state up, role E):
  preemptions in the interval > 0 and inflight <= cap (the previous cut has bitten) -> cap = max(MIN, floor(cap * DEC)), then no
                                       further decrease for HOLD intervals
  no preemption for QUIET intervals, kv < KV_HI and vLLM waiting == 0 and inflight >= cap - 1  -> cap += 1
Log: V4/capctl.log (one line per change). usage: python capctl.py [--min 24 --max 96 --dec 0.85 --hold 3 --quiet 3 --kv-hi 0.95]"""
import argparse, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("--v4", default="/data/fd/mingdazhang/selfedit/v4")
ap.add_argument("--min", type=int, default=24)
ap.add_argument("--max", type=int, default=96)
ap.add_argument("--dec", type=float, default=0.85)
ap.add_argument("--hold", type=int, default=3)
ap.add_argument("--quiet", type=int, default=3)
ap.add_argument("--kv-hi", type=float, default=0.95)
args = ap.parse_args()
V4 = args.v4
ST, EJ = f"{V4}/stats.jsonl", f"{V4}/engines.json"
log = lambda m: (print(time.strftime("%H:%M:%S"), m, flush=True), open(f"{V4}/capctl.log", "a").write(f"{time.time():.0f} {m}\n"))

pos = os.path.getsize(ST)          # only new stats lines
last = {}                          # server -> previous record
hold, quiet = {}, {}


def set_caps(new):
    e = json.load(open(EJ))
    for x in e["servers"]:
        srv = f"{x.get('host', '127.0.0.1')}:{x['port']}"
        if srv in new and x.get("state", "up") == "up":
            x["cap"] = new[srv]
    e["updated"] = time.time()
    json.dump(e, open(EJ + ".capctl.tmp", "w"), indent=1)
    os.replace(EJ + ".capctl.tmp", EJ)


log(f"start {vars(args)}")
while True:
    time.sleep(10)
    if os.path.getsize(ST) < pos:
        pos = 0
    with open(ST) as f:
        f.seek(pos)
        lines = f.readlines()
        pos = f.tell()
    new = {}
    for l in lines:
        try:
            d = json.loads(l)
        except ValueError:
            continue
        if d.get("role") not in ("E", "T") or d.get("state") != "up" or d.get("preemptions") is None or "cap" not in d:
            last.pop(d.get("server"), None)
            continue
        srv, p = d["server"], last.get(d["server"])
        last[srv] = d
        if p is None:
            continue
        cap = new.get(srv, d["cap"])
        dp = d["preemptions"] - p["preemptions"]
        if dp < 0:                  # server restarted
            continue
        hold[srv] = max(0, hold.get(srv, 0) - 1)
        if dp > 0:
            quiet[srv] = 0
            if hold[srv] == 0 and cap > args.min and d.get("inflight", 0) <= cap:     # the previous cut has bitten
                c2 = max(args.min, int(cap * args.dec))
                new[srv] = c2
                hold[srv] = args.hold
                log(f"{srv} card {d.get('card')}: {dp:.0f} preemptions, kv {d.get('kv_usage') or 0:.2f}, running {d.get('running')}: cap {cap} -> {c2}")
        else:
            quiet[srv] = quiet.get(srv, 0) + 1
            if (quiet[srv] >= args.quiet and cap < args.max and (d.get("kv_usage") or 0) < args.kv_hi and not d.get("waiting")
                    and d.get("inflight", 0) >= cap - 1):
                new[srv] = cap + 1
                quiet[srv] = args.quiet - 1          # next +1 after one more quiet interval
                log(f"{srv} card {d.get('card')}: quiet, kv {d.get('kv_usage') or 0:.2f}: cap {cap} -> {cap + 1}")
    if new:
        try:
            set_caps(new)
        except Exception as ex:
            log(f"engines.json write failed: {ex!r}")
