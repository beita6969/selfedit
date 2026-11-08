#!/usr/bin/env python3
"""Mac: keeps one card worker per card of cw/cards.txt running, when there is unclaimed work. Only starts things through gpu bg; a dropped link only delays a start."""
import os, re, shlex, subprocess, time
H = os.path.dirname(os.path.abspath(__file__)); V = "/workspace/fudan/mingdazhang/selfedit/code-xf"; Wf = "/data/fd/mingdazhang/selfedit/rsi/mm"
def sh(c, t=120):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t).stdout
    except subprocess.TimeoutExpired:
        return ""
CHK = f"""python3 - <<'P'
import os
CL='{Wf}/hb/claims'; allow=tuple(open(CL+'/ALLOW').read().split()); n=0
for f in ('{Wf}/evoq.txt','{Wf}/evoq_extra.txt'):
    if os.path.exists(f):
        for l in open(f):
            k=l.split(chr(9))[0]
            if chr(9) in l and k.startswith(allow) and not os.path.exists(CL+'/'+k): n+=1
print('OPEN',n)
P"""
while True:
    if sh('pgrep -f "gpu (get|put) "', 10).strip():      # a transfer is on the link: look next pass
        time.sleep(30)
    jobs = sh("gpu jobs")
    if "selfedit-" in jobs:                              # the supervisor reads this and makes no call of its own
        open(f"{H}/jobs.cache.tmp", "w").write(jobs)
        os.replace(f"{H}/jobs.cache.tmp", f"{H}/jobs.cache")
    if "selfedit-" in jobs and not os.path.exists(f"{H}/hold"):
        up = {m[0]: int(m[1]) for m in re.findall(r"selfedit-cw(\d)_(\d+)\s+running", jobs)}
        num = max([int(x) for x in re.findall(r"selfedit-cw\d_(\d+)", jobs)] or [0]) + 1
        need = [c for c in open(f"{H}/cards.txt").read().split() if c not in up]
        if need:
            m = re.search(r"OPEN (\d+)", sh(f"gpu run {shlex.quote(CHK)}"))
            if m and int(m[1]) > 0:
                for c in need[:int(m[1])]:
                    r = sh(f"gpu bg --name selfedit-cw{c}_{num} --gpus {c} --mem 78 " + shlex.quote(f"python3 -u {V}/cw/card_worker.py"))
                    print(time.strftime("%H:%M:%S"), "card", c, r.strip()[-80:], flush=True)
    time.sleep(120)
