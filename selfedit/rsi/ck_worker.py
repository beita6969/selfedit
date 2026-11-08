#!/usr/bin/env python3
"""Checkpoint builds without a card: vision_patch.py is CPU only and takes seconds, and a card worker only claims when its card is free (r2 waited 8 min for an 8 s job). Same claim, log and .rc as a card worker.
Env for a test on its own queue: CK_W (dir with evoq.txt, evoq_extra.txt), CK_CL (claims), CK_LOG (logs), CK_CODE (cd there), CK_ONCE=1 (one pass, then out)."""
import json, os, subprocess, time
D = "/data/fd/mingdazhang/selfedit"
W = os.environ.get("CK_W", f"{D}/rsi/mm")
CL = os.environ.get("CK_CL", f"{W}/hb/claims")
L = os.environ.get("CK_LOG", "/data/fd/mingdazhang/gpu-logs")
CODE = os.environ.get("CK_CODE", "$SE/code-xf")
HEAD = (f"SE=/workspace/fudan/mingdazhang/selfedit; D=/data/fd/mingdazhang/selfedit; W=$D/rsi/mm; . $SE/venv/bin/activate; unset LD_LIBRARY_PATH; export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 HF_HOME=$D/.hf; cd {CODE}; export PYTHONPATH=.:pb; ")
while not os.path.exists(f"{CL}/STOP"):
    allow = tuple(open(f"{CL}/ALLOW").read().split())
    for f in (f"{W}/evoq.txt", f"{W}/evoq_extra.txt"):
        if not os.path.exists(f):
            continue
        for l in open(f):
            if "\t" not in l:
                continue
            name, cmd = l.rstrip("\n").split("\t", 1)
            if not name.startswith(allow) or "rsi/vision_patch.py" not in cmd or os.path.exists(f"{CL}/{name}"):
                continue
            try:
                fd = os.open(f"{CL}/{name}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.write(fd, json.dumps(dict(card="cpu", t=time.time(), pid=os.getpid())).encode()); os.close(fd)
            t0 = time.time()
            with open(f"{L}/selfedit-{name}.log", "ab") as o:
                rc = subprocess.run(["bash", "-c", HEAD + cmd], stdout=o, stderr=subprocess.STDOUT).returncode
            open(f"{L}/selfedit-{name}.rc", "w").write(f"{rc}\n")
            print(time.strftime("%H:%M:%S"), name, "rc", rc, round(time.time() - t0), flush=True)
    if os.environ.get("CK_ONCE"):
        break
    time.sleep(5)
