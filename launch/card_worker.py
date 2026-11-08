#!/usr/bin/env python3
"""One per Vesta card, started with gpu bg --gpus N. Takes queue lines from the driver's file on the server, one claim file per line, so batches keep starting while the Mac link is down.
A claim is O_EXCL: two workers, or a worker and the node bridge, cannot take one line. Runs the same command with the same environment, same .log and .rc as gpu bg. Children stay in this process group, so gpu stop ends both. Leaves after 45 idle minutes, holding nothing."""
import json, os, subprocess, sys, time
D = "/data/fd/mingdazhang/selfedit"; W = f"{D}/rsi/mm"; CL = f"{W}/hb/claims"; L = "/data/fd/mingdazhang/gpu-logs"
C = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
HEAD = ("SE=/workspace/fudan/mingdazhang/selfedit; D=/data/fd/mingdazhang/selfedit; W=$D/rsi/mm; . $SE/venv/bin/activate; "
        "unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset $(env | grep -o '^TRITON_[A-Z_]*'); export MPLBACKEND=Agg HF_HOME=$D/.hf VLLM_CACHE_ROOT=$D/.vllm-cache; cd $SE/code-xf; ")
os.makedirs(CL, exist_ok=True)
def say(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)
def words(f):
    return open(f).read().split() if os.path.exists(f) else []
def lines():
    out = []
    for f in (f"{W}/evoq.txt", f"{W}/evoq_extra.txt"):
        if os.path.exists(f):
            for l in open(f):
                if "\t" in l:
                    out.append(l.rstrip("\n").split("\t", 1))
    return out
def used():
    try:
        return int(subprocess.run(f"nvidia-smi -i {C} --query-gpu=memory.used --format=csv,noheader,nounits", shell=True, capture_output=True, text=True, timeout=60).stdout.strip())
    except Exception:
        return 99999
def claim(name):
    try:
        fd = os.open(f"{CL}/{name}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, json.dumps(dict(card=C, t=time.time(), pid=os.getpid())).encode()); os.close(fd)
    return True
# a batch this card was running when its worker was stopped has a claim and no .rc: free it (rename, nothing deleted)
for f in os.listdir(CL):
    p = f"{CL}/{f}"
    try:
        x = json.load(open(p))
    except Exception:
        continue
    if x.get("card") == C and not os.path.exists(f"{L}/selfedit-{f}.rc"):
        os.replace(p, f"{p}.lost.{int(time.time())}")
        say(f"freed {f}, left without a result")
idle = time.time()
while not os.path.exists(f"{CL}/STOP") and time.time() - idle < 2700:
    if used() >= 8000:
        time.sleep(30)
        continue
    allow, low = tuple(words(f"{CL}/ALLOW")), tuple(words(f"{CL}/LOW"))
    took = None
    for name, cmd in sorted((x for x in lines() if x[0].startswith(allow)), key=lambda x: x[0].startswith(low)):
        if not os.path.exists(f"{CL}/{name}") and claim(name):
            took = (name, cmd)
            break
    if not took:
        time.sleep(15)
        continue
    name, cmd = took
    say(f"start {name}")
    t0 = time.time()
    with open(f"{L}/selfedit-{name}.log", "ab") as o:
        rc = subprocess.run(["bash", "-c", HEAD + cmd], stdout=o, stderr=subprocess.STDOUT).returncode
    open(f"{L}/selfedit-{name}.rc", "w").write(f"{rc}\n")
    say(f"end {name} rc {rc} {time.time() - t0:.0f}s")
    idle = time.time()
    time.sleep(10)
say("leaving")
