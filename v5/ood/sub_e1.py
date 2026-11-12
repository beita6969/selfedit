"""Submit the entry-1-always arms (coordinator 11-11): ood_e1always (768 OOD v2 x s0-1, prio -10, after ood_tipE), sig_e1always and sig_tipE
(324 signal rows x s0-1, prio -8, after ood_floor8k which runs at -9). E path; keys prefixed so they never collide with earlier reads."""
import json, os, subprocess, sys, time
V4 = "/data/fd/mingdazhang/selfedit/v4"; O = "/workspace/fudan/mingdazhang/selfedit-v4/ood"; E = "/workspace/fudan/mingdazhang/selfedit-v4/engine_v4"; X = f"{V4}/scratch/ext"
run = lambda *a: print(subprocess.run(["python3", *a], capture_output=True, text=True, cwd=V4).stderr.strip()[-600:])
run(f"{O}/tipview_e1.py", "t007c", "ood_v2", "0,1", f"{X}/tip_ood_e1.jsonl", "1", f"{X}/e1_routed_ood.json")
run(f"{O}/tipview_e1.py", "t007c", "show,monitor", "0,1", f"{X}/tip_sig_e1.jsonl", "1", f"{X}/e1_routed_sig.json")
run(f"{E}/tipview.py", "t007c", "show,monitor", "0,1", f"{X}/tip_sig.jsonl")
def sub(name, src, tag, prio):
    L = [dict(q, key=tag + "|" + q["key"], engine="E", priority=prio) for q in map(json.loads, open(src))]
    assert len({q["key"] for q in L}) == len(L) and all("edit" not in q and "ops" not in q for q in L)
    p = f"{V4}/q/in/{name}.jsonl"; open(p + ".tmp", "w").write("".join(json.dumps(x) + "\n" for x in L)); os.replace(p + ".tmp", p)
    open(f"{X}/submit_times.jsonl", "a").write(json.dumps(dict(batch=name, t_submit=time.time(), n=len(L))) + "\n")
    print(name, len(L)); time.sleep(1.1)
sub("ood_e1always", f"{X}/tip_ood_e1.jsonl", "oe1", -10)
sub("sig_e1always", f"{X}/tip_sig_e1.jsonl", "se1", -8)
sub("sig_tipE", f"{X}/tip_sig.jsonl", "stip", -8)
