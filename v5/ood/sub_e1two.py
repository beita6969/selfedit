"""PROBE e1two_probe (coordinator 11-11; hand-drafted hypothesis wording, NOT a system proposal): tip t007c with entry 1's text replaced by
e1two_probe_text.txt and its condition = always. ood_e1two_probe (768 x s0-1) and sig_e1two_probe (324 x s0-1), E path, prio -7 (after ood_wo_*, before lrn_wo_*)."""
import json, os, subprocess, time
V4 = "/data/fd/mingdazhang/selfedit/v4"; O = "/workspace/fudan/mingdazhang/selfedit-v4/ood"; X = f"{V4}/scratch/ext"
for role, side in (("ood_v2", "ood"), ("show,monitor", "sig")):
    e = subprocess.run(["python3", f"{O}/tipview_e1two.py", "t007c", role, "0,1", f"{X}/tip_{side}_e1two.jsonl", "1", f"{X}/e1_routed_{side}.json", f"{O}/e1two_probe_text.txt"], cwd=V4, capture_output=True, text=True)
    print(side, e.stderr.strip().splitlines()[-2:]); assert e.returncode == 0, e.stderr[-800:]
for name, src, tag in (("ood_e1two_probe", "tip_ood_e1two.jsonl", "oe2p"), ("sig_e1two_probe", "tip_sig_e1two.jsonl", "se2p")):
    L = [dict(q, key=tag + "|" + q["key"], engine="E", priority=-7, probe="e1two_probe") for q in map(json.loads, open(f"{X}/{src}"))]
    assert len({q["key"] for q in L}) == len(L)
    p = f"{V4}/q/in/{name}.jsonl"; open(p + ".tmp", "w").write("".join(json.dumps(x) + "\n" for x in L)); os.replace(p + ".tmp", p)
    open(f"{X}/submit_times.jsonl", "a").write(json.dumps(dict(batch=name, t_submit=time.time(), n=len(L))) + "\n")
    print(name, len(L)); time.sleep(1.1)
