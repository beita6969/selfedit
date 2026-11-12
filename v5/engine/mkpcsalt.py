"""Salted prefix-cache A/A batch (coordinator 22:5x item B): 444 JUDGE rows under the current tip's view (t002 = t005r parent view),
samples 950/951 (same CRN seeds in both arms). Arm S: the two samples share one cache_salt (sent back to back to one server);
arm P: plain (unique salt). Engine "PC" = only the servers with prefix caching on (roles E + PC). usage: python mkpcsalt.py [go]"""
import json, glob, sys, os
V4 = "/data/fd/mingdazhang/selfedit/v4"
role = {json.loads(l)["id"]: json.loads(l)["v4_role"] for l in open(f"{V4}/rows/learn.jsonl")}
par = {}
for l in open(glob.glob(f"{V4}/q/in/g004_parentbg_*.jsonl")[0]):
    q = json.loads(l)
    if q["sample"] == 4 and role[q["row"]] == "judge":
        par[q["row"]] = q
out = []
for r, q0 in sorted(par.items()):
    sg = q0["key"].split("|")[1]
    for arm in ("S", "P"):
        for s in (950, 951):
            q = dict(key=f"pc{arm}|{sg}|{r}|{s}", mode="read", engine="PC", priority=7, row=r, sample=s, prefix=q0.get("prefix"), settings=q0["settings"])
            if arm == "S":
                q["salt"] = f"pcS|{sg}|{r}"
            out.append(q)
print(len(par), "rows,", len(out), "reads")
if len(sys.argv) > 1 and sys.argv[1] == "go":
    open(f"{V4}/q/in/.pcsalt.tmp", "w").write("".join(json.dumps(x) + "\n" for x in out))
    os.replace(f"{V4}/q/in/.pcsalt.tmp", f"{V4}/q/in/pcsalt_t002_1.jsonl"); print("submitted pcsalt_t002_1")
