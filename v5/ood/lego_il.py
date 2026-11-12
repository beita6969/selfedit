"""Pilot (coordinator 11-11): official interleaved layout for lego rows, WITHOUT touching engine_v4/common.py.
Each lego learn row gets a pilot twin legoil/<id> (role pilot_lego_il) whose prompt has every "<image k>" marker replaced by "<image>" and
image_paths reordered to marker order, so common.content_for's interleaved branch puts image k at marker k (prefix part first, as there).
Rows whose markers are not exactly a permutation of 1..n_images keep the standard layout (twin = original prompt; counted).
Requests carry seed = crn_seed(original row, sample), so the twin reads use the same seeds. Arms (E path, s0-1, prio -9):
lego_std_stock, lego_il_stock (stock settings, no prefix), lego_std_tip, lego_il_tip (tip t007c view). Usage: lego_il.py [--dry]"""
import collections, hashlib, json, os, re, shutil, subprocess, sys, time, zlib
V4 = "/data/fd/mingdazhang/selfedit/v4"; F = f"{V4}/rows/learn.jsonl"; X = f"{V4}/scratch/ext"; E = "/workspace/fudan/mingdazhang/selfedit-v4/engine_v4"
crn = lambda row, s: zlib.crc32(f"101|0|{row}".encode()) % (2 ** 31 - 1024) + int(s)
lst = lambda v: v if isinstance(v, list) else ([v] if v else [])
L = [json.loads(l) for l in open(F)]
lego = [r for r in L if "lego" in r["bench"].lower() and r.get("v4_role") in ("show", "monitor", "judge")]
print("lego learn rows", len(lego), collections.Counter(r["bench"] for r in lego), "multi-image", sum(len(lst(r["image_paths"])) > 1 for r in lego))
twins, fb, plain = [], [], 0
for r in lego:
    ims = lst(r["image_paths"]); ks = [int(k) for k in re.findall(r"<image (\d+)>", r["prompt"])]
    if "<image>" in r["prompt"]: plain += 1
    t = dict(r, id="legoil/" + r["id"], v4_role="pilot_lego_il", hb_role="pilot_lego_il", il_src=r["id"])
    if ims and sorted(ks) == list(range(1, len(ims) + 1)) and "<image>" not in r["prompt"]:
        t["prompt"] = re.sub(r"<image \d+>", "<image>", r["prompt"]); t["image_paths"] = [ims[k - 1] for k in ks]; t["il"] = True
    else:
        fb.append((r["id"], len(ims), ks)); t["il"] = False
    twins.append(t)
print(f"interleaved {sum(t['il'] for t in twins)}; fall back {len(fb)} {fb[:6]}; prompts with plain <image> {plain}")
if "--dry" in sys.argv: sys.exit()
B = f"{V4}/rows/learn.jsonl.pre_lego_1111"; assert not os.path.exists(B); shutil.copy2(F, B)
s0 = hashlib.sha256(open(F, "rb").read()).hexdigest(); assert not any(r["id"].startswith("legoil/") for r in L)
open(F + ".tmp", "w").write("".join(json.dumps(r) + "\n" for r in L + twins)); os.replace(F + ".tmp", F)
print(json.dumps(dict(before=s0, after=hashlib.sha256(open(F, "rb").read()).hexdigest(), rows=len(L) + len(twins), backup=B)))
time.sleep(2)
subprocess.run(["python3", f"{E}/tipview.py", "t007c", "show,monitor,judge", "0,1", f"{X}/tip_lrn_all.jsonl"], cwd=V4, capture_output=True)
ids = {r["id"] for r in lego}; T = [q for q in map(json.loads, open(f"{X}/tip_lrn_all.jsonl")) if q["row"] in ids]; assert len(T) == 2 * len(lego)
STOCK = {"min_image_tokens": None, "max_image_tokens": None, "temperature": None}
A = collections.defaultdict(list)
for q in T:
    r, s = q["row"], q["sample"]; tw = "legoil/" + r
    base = dict(mode="read", engine="E", priority=-9, sample=s, seed=crn(r, s))
    A["lego_std_stock"].append(dict(base, key=f"lgss|{r}|{s}", row=r, prefix=None, settings=STOCK))
    A["lego_il_stock"].append(dict(base, key=f"lgis|{r}|{s}", row=tw, prefix=None, settings=STOCK))
    A["lego_std_tip"].append(dict(q, **base, key="lgst|" + q["key"], row=r))
    A["lego_il_tip"].append(dict(q, **base, key="lgit|" + q["key"], row=tw))
for name in ("lego_std_stock", "lego_il_stock", "lego_std_tip", "lego_il_tip"):
    Q = A[name]; assert len({q["key"] for q in Q}) == len(Q)
    p = f"{V4}/q/in/{name}.jsonl"; open(p + ".tmp", "w").write("".join(json.dumps(x) + "\n" for x in Q)); os.replace(p + ".tmp", p)
    open(f"{X}/submit_times.jsonl", "a").write(json.dumps(dict(batch=name, t_submit=time.time(), n=len(Q))) + "\n")
    print(name, len(Q), flush=True); time.sleep(1.1)
