"""1109c fix 11-09: null draws answer at their own seed. CPU only, writes only under OUT.
 1 a draw's queue line carries --seed base + d * DRAW_OFF; a non-draw line keeps the base seed; draws never go to the stock worker
 2 a stale draw file stamped with the base seed (what code-xf cards made) is refused, and a ledger entry pointing to it does not stop the re-ask
 3 a file stamped with the draw seed is read: load_member gives the draw's rows
usage: t_drawseed.py OUT"""
import sys, os, json, re, shutil
N = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [N + "/rsi", N + "/mm", N]
import hboost as H
OUT = sys.argv[1]
assert not os.path.exists(OUT), "OUT is used"
W = "/data/fd/mingdazhang/selfedit/rsi/mm/hb/p1109c"
os.makedirs(OUT)
pre = json.load(open(f"{W}/prereg.json"))
pre["plain"] = True                       # every dir under OUT
json.dump(pre, open(f"{OUT}/prereg.json", "w"))
shutil.copy(f"{W}/amendment.json", f"{OUT}/amendment.json")
H.set_run(OUT); H.HB = f"{OUT}/hb"; H.EVR = OUT
for d in ("jobs", "rows", "claims"):
    os.makedirs(f"{H.HB}/{d}", exist_ok=True)
am = json.load(open(f"{OUT}/amendment.json"))
data = H.Data(None, am["iid"], am.get("ood"), am.get("exclude"))
st = [m for m in pre["seeds"] if m["name"] == "stock"][0]
H.REG["stock"] = st
lines = []
class Q:
    logs = f"{OUT}/logs"; qfile = f"{OUT}/evoq.txt"
    def submit(self, name, cmd): lines.append((name, cmd))
    def failed(self, name): return False
ev = H.Ev(OUT, data); ev.q = Q(); ev.use_worker = True
ev.worker_idle = lambda: True; ev.to_worker = lambda b: lines.append(("WORKER", [i["out"] for i in b]))
rows = list(data.of("select"))[:6]
x4 = H.draw_of(st, 4)
fails = 0
def check(name, ok, info=""):
    global fails
    fails += not ok
    print(("PASS " if ok else "FAIL ") + name, info)
# 2 first: a stale base-seed file in the draw dir with a ledger entry over the rows
d = ev.dir_of(x4); os.makedirs(d, exist_ok=True)
stale = f"{d}/a.rSTALE0000.jsonl"
with open(stale, "w") as o:
    for i in rows:
        for s in (0, 1):
            o.write(json.dumps(dict(id=i, kind="mc", answer="A", sample=s, reasoning="x", tokens=1)) + "\n")
json.dump(dict(H.GEN_KEY, model="stock"), open(stale + ".gen", "w"))
ev.note(d, dict(rows=rows, out=stale, name="hbXX_bstale"))
check("T2_stale_refused", not H.gen_ok(stale), H.draw_key(stale)["seed"])
ev.evaluate_many([(x4, rows), (st, rows)])
seeds = {n: re.search(r"--seed (\d+)", c).group(1) for n, c in lines if n != "WORKER"}
base = int(H.GEN_KEY["seed"])
dl = [n for n, c in lines if n != "WORKER" and "d4_" in json.dumps(json.load(open(c.split()[4])))]
check("T1_draw_line_seed", dl and all(int(seeds[n]) == base + 4 * H.DRAW_OFF for n in dl), (seeds, dl))
check("T1_draws_not_to_worker", all("d4_" not in json.dumps(v) for n, v in lines if n == "WORKER"), [n for n, _ in lines])
check("T2_stale_ledger_does_not_block_reask", bool(dl), "draw rows asked again")
it = json.load(open(next(c for n, c in lines if n in dl).split()[4]))[0]
check("T2_new_out_name", it["out"] != stale and os.path.basename(it["out"]) not in os.listdir(d), it["out"])
# 3 the card's answer at the draw seed is read
with open(it["out"], "w") as o:
    for i in it["ids"]:
        for s in (0, 1):
            o.write(json.dumps(dict(id=i, kind=data.rows[i].get("kind", "mc"), answer="A", sample=s, reasoning="x", tokens=1)) + "\n")
json.dump(dict(H.GEN_KEY, seed=str(base + 4 * H.DRAW_OFF), model="stock"), open(it["out"] + ".gen", "w"))
H.POOL = {i: data.rows[i] for i in data.learn}
v = H.load_member(x4)
check("T3_draw_seed_file_read", set(it["ids"]) <= set(v), (len(v), len(it["ids"])))
print("ALL GREEN" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
