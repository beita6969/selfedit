"""Orchestrated fusion, separate from the driver: reads main2's answers, writes only under hb/orch/<run>. Lines are filler priority (name _so). argv: run step [--rows 300] [--role show,score]. Steps: plan -> talk lines; combine -> decision lines; read -> tables and demos."""
import collections, glob, json, os, re, sys
A = sys.argv[1:]; sys.argv = ["x"]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "..", "mm"), os.path.join(HERE, "..", "pb")]
import numpy as np
import hboost as h
import orch_program as P_
run, step = A[0], A[1]
opt = lambda k, d: A[A.index(k) + 1] if k in A else d
N, ROLES = int(opt("--rows", 300)), opt("--role", "show,score").split(",")
ORCH = f"{h.HB}/orch/{run}"; os.makedirs(ORCH, exist_ok=True)
W = f"{h.HB}/main2"
h.set_run(W)
D = h.Run(W)
am = h.active_amendment(W, 3)
D.data = D.ev.data = h.Data(D.a.get("limit"), am["iid"], am["ood"], am.get("exclude"))
data = D.data
members = D.state["members"]
JUDGES = {"stock": next(m for m in members if m["name"] == "stock"), "patch": next(m for m in members if m["name"].startswith("r1s5"))}
PATCH_NOPROG = dict(JUDGES["patch"], program=None)

def records(m):
    """id -> sample -> the whole answer record, from the member's own files."""
    out = collections.defaultdict(dict)
    for d in h.member_dirs(m):
        for f in sorted(glob.glob(f"{d}/a.*.jsonl")):
            for l in open(f):
                r = json.loads(l)
                if r.get("sample", 0) < 2:
                    out[r["id"]].setdefault(r.get("sample", 0), r)
    return out

def submit(name_hint, m, rows, sub):
    """Rows for one checkpoint and engine, in 100-row lines, same job-batch command as the driver's."""
    d = f"{ORCH}/{sub}"; os.makedirs(d, exist_ok=True)
    g, names = m["genome"], []
    for k in range(0, len(rows), 100):
        part = rows[k:k + 100]
        tag = h.sha(h.canon(sorted(x["id"] for x in part)) + name_hint)[:10]
        src, out = f"{d}/rows.{tag}.jsonl", f"{d}/a.r{tag}.jsonl"
        if os.path.exists(out):
            continue
        with open(src + ".tmp", "w") as o:
            for x in part:
                o.write(json.dumps(x, ensure_ascii=False) + "\n")
        os.replace(src + ".tmp", src)
        it = dict(kind="eval", model=D.ev.model(g), engine=g["engine"], rows=src, out=out, n=len(part), old=f"none_{tag}")
        name = f"hb{D.ev.tag}_so{h.sha(out)[:12]}"
        h.jdump(f"{h.HB}/jobs/{name}.json", [it])
        D.ev.q.submit(name, f"python -u rsi/hboost.py job-batch {h.HB}/jobs/{name}.json {name} --samples 2 --model {it['model']} --pool {h.IID} " + h.vh.TAIL.format(lib=h.vh.DIRLIB))
        names.append(name)
    return names

def answers_in(sub):
    out = collections.defaultdict(dict)
    for f in sorted(glob.glob(f"{ORCH}/{sub}/a.r*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            out[r["id"]].setdefault(r.get("sample", 0), r)
    return out

def tablerec(votes, ids, cluster_of):
    """Right and n per member and category, and overall, on the given rows; per cluster parts so a row can leave its own picture out."""
    rec = {m: collections.defaultdict(lambda: [0, 0]) for m in votes}
    by_cl = {m: collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0])) for m in votes}
    for m, v in votes.items():
        for i in ids:
            if {0, 1} <= set(v.get(i, {})):
                c, k, x = data.cluster(i), h.vh.key(data.rows[i]), .5 * (bool(v[i][0]["ok"]) + bool(v[i][1]["ok"]))
                rec[m][k][0] += x; rec[m][k][1] += 1
                by_cl[m][c][k][0] += x; by_cl[m][c][k][1] += 1
    return rec, by_cl

def out_of_cluster(rec, by_cl, m, cluster):
    r = {k: tuple(v) for k, v in rec[m].items()}
    for k, v in by_cl[m].get(cluster, {}).items():
        a, b = r.get(k, (0, 0))
        r[k] = (a - v[0], b - v[1])
    return r

def setup():
    votes = {m["name"]: h.load_member(m) for m in members}
    ids = [i for i in data.learn if data.role[i] in ROLES]
    return votes, ids

if step == "plan":
    votes, ids = setup()
    full = {m["name"]: records(m) for m in members}
    rec, by_cl = tablerec(votes, ids, data.cluster)
    def key0(m, i):
        return votes[m].get(i, {}).get(0, {}).get("key")
    disagree = [i for i in ids if len({key0(m, i) for m in votes if 0 in votes[m].get(i, {})} - {None}) >= 2]
    pick = sorted(disagree, key=lambda i: h.sha(f"orch:{i}"))[:N]
    print(f"{len(ids)} rows, {len(disagree)} disagree, taking {len(pick)}")
    plan, per_member = {}, collections.defaultdict(list)
    for i in pick:
        row = dict(rid=i, category=h.vh.key(data.rows[i]), kind=data.rows[i]["kind"])
        c = data.cluster(i)
        have = [m for m in votes if {0, 1} <= set(votes[m].get(i, {})) and key0(m, i) is not None]
        all_ = {}
        recs = {}
        for m in have:
            recs[m] = out_of_cluster(rec, by_cl, m, c)
            all_[m] = (sum(v[0] for v in recs[m].values()), sum(v[1] for v in recs[m].values()))
        ask = P_.route(row, have, recs, all_)
        pos = {m: key0(m, i) for m in ask}
        x = dict(consulted=ask, positions=pos, category=row["category"], agree=len(set(pos.values())) == 1)
        if not x["agree"]:
            for m in P_.who_talks(pos):
                mm = next(z for z in members if z["name"] == m)
                base = h.program_rows(data, mm, [i], render=True)[0]
                if not base:
                    continue
                mine = dict(answer=pos[m], why=P_.short(full[m][i][0]))
                others = [dict(answer=pos[o], why=P_.short(full[o][i][0])) for o in ask if o != m]
                per_member[m].append(dict(base[0], prompt=P_.talk_prompt(base[0]["prompt"], mine, others)))
        plan[i] = x
    h.jdump(f"{ORCH}/plan.json", dict(rows=plan, roles=ROLES))
    names = []
    for m, rows in per_member.items():
        names += submit(f"talk:{m}", next(z for z in members if z["name"] == m), rows, f"talk/{m}")
    print("consulted count", collections.Counter(len(x["consulted"]) for x in plan.values()), "agree after routing", sum(x["agree"] for x in plan.values()))
    print("talk rows", {m: len(v) for m, v in per_member.items()}, "lines", len(names))

def votes_dir(sub):
    out = {}
    for f in sorted(glob.glob(f"{ORCH}/{sub}/a.r*.jsonl")):
        for i, v in h.votes_of(f).items():
            out.setdefault(i, {}).update({k: x for k, x in v.items() if k not in out.get(i, {})})
    return out

if step == "combine":
    plan = h.jload(f"{ORCH}/plan.json")["rows"]
    talk = {m["name"]: answers_in(f"talk/{m['name']}") for m in members}
    tv = {m["name"]: votes_dir(f"talk/{m['name']}") for m in members}
    rows, miss, names = [], 0, {}
    for i, x in plan.items():
        if x["agree"]:
            continue
        finals = []
        for m in x["consulted"]:
            t = talk[m].get(i, {}).get(0)
            if t is None:
                miss += 1
                continue
            k = tv[m].get(i, {}).get(0, {}).get("key")
            finals.append(dict(answer=k if k is not None else "no readable answer", why=P_.short(t), changed=k != x["positions"][m]))
        if finals:
            rows.append(dict(data.rows[i], prompt=P_.decide_prompt(data.rows[i]["prompt"], finals)))
    print("decision rows", len(rows), "missing talk answers", miss)
    for j, jm in (("stock", JUDGES["stock"]), ("patch", PATCH_NOPROG)):
        names[j] = submit(f"judge:{j}", jm, rows, f"judge/{j}")
    print({k: len(v) for k, v in names.items()}, "lines")

if step == "read":
    plan = h.jload(f"{ORCH}/plan.json")["rows"]
    votes, ids = setup()
    full = {m["name"]: records(m) for m in members}
    tv = {m["name"]: votes_dir(f"talk/{m['name']}") for m in members}
    tfull = {m["name"]: answers_in(f"talk/{m['name']}") for m in members}
    jv = {j: votes_dir(f"judge/{j}") for j in ("stock", "patch")}
    jfull = {j: answers_in(f"judge/{j}") for j in ("stock", "patch")}
    vt, tab, Pl, sp, st = D.compute(members)
    F, Foof = st.cur.F, st.cur.F_oof
    fit_pos = {int(k): n for n, k in enumerate(sp.fit)}
    rate = lambda v: .5 * (bool(v[0]["ok"]) + bool(v[1]["ok"])) if v and {0, 1} <= set(v) else None
    def poolr(i, oof=False):
        k = tab.index[i]
        mk = tab.mask[k]
        f = Foof[fit_pos[k]] if oof else F[k]
        return float(np.where(mk, f, -np.inf).argmax() == tab.y[k])
    rows = {}
    for i, x in plan.items():
        cons = x["consulted"]
        first = [votes[m][i] for m in cons]
        o = dict(pool=poolr(i), pool_oof=poolr(i, True) if tab.index[i] in fit_pos else None, stock=rate(votes["stock"].get(i)))
        for m in votes:
            o[m] = rate(votes[m].get(i))
        def maj(vs, tie):
            out = []
            for s in (0, 1):
                c = collections.Counter(v[s]["key"] for v in vs if s in v and v[s]["key"] is not None)
                if not c:
                    out.append(0.); continue
                top = max(c.values())
                win = next((v for v in vs if s in v and v[s]["key"] is not None and c[v[s]["key"]] == top), None)
                out.append(float(win[s]["ok"]))
            return sum(out) / 2
        o["first_vote"] = maj(first, None)
        if x["agree"]:
            o.update(talk_vote=o["first_vote"], judge_stock=o["first_vote"], judge_patch=o["first_vote"], calls=0)
        else:
            t = [tv[m].get(i) for m in cons if tv[m].get(i)]
            if not t or i not in jv["stock"] or i not in jv["patch"]:
                continue
            o.update(talk_vote=maj(t, None), judge_stock=rate(jv["stock"][i]), judge_patch=rate(jv["patch"][i]), calls=len(t) + 1)
        rows[i] = o
    print(f"read {len(rows)} of {len(plan)} rows; agree after routing {sum(plan[i]['agree'] for i in rows)}")
    ks = ["pool", "pool_oof"] + list(votes) + ["first_vote", "talk_vote", "judge_stock", "judge_patch"]
    cl = lambda i: data.cluster(i)
    def boot(a, b, sel, B=2000):
        by = collections.defaultdict(list)
        for i in sel:
            if rows[i].get(a) is not None and rows[i].get(b) is not None:
                by[cl(i)].append(rows[i][a] - rows[i][b])
        if not by:
            return None
        v = list(by.values()); rng = np.random.default_rng(0); nn = sum(map(len, v))
        d = [sum(sum(v[j]) for j in idx) / max(1, sum(len(v[j]) for j in idx)) for idx in (rng.integers(0, len(v), len(v)) for _ in range(B))]
        return round(100 * sum(map(sum, v)) / nn, 1), round(100 * float(np.quantile(d, .05)), 1), round(100 * float(np.quantile(d, .95)), 1), nn
    best = next(n for n in votes if n.startswith("r1s5"))
    L = []
    for tag, sel in (("all", list(rows)), ("no vlmsbiased", [i for i in rows if not i.startswith("vlmsbiased")])):
        L.append(f"== {tag}: {len(sel)} rows")
        L.append("accuracy: " + ", ".join(f"{k} {100 * np.mean([rows[i][k] for i in sel if rows[i].get(k) is not None]):.1f}" for k in ks))
        for a in ("first_vote", "talk_vote", "judge_stock", "judge_patch"):
            L.append(f"{a}: vs pool_oof {boot(a, 'pool_oof', sel)} vs pool {boot(a, 'pool', sel)} vs {best} {boot(a, best, sel)}")
        by = collections.defaultdict(list)
        for i in sel:
            by[i.split('/')[0]].append(i)
        L.append("per bench (n, pool_oof, " + best + ", talk_vote, judge_stock, judge_patch): " + "; ".join(f"{b} {len(v)} " + " ".join(f"{100 * np.mean([rows[i][k] for i in v if rows[i].get(k) is not None]):.0f}" for k in ("pool_oof", best, "talk_vote", "judge_stock", "judge_patch")) for b, v in sorted(by.items())))
    # does talking carry information
    fl = collections.Counter(); perm = collections.defaultdict(collections.Counter)
    for i, o in rows.items():
        if plan[i]["agree"]:
            continue
        for m in plan[i]["consulted"]:
            v, t = votes[m].get(i, {}), tv[m].get(i, {})
            if 0 in v and 0 in t:
                ch = v[0]["key"] != t[0]["key"]
                k = ("changed" if ch else "held", "right" if v[0]["ok"] else "wrong", "right" if t[0]["ok"] else "wrong")
                fl[k] += 1; perm[m][k] += 1
    L.append("talk, sample 0, (changed/held, before, after): " + str(sorted(fl.items())))
    for m, c in perm.items():
        L.append(f"  {m}: " + str(sorted(c.items())))
    dis = [i for i in rows if not plan[i]["agree"] and best in plan[i]["consulted"]]
    pa = [i for i in dis if votes[best][i][0]["key"] != votes["stock"].get(i, {}).get(0, {}).get("key")]
    L.append(f"{best} vs stock differ on {len(pa)} read rows: {best} {100 * np.mean([rows[i][best] for i in pa]):.1f}, stock {100 * np.mean([rows[i]['stock'] or 0 for i in pa]):.1f}, judge_patch {100 * np.mean([rows[i]['judge_patch'] for i in pa]):.1f}, judge_stock {100 * np.mean([rows[i]['judge_stock'] for i in pa]):.1f}")
    disc = [i for i in rows if not plan[i]["agree"]]
    L.append(f"calls per question on disputes {np.mean([rows[i]['calls'] for i in disc]):.2f}; over all read rows {np.mean([rows[i]['calls'] for i in rows]):.2f} (a row runs 2 samples; first answers cached)")
    open(f"{ORCH}/table.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    # demos: where the decision differs from the pool, both ways
    judge = opt("--judge", "patch")
    ch = [i for i in rows if not plan[i]["agree"] and rows[i][f"judge_{judge}"] != rows[i]["pool"]]
    won = [i for i in ch if rows[i][f"judge_{judge}"] > rows[i]["pool"]]; lost = [i for i in ch if rows[i][f"judge_{judge}"] < rows[i]["pool"]]
    print("changes against pool", len(ch), "right", len(won), "wrong", len(lost))
    with open(f"{ORCH}/demos.{judge}.md", "w") as o:
        for tag, ls in (("ORCH RIGHT", won), ("ORCH WRONG", lost)):
            for i in sorted(ls, key=lambda i: h.sha(i))[:int(opt("--demos", 15))]:
                p = plan[i]
                o.write(f"\n\n# {tag} {i} | {p['category']} | gold {data.rows[i]['answer']} | pool {rows[i]['pool']} judge {rows[i]['judge_' + judge]} | {best} {rows[i][best]}\n{data.rows[i]['prompt'][:500]}\n")
                for m in p["consulted"]:
                    v, t = votes[m].get(i, {}).get(0, {}), tv[m].get(i, {}).get(0, {})
                    o.write(f"- {m}: first {v.get('key')} ok {v.get('ok')} -> talk {t.get('key')} ok {t.get('ok')}\n  why: {P_.short(tfull[m].get(i, {}).get(0, {}))[-350:]!r}\n")
                r = jfull[judge].get(i, {}).get(0, {})
                o.write(f"JUDGE {jv[judge].get(i, {}).get(0, {}).get('key')} ok {jv[judge].get(i, {}).get(0, {}).get('ok')}: {P_.short(r)[-600:]!r}\n")
    h.jdump(f"{ORCH}/rows.json", rows)

if step == "routing":
    votes, ids = setup()
    rec, _ = tablerec(votes, ids, data.cluster)
    cats = sorted({k for m in rec for k in rec[m]})
    names = list(votes)
    print("category, n, " + ", ".join(names) + ", consulted")
    all_ = {m: (sum(v[0] for v in rec[m].values()), sum(v[1] for v in rec[m].values())) for m in names}
    for c in cats:
        have = [m for m in names if rec[m].get(c, [0, 0])[1]]
        rs = {m: {k: tuple(v) for k, v in rec[m].items()} for m in names}
        ask = P_.route(dict(category=c), have, rs, all_)
        n = max(rec[m].get(c, [0, 0])[1] for m in names)
        print(c, n, " ".join(f"{100 * rec[m][c][0] / rec[m][c][1]:.0f}/{rec[m][c][1]}" if rec[m].get(c, [0, 0])[1] else "-" for m in names), [a[:6] for a in ask])
