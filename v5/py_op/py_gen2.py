"""PY generation 2 (coordinator 11-11 23:32): adjudication by measurement, evaluated AS DEPLOYED on the fixed replicate pairs of cached
base reads (s0,s1), (s2,s3), (s4,s5), (s6,s7) of every learn row of one role.
  replicate value   pair agrees -> its answer; pair disagrees -> the ONE escalation read of that replicate decides (no answer = wrong)
  row value         mean over the four replicates;  tokens likewise (pair + escalation read; generated, and with prompts)
  third reference   pair + base read s(2k+2 mod 8), plurality, ties to the earlier read;  any-of-two = either read of the pair right
  variant vs V1     judge_v5.Judge.score / decide with base = V1 (the Builder's "candidate against another cached system": extra generated
                    tokens are charged at the base frontier's increment; none when the variant uses no more tokens), sign-flip of picture
                    clusters, p_macro / p_gen; raw difference and tokens beside it.
usage: py_gen2.py signal|judge NAME=COMPACT[,COMPACT...] ...   (the arm named V1 is the reference)   [--out FILE] [--decide NAME]"""
import collections
import gzip
import json
import os
import sys

V5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v5")
sys.path.insert(0, V5)
import judge_v5 as JV          # noqa: E402
import policy5 as P            # noqa: E402
import reads5 as R             # noqa: E402
import stage1_data as SD       # noqa: E402
import stage3_settings as S3   # noqa: E402

B = R.BENCHES
OUT = []


def say(s=""):
    OUT.append(s)
    print(s)


def opn(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def load(paths, ids):
    arm = {}
    for p in paths.split(","):
        for l in opn(p):
            x = json.loads(l)
            if x.get("error") or x.get("ok") is None or x["row"] not in ids:
                continue
            arm.setdefault((x["row"], int(x["sample"])), dict(ok=bool(x["ok"]) and x.get("pred") is not None, tokens=float(x["tokens"]), total=float(x["tokens"]) + float(x["prompt_tokens"]),
                                                              none=x.get("pred") is None, runs=x.get("n_runs") or 0, rep=x.get("rep"), hint=x.get("hint"), pred=x.get("pred")))
    return arm


def main():
    role = sys.argv[1]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    dec = sys.argv[sys.argv.index("--decide") + 1] if "--decide" in sys.argv else None
    specs = [a for a in sys.argv[2:] if "=" in a]
    rows, full = SD.rows(), SD.full_rows()
    floor = S3.base(rows, full)
    Js, S, J, pols = S3.judges(rows, floor)
    if role == "signal":
        Jx, ids = Js, S
    else:
        Jx = JV.Judge(rows, {r: floor[r] for r in J}, benches=B, policies=pols, hull_from={p[2] for p in Js.F.vertices}, hull_from_total={p[2] for p in Js.Ft.vertices})
        ids = J
    idset = set(ids)
    arms = {s.split("=", 1)[0]: load(s.split("=", 1)[1], idset) for s in specs}
    tok = lambda x: (x["tokens"], x["tokens"] + float(x.get("prompt_tokens") or 0))
    okb = lambda x: bool(x["ok"]) and P.cls_of(x) != P.NONE
    # ---- replicates
    reps = {}                                   # row -> [(k, a, b, third, disagree, j)]
    for r in ids:
        by = {x["s"]: x for x in floor[r]}
        j, L = 0, []
        for k in range(4):
            a, b, c = by[2 * k], by[2 * k + 1], by[(2 * k + 2) % 8]
            dis = P.cls_of(a) != P.cls_of(b)
            L.append((k, a, b, c, dis, j if dis else None))
            j += dis
        reps[r] = L
    n_esc = sum(d for L in reps.values() for (_, _, _, _, d, _) in L)
    say(f"PY GENERATION 2, {role.upper()} rows ({len(ids)}), deployed replicates: {4 * len(ids)} replicates, {n_esc} escalate ({100 * n_esc / (4 * len(ids)):.1f}%), "
        f"on {sum(any(x[4] for x in L) for L in reps.values())} rows")

    def system(get):
        """get(r, k, a, b, c, j) -> (ok, generated tokens, total tokens) of the escalation of a disagreeing replicate"""
        v, t, T = {}, {}, {}
        for r in ids:
            o = g = G = 0.0
            for k, a, b, c, dis, j in reps[r]:
                ta, tb = tok(a), tok(b)
                if not dis:
                    o += okb(a)
                    g += ta[0] + tb[0]
                    G += ta[1] + tb[1]
                else:
                    e = get(r, k, a, b, c, j)
                    o += e[0]
                    g += ta[0] + tb[0] + e[1]
                    G += ta[1] + tb[1] + e[2]
            v[r], t[r], T[r] = o / 4, g / 4, G / 4
        return dict(value=v, tokens=t, total=T)

    def from_arm(name):
        A = arms[name]
        miss = [(r, j) for r in ids for (_, _, _, _, d, j) in reps[r] if d and (r, j) not in A]
        if miss:
            raise SystemExit(f"{name}: {len(miss)} escalation reads missing (e.g. {miss[0]})")
        return system(lambda r, k, a, b, c, j: (float(A[(r, j)]["ok"]), A[(r, j)]["tokens"], A[(r, j)]["total"]))

    third = system(lambda r, k, a, b, c, j: (float(P._plur([a, b, c])[1]), tok(c)[0], tok(c)[1]))
    anyof = system(lambda r, k, a, b, c, j: (float(okb(a) or okb(b)), 0.0, 0.0))
    one = dict(value={r: Jx.tip_v[r] for r in ids})
    sysv = {n: from_arm(n) for n in arms}
    mm = lambda d: sum(sum(d[r] for r in ids if rows[r]["bench"] == b) / sum(rows[r]["bench"] == b for r in ids) for b in B) / len(B)
    bmv = lambda d, b: sum(d[r] for r in ids if rows[r]["bench"] == b) / sum(rows[r]["bench"] == b for r in ids)
    say(f"{'policy (2 base reads, then on disagreement)':46s} " + " ".join(f"{b[:6]:>6s}" for b in B) + f" {'MACRO':>6s} {'gen x':>6s} {'tot x':>6s}   (x = multiples of one base read)")
    lines = [("a third base read (reference)", third), ("any of the two right (ceiling, information)", anyof)] + [(n + " read decides", sysv[n]) for n in arms]
    for lab, s in lines:
        tk = f"{mm(s['tokens']) / Jx.base_tokens:6.2f} {mm(s['total']) / Jx.base_total:6.2f}" if lab[0] != "a" or "third" in lab else "     -      -"
        if "any of" in lab:
            tk = "     -      -"
        say(f"{lab:46s} " + " ".join(f"{100 * bmv(s['value'], b):6.1f}" for b in B) + f" {100 * mm(s['value']):6.2f} {tk}")
    say(f"{'one base read':46s} " + " ".join(f"{100 * bmv(one['value'], b):6.1f}" for b in B) + f" {100 * mm(one['value']):6.2f}   1.00   1.00")
    # ---- escalations only
    say("escalated replicates only (accuracy; n = replicates):")
    say(f"  {'':10s} {'n':>4s} " + " ".join(f"{n:>7s}" for n in arms) + f" {'third':>7s} {'any-2':>7s} | no answer / ran code / runs per read: "
        + "; ".join(f"{n} {100 * sum(arms[n][(r, j)]['none'] for r in ids for (_, _, _, _, d, j) in reps[r] if d) / max(n_esc, 1):.1f}% / "
                    f"{100 * sum(arms[n][(r, j)]['runs'] > 0 for r in ids for (_, _, _, _, d, j) in reps[r] if d) / max(n_esc, 1):.0f}% / "
                    f"{sum(arms[n][(r, j)]['runs'] for r in ids for (_, _, _, _, d, j) in reps[r] if d) / max(n_esc, 1):.2f}" for n in arms))
    for b in ["ALL"] + list(B):
        E = [(r, k, a, bb, c, j) for r in ids if b in ("ALL", rows[r]["bench"]) for (k, a, bb, c, d, j) in reps[r] if d]
        if not E:
            continue
        say(f"  {b:10s} {len(E):4d} " + " ".join(f"{100 * sum(arms[n][(r, j)]['ok'] for (r, k, a, bb, c, j) in E) / len(E):7.1f}" for n in arms)
            + f" {100 * sum(P._plur([a, bb, c])[1] for (r, k, a, bb, c, j) in E) / len(E):7.1f} {100 * sum(okb(a) or okb(bb) for (r, k, a, bb, c, j) in E) / len(E):7.1f}")
    # ---- which answer did the escalation read give relative to the pair's two answers (description)
    show = lambda q: str(int(q)) if isinstance(q, float) and q == int(q) else str(q)
    hinted = next((n for n in arms if any(v.get("hint") for v in arms[n].values())), None)
    if hinted:
        say("which answer the escalation read gave, relative to the two answers of its pair (count, % right):")
        for n in arms:
            c, okc = collections.Counter(), collections.Counter()
            for r in ids:
                for (_, _, _, _, d, j) in reps[r]:
                    if not d:
                        continue
                    h, x = arms[hinted][(r, j)].get("hint"), arms[n][(r, j)]
                    if not h:
                        kind = "pair without two answers"
                    elif x["pred"] is None:
                        kind = "no answer"
                    else:
                        ps = show(x["pred"])
                        kind = ("first shown" if ps == h[0] else "second shown" if ps == h[1] else "neither") if arms[n][(r, j)].get("hint") else ("one of the two" if ps in h else "neither")
                    c[kind] += 1
                    okc[kind] += x["ok"]
            say(f"  {n:5s} " + "; ".join(f"{k} {v} ({100 * okc[k] / v:.0f}%)" for k, v in sorted(c.items())))
    say("")
    # ---- variants against V1 and against the third-base-read reference
    cands = []
    for n in arms:
        if n == "V1":
            continue
        c = dict(name=f"{n} vs V1", kind="policy", source=[], routed=None, base=sysv["V1"], **sysv[n])
        cands.append(c)
    for n in arms:
        c = dict(name=f"{n} vs third base read", kind="policy", source=[], routed=None, base=third, **sysv[n])
        rs = Jx.score(c)
        e, raw = rs["eq"], rs["raw"]
        say(f"{c['name']:26s} raw {100 * raw['d']:+5.2f} +- {100 * raw['se']:4.2f} (p_gen {raw['p_gen']:.4f}; " + " ".join(f"{b[:5]} {100 * raw['per_bench'][b]['d']:+5.1f}" for b in B) + f") | tokens x{rs['multiple']:.3f} of the reference | charged {100 * e['d']:+5.2f} +- {100 * e['se']:4.2f} p_macro {e['p_macro']:.4f} "
            f"p_gen {e['p_gen']:.4f} | " + " ".join(f"{b[:5]} {100 * e['per_bench'][b]['d']:+5.1f}" for b in B))
    say("")
    recs = Jx.decide([c for c in cands if dec is None or c["name"].startswith(dec + " ")], family="PY generation 2 (K = 1)") if (role == "judge" and dec) else [Jx.score(c) for c in cands]
    for rs in recs:
        e, raw = rs["eq"], rs["raw"]
        say(f"{rs['name']:14s} policy acc {100 * rs['acc']:.2f} vs V1 {100 * mm(sysv['V1']['value']):.2f} | generated tokens x{rs['multiple']:.3f} of V1's | RAW {100 * raw['d']:+5.2f} +- {100 * raw['se']:4.2f} (p_macro {raw['p_macro']:.4f}, p_gen {raw['p_gen']:.4f})")
        say(f"    at equal generated tokens (extra tokens charged at the base frontier, reference {JV.ref_text(rs['ref'])}): {100 * e['d']:+5.2f} +- {100 * e['se']:4.2f}  p_macro {e['p_macro']:.4f}  "
            f"p_gen {e['p_gen']:.4f} (worst without {e['carrier']})  positive {e['npos']}/{e['k']}")
        say("    per benchmark: " + " ".join(f"{b[:5]} {100 * e['per_bench'][b]['d']:+5.1f}+-{100 * e['per_bench'][b]['se']:3.1f}" for b in B) + f" | significantly negative: {e['neg'] if e['neg'] else 'none'}")
        t = rs.get("total") or {}
        if t.get("d") is not None:
            say(f"    with prompts: total tokens x{t['multiple']:.3f} of V1's; charged {100 * t['d']:+5.2f} +- {100 * t['se']:4.2f} p_macro {t['p_macro']:.4f} p_gen {t['p_gen']:.4f} (printed, not gating)")
        if "accept" in rs:
            say(f"    VERDICT: {'ACCEPT' if rs['accept'] else 'REJECT'}: {rs['reason']}")
        elif role == "signal":
            ok = e["d"] > 0 and e["p_macro"] <= 0.25
            say(f"    pre-screen (d > 0 and p_macro <= .25 at equal generated tokens): {'PASS' if ok else 'fail'}")
    if out:
        open(out, "w").write("\n".join(OUT) + "\n")
        JV.dump(dict(recs=recs), out.replace(".txt", ".json"))


if __name__ == "__main__":
    main()
