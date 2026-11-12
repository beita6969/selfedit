"""Finalise g004c30's one annotation once the formal server confirm is in (coordinator 11-10 22:08 CST): all 444 JUDGE rows at >= 4
server samples; pooled one-sided p <= .025 and LOBO -> "replicated" (the coordinator adds it to generation 6 as a cell candidate), else
withdrawn = True with d, se, p. Writes RUN/annotate/g004c30_final.json; the running loop applies it at its next save (gen.apply_annotations),
so nothing outside the loop writes state.json. No other state change; the tip and the running generation are untouched.
usage: c30_final.py RUN_DIR CONFIRM_JSON [--dry]"""
import json, os, sys

RUN, CJ = sys.argv[1], sys.argv[2]
DRY = "--dry" in sys.argv
conf = json.load(open(CJ))
P = conf["pooled"]
assert conf["new"] == "t004" and conf["old"] == "t002" and not conf.get("all"), "the confirm must be t004 vs t002 on server reads"
assert conf["rows"] == 444 and conf["min"] >= 4, f"the formal confirm needs all 444 JUDGE rows at >= 4 server samples (have {conf['rows']} at min {conf['min']})"
ok = P["p"] <= .025 and conf["lobo"]
cnum = dict(min_server_samples=conf["min"], rows=conf["rows"], d=P["d"], se=P["se"], p=P["p"], clusters=P["clusters"], tok_ratio=P["tok_ratio"],
            acc_new=P["acc_new"], acc_old=P["acc_old"], lobo=conf["lobo"], macro=conf.get("macro"),
            sources={b: dict(d=s["d"], se=s["se"], p=s["p"], n=s["n"], tok_ratio=s["tok_ratio"]) for b, s in conf["sources"].items()})
num = (f"d {P['d']:+.4f}, se {P['se']:.4f}, p {P['p']:.4f}, LOBO {'holds' if conf['lobo'] else 'fails'}, tokens x{P['tok_ratio']:.3f} "
       f"at {conf['min']} server samples per row on {conf['rows']} JUDGE rows")
if ok:
    A = dict(gen=4, cid="g004c30", set=dict(withdrawn=False, replicated=True, confirm_server=cnum),
             status_append=f"replicated on server reads ({num}); a cell candidate for generation 6 (coordinator)",
             ledger=dict(replicated=True, withdrawn=False, confirm_server=cnum))
else:
    A = dict(gen=4, cid="g004c30", set=dict(withdrawn=True, replicated=False, confirm_server=cnum),
             status_append=f"not replicated on server reads ({num}), withdrawn",
             ledger=dict(replicated=False, withdrawn=True, confirm_server=cnum))
print(("REPLICATED" if ok else "WITHDRAWN") + ": " + A["status_append"])
if not DRY:
    d = os.path.join(RUN, "annotate")
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, "g004c30_final.json")
    assert not os.path.exists(f) and not any(x.startswith("g004c30_final.json.applied") for x in os.listdir(d)), "already finalised"
    json.dump(A, open(f + ".tmp", "w"), indent=1)
    os.rename(f + ".tmp", f)
    print(f"wrote {f}; the running loop applies it at its next save")
