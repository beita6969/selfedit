"""AM23 hand bundle: write the "add" annotation of the synthetic candidate H1 (from h1_build.py's h1_cand.json) with the coordinator's judge_bh
numbers. Accepted -> accepted True, carry_merge True: the next generation boundary merges ALL its entries (gen.carry_merges) when the tip
still has the content H1 was read against (or against the previous generation's merge by contribution_lb). Not accepted -> kept with its
numbers only. The running loop applies it at its next save (gen.apply_annotations; loops from 11-10 00:50 on).
usage: h1_annot.py RUN_DIR H1_CAND_JSON JUDGE_JSON [--note TEXT] [--dry]
JUDGE_JSON: one judge_bh object (or a list holding one named like the candidate) with d, se / se_used, p, bh, accept, lb, contribution_lb, K"""
import json, os, sys

RUN, HC, J = sys.argv[1], json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
DRY = "--dry" in sys.argv
NOTE = sys.argv[sys.argv.index("--note") + 1] if "--note" in sys.argv else None
if isinstance(J, list):
    J = [x for x in J if x.get("name") in (HC["cid"], None)] if len(J) > 1 else J
    assert len(J) == 1, "the judge list must hold exactly one entry for the bundle"
    J = J[0]
if "se" not in J and "se_used" in J:
    J = dict(J, se=J["se_used"])
miss = [k for k in ("d", "se", "p", "accept", "contribution_lb") if k not in J]
assert not miss, f"judge output lacks {miss}"
st = json.load(open(os.path.join(RUN, "state.json")))
assert not any(x["cid"] == HC["cid"] for x in st.get("synthetic", [])), f"{HC['cid']} is in state already"
tips = {t["id"]: t for t in st["tips"] + [st["tip"]]}
assert HC["parent"] in tips, f"parent {HC['parent']} unknown"
acc = bool(J["accept"])
ci = dict(J, by="hand (coordinator judge_bh on the engine table)", rule="AM23 hand bundle", **({"note": NOTE} if NOTE else {}))
num = (f"pooled d {J['d']:+.4f}, se {J['se']:.4f}, p {J['p']:.4f}" + (f" on {J['rows']} rows" if J.get("rows") else "")
       + f", K {J.get('K')}, contribution_lb {J['contribution_lb']:+.4f}")
status = (f"AM23 hand bundle of {len(HC['entry']['parts'])} entries on {HC['parent']}: {num} -> " + ("accepted (merges all entries at the next boundary)"
          if acc else "not accepted") + (f"; {NOTE}" if NOTE else ""))
x = dict(HC, carry_in=ci, accepted=acc, status=status, **({"carry_merge": True} if acc else {}))
A = dict(gen=int(st.get("gen") or 0), add=x, status_append=status, ledger=dict(carry_in=ci, rule="AM23 hand bundle"))
print(json.dumps(dict(A, add=dict(x, entry=f"<{len(x['entry']['parts'])} entries>")), indent=1))
if not DRY:
    d = os.path.join(RUN, "annotate", "add")                # only loops with the "add" annotation read annotate/add/
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, f"{HC['cid']}_add.json")
    assert not os.path.exists(f), "already written"
    json.dump(A, open(f + ".tmp", "w"), indent=1)
    os.rename(f + ".tmp", f)
    print(f"wrote {f}; the running loop applies it at its next save")
