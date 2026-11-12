"""AM21 LITE carry-in by hand (coordinator 11-10): write the annotation of one re-judged candidate from the coordinator's judge_bh output,
for the running loop to apply (gen.apply_annotations). Accepted -> accepted True, carry_merge True: the next generation boundary merges it
when it was read against the tip's content (gen.carry_merges), else it stays accepted for the stack. Not accepted -> numbers only.
usage: carry_annot.py RUN_DIR CID JUDGE_JSON [--note TEXT] [--dry]
JUDGE_JSON: {"d":, "se":, "se_used":, "p":, "bh":, "lobo":, "accept":, "lb":, "contribution_lb":, "K":, "family":, "samples":, "rows":, "clusters":}
  or a judge_bh list of such objects with "name" (the one named CID is used); "se" falls back to "se_used".
--note TEXT: extra text appended to the status line and kept as carry_in.note."""
import json, os, sys

RUN, CID, J = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
DRY = "--dry" in sys.argv
NOTE = sys.argv[sys.argv.index("--note") + 1] if "--note" in sys.argv else None
if isinstance(J, list):
    J = [x for x in J if x.get("name") == CID]
    assert len(J) == 1, f"the judge list must hold exactly one entry named {CID}"
    J = J[0]
if "se" not in J and "se_used" in J:
    J = dict(J, se=J["se_used"])
st = json.load(open(os.path.join(RUN, "state.json")))
c = next((c for G in st["gens"].values() for c in G.get("cands", []) if c["cid"] == CID), None)
assert c is not None, f"{CID} not in state"
assert not c.get("accepted") and not c.get("merged"), f"{CID} is already accepted or merged"
need = ("d", "se", "p", "bh", "lobo", "accept", "contribution_lb", "K", "samples")
miss = [k for k in need if k not in J]
assert not miss, f"judge output lacks {miss}"
acc = bool(J["accept"])
num = (f"pooled d {J['d']:+.4f}, se {J['se']:.4f}, p {J['p']:.4f} at {J['samples']} samples per row"
       + (f" on {J['rows']} rows" if J.get("rows") else "") + f", BH at .10 over K {J['K']} of {J.get('family') or c['author'] + '|g' + str(c.get('gen'))}"
       + (f", LOBO {'holds' if J['lobo'] else 'fails'}" if c["entry"].get("cond") == "always" or (c.get("share") or 0) >= 1 else "")
       + f", contribution_lb {J['contribution_lb']:+.4f}")
ci = dict(J, by="hand (coordinator; engine agent reads samples 1-3)", rule="AM21 LITE carry-in", **({"note": NOTE} if NOTE else {}))
A = dict(gen=int(c.get("gen") or int(CID[1:4])), cid=CID,
         set=dict(carry_in=ci, **({"accepted": True, "carry_merge": True} if acc else {})),
         status_append=f"re-judged under AM21 LITE carry-in (by hand, coordinator): {num} -> {'accepted' if acc else 'not accepted'}" + (f"; {NOTE}" if NOTE else ""),
         ledger=dict(carry_in=ci))             # "accepted" comes from the candidate after "set" (loops before 11-10 23:40 refuse it twice)
print(json.dumps(A, indent=1))
if not DRY:
    d = os.path.join(RUN, "annotate")
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, f"{CID}_carry.json")
    assert not os.path.exists(f), "already written"
    json.dump(A, open(f + ".tmp", "w"), indent=1)
    os.rename(f + ".tmp", f)
    print(f"wrote {f}; the running loop applies it at its next save, the next generation boundary merges it if it was accepted")
