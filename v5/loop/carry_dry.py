"""Dry run of the boundary carry-in merge on a COPY of a run's state (no live writes): scenario A = the tip unchanged since the content the
candidate was read against; B = the previous generation merged X on that content with a smaller contribution_lb; C = X larger.
usage: carry_dry.py LIVE_RUN_DIR TMP_DIR CID CONTRIB_LB"""
import copy, json, os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
import gen  # noqa: E402

LIVE, TMP, CID, CLB = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
base = json.load(open(os.path.join(LIVE, "state.json")))
g_now = max(int(k) for k in base["gens"])
for sc in ("A", "B", "C"):
    d = os.path.join(TMP, sc)
    os.makedirs(os.path.join(d, "annotate"), exist_ok=True)
    st = copy.deepcopy(base)
    if sc in ("B", "C"):                                      # the previous generation merged X on the same content
        G = st["gens"][str(g_now)]
        X = next(c for c in G["cands"] if c.get("entry") and not c["entry"].get("parts"))
        X.update(accepted=True, merged=True, judge=dict(contribution_lb=CLB - 0.01 if sc == "B" else CLB + 0.01))
        old = st["tip"]
        st["tips"].append(old)
        st["tip"] = dict(id=f"t{g_now:03d}", entries=old["entries"] + [X["entry"]], born=g_now, parent=old["id"], from_cand=X["cid"], author=X["author"])
    json.dump(st, open(os.path.join(d, "state.json"), "w"))
    open(os.path.join(d, "ledger.jsonl"), "w").close()
    A = dict(gen=int(CID[1:4]), cid=CID, set=dict(accepted=True, carry_merge=True, carry_in=dict(contribution_lb=CLB, by="DRY RUN")),
             status_append="DRY RUN carry-in", ledger=dict(dry=True))
    json.dump(A, open(os.path.join(d, "annotate", f"{CID}_carry.json"), "w"))
    R = gen.Run(gen.args(["--run", d]))
    R.g = g_now + 1
    R.state["gens"].setdefault(str(R.g), dict(done=[], times={}))
    R.save()                                                  # applies the annotation
    R.carry_merges()
    t = R.state["tip"]
    c = next(c for G in R.state["gens"].values() for c in G.get("cands", []) if c["cid"] == CID)
    print(f"[{sc}] tip {t['id']} parent {t.get('parent')} entries {len(t['entries'])} last {json.dumps(t['entries'][-1], ensure_ascii=False)[:120]}")
    print(f"     {CID}: accepted {c.get('accepted')} merged {c.get('merged')} | {c['status'].split(' | ')[-1][:160]}")
    if sc in ("B", "C"):
        x = next(x for x in R.state["gens"][str(g_now)]["cands"] if x["cid"] == X["cid"])
        print(f"     X {x['cid']}: merged {x.get('merged')} | {x['status'].split(' | ')[-1][:160]}")
    print(f"     carry_merges {R.state.get('carry_merges')}; ledger lines {sum(1 for _ in open(os.path.join(d, 'ledger.jsonl')))}; "
          f"tips/ {sorted(os.listdir(os.path.join(d, 'tips')))}")
    R2 = gen.Run(gen.args(["--run", d]))                       # a second boundary does nothing more
    R2.g = g_now + 2
    R2.state["gens"].setdefault(str(R2.g), dict(done=[], times={}))
    n0 = len(R2.state["tips"])
    R2.carry_merges()
    assert len(R2.state["tips"]) == n0, "second boundary merged again"
print("dry run done")
