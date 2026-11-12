"""State work after generation 5 was stopped at 21:28 CST (coordinator 11-10): set generation 5 aside, keep or revert t004 by the
confirm, mark offline-vs-server weight-cell comparisons, freeze cut-off gen-4 reads, so run.sh can relaunch generation 5 fresh.
Nothing is deleted: backups are copies, set-aside files are renamed. Branch "revert": revert_pending = [g004c30] (revert_merges makes
t005r = t002's entries at the start of the fresh generation 5); branch "stands": c30 keeps its acceptance, confirmed on server reads.
Branch "pending" (coordinator 11-10 22:08): as revert, but c30 is withdrawn = "pending" while its server replication finishes
(c30_final.py then finalises that one annotation through RUN/annotate/, applied by the running loop); CONFIRM_JSON may be "-".
usage: g5_reset.py RUN_DIR revert|stands|pending CONFIRM_JSON [--dry]"""
import glob, json, os, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
import gen  # noqa: E402

RUN, BR, CJ = sys.argv[1], sys.argv[2], sys.argv[3]
DRY = "--dry" in sys.argv
assert BR in ("revert", "stands", "pending")
TAG = time.strftime("%H%M")
PEND = ("accepted on offline reads (single sample, selected 1 of 4); server replication in progress: interim +1.06 pp, se .94, 273 rows at "
        ">= 4 samples; lineage continues from the last replicated tip t002")
MIXED = ("read on the offline engine against a server reference (regime-mixed; W-null A/A on stock weights -0.11 pp, se 1.11, so no engine "
         "offset detected); kept as recorded, not accepted")
conf = json.load(open(CJ)) if BR != "pending" else None
P = conf["pooled"] if conf else None
cnum = None if not conf else dict(min_server_samples=conf["min"], rows=conf["rows"], d=P["d"], se=P["se"], p=P["p"], clusters=P["clusters"], tok_ratio=P["tok_ratio"],
            acc_new=P["acc_new"], acc_old=P["acc_old"], lobo=conf["lobo"], macro=conf.get("macro"),
            sources={b: dict(d=s["d"], se=s["se"], p=s["p"], n=s["n"], tok_ratio=s["tok_ratio"]) for b, s in conf["sources"].items()})
if conf:
    rule_ok = P["p"] <= .025 and conf["lobo"]
    assert (BR == "stands") == rule_ok, f"branch {BR} does not match the rule (p {P['p']:.4f}, LOBO {conf['lobo']})"
    assert conf["rows"] == 444 and conf["min"] >= 4, "the formal confirm needs all 444 JUDGE rows at >= 4 server samples"
num = None if not conf else f"d {P['d']:+.4f}, se {P['se']:.4f}, p {P['p']:.4f}, LOBO {'holds' if conf['lobo'] else 'fails'}, tokens x{P['tok_ratio']:.3f} at {conf['min']} server samples per row on {conf['rows']} JUDGE rows"

R = gen.Run(gen.args(["--run", RUN]))
st = R.state
acts = []
assert "5" in st["gens"] and "parent" in st["gens"]["5"].get("done", []) and "rung1" not in st["gens"]["5"].get("done", []), "generation 5 is not the stopped partial one"

# 1. backups (copies)
for f in ("state.json", "ledger.jsonl"):
    acts.append(f"copy {f} -> {f}.pre_g5reset_{TAG}")
    if not DRY:
        shutil.copy2(os.path.join(RUN, f), os.path.join(RUN, f"{f}.pre_g5reset_{TAG}"))

# 2. generation 5 aside
g5 = st["gens"].pop("5")
g5["aborted"] = (f"coordinator 11-10 21:28 CST: gen 5 started 20:16 on t004 (code a33f1bbc, --w-cells 2); stopped by gpu stop in rung1 (12 candidates, "
                 f"none decided) because t004's accepted cell g004c30 was read on the offline W engine against a server-read reference; its t004 "
                 f"parent reads are the server-side confirm data; set aside by the builder at {TAG} CST, gen 5 restarts fresh ({BR} branch)")
st.setdefault("gens_aborted", {})["5@2128"] = g5
acts.append(f"gens[5] -> gens_aborted['5@2128'] ({len(g5.get('cands', []))} candidates, done {g5.get('done')})")

# 3. generation-5 files aside (rename)
for f in sorted(glob.glob(os.path.join(RUN, "*_g005*"))):
    if ".aside_" in f:
        continue
    acts.append(f"rename {os.path.basename(f)} -> {os.path.basename(f)}.aside_2128")
    if not DRY:
        os.rename(f, f + ".aside_2128")

# 4. cancel files for generation 5's batches (not the t004 parent reads)
root = R.a.root
for f in sorted(glob.glob(os.path.join(root, "q/in", "g005_*.jsonl"))):
    b = os.path.basename(f)[:-6]
    if b.startswith(("g005_parent_", "g005_parentbg_")):
        continue
    cf = os.path.join(root, "q/cancel", b)
    if not os.path.exists(cf):
        acts.append(f"q/cancel/{b} (was not cancelled)")
        if not DRY:
            open(cf, "w").close()

# 5. generation 4: weight cells read offline vs a server reference; c30 by the branch; cut-off reads frozen
G4 = st["gens"]["4"]
tips = {t["id"]: t for t in st["tips"] + [st["tip"]]}
R.g = 4
cur_tip = st["tip"]
st["tip"] = tips[G4["cands"][0]["parent"]]
R.load_routes(st["tip"]["entries"], R.judge_rows)
R.load_routes([c["entry"] for c in G4["cands"] if c.get("entry")], R.judge_rows)
led_lines = []
for c in G4["cands"]:
    if not R.is_w(c):
        continue
    if c.get("pending_reads"):
        mine = {R.read_req(R.cand_entries(c), r, 0, 1)[0]["key"]: r for r in R.rung_rows(c, 1)}
        hand = [r for k, r in mine.items() if R.io.get(k) is not None or k in R.io.err]
        miss = len(mine) - len(hand)
        if miss:
            c.update(reads_cancelled=True, rows_in_hand=hand, cancelled_reads=miss, cancelled_at=time.time())
            acts.append(f"{c['cid']}: {miss} W reads never landed (cut off by the engine change) -> frozen on the {len(hand)} rows in hand (AM17)")
    eng = R.engine_types(c)
    if not eng.get("mixed"):
        acts.append(f"{c['cid']}: engine field computed {eng}; set mixed = true by ruling (W cell read offline, parent read on servers)")
    c["engine"] = dict(eng, mixed=True)
    if BR == "pending" and c["cid"] == "g004c30":
        c.update(accepted_offline=True, accepted=False, withdrawn="pending")
        c["status"] += " | " + PEND
    elif BR == "pending":
        c["regime_mixed"] = True
        c["status"] += " | " + MIXED
    elif c["cid"] == "g004c30":
        c["confirm_server"] = cnum
        if BR == "revert":
            c.update(accepted_offline=True, accepted=False, withdrawn=True)
            c["status"] += f" | accepted on offline reads, not confirmed on server reads ({num}), withdrawn"
        else:
            c["status"] += f" | accepted on offline reads; confirmed on server reads ({num})"
            c["confirmed_server"] = True
    else:
        c["invalid_comparison"] = True
        c["status"] += " | read on the offline engine against a server reference: invalid comparison"
    if BR == "pending":
        extra = ({"withdrawn": "pending", "accepted_offline": True} if c["cid"] == "g004c30" else {"regime_mixed": True})
    else:
        extra = ({"confirm_server": cnum, "withdrawn": BR == "revert"} if c["cid"] == "g004c30" else {"invalid_comparison": True})
    led_lines.append(dict(id=c["cid"], gen=4, annotation=True, t=time.time(), status=c["status"], accepted=bool(c.get("accepted")),
                          engine=c["engine"], **extra))
    acts.append(f"{c['cid']}: {c['status'].split(' | ')[-1]}")
st["tip"] = cur_tip
if BR in ("revert", "pending"):
    assert st["tip"]["id"] == "t004" and st["tip"].get("from_cand") == "g004c30"
    st["revert_pending"] = ["g004c30"]
    st["revert_reason"] = ("coordinator 11-10 22:08: g004c30 accepted on offline reads, server replication in progress; lineage continues "
                           "from the last replicated tip t002" if BR == "pending" else "g004c30 not confirmed on server reads at 4 samples")
    acts.append("revert_pending = ['g004c30'] (revert_merges makes t005r = t002's entries at the start of generation 5)")

# 6. STOP renamed
if os.path.exists(os.path.join(RUN, "STOP")):
    acts.append(f"rename STOP -> STOP.done_g5reset_{TAG}")
for a in acts:
    print(("DRY  " if DRY else "DONE ") + a)
if not DRY:
    R.save()
    with open(os.path.join(RUN, "ledger.jsonl"), "a") as f:
        for l in led_lines:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    if os.path.exists(os.path.join(RUN, "STOP")):
        os.rename(os.path.join(RUN, "STOP"), os.path.join(RUN, f"STOP.done_g5reset_{TAG}"))
    print(f"saved state: gens {sorted(st['gens'], key=int)}, tip {st['tip']['id']}, revert_pending {st.get('revert_pending')}, "
          f"aborted {sorted(st['gens_aborted'])}; {len(led_lines)} ledger annotation lines")
