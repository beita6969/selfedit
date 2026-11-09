"""Builds harness/frozen/prereg_1109c_final.json (the 1109c-final rules) and c1109deploy/prereg_final.json (the run prereg) from prereg_full.json.
Run on the Mac before upload; code shas are of this tree, the launch checks the server tree against them."""
import hashlib, json, os, sys
T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(T); sys.path.insert(0, T)
GP = "/data/fd/mingdazhang/selfedit/rsi/mm/hb/gate_prompts_v3.jsonl"
FALLBACK = "Count the number of distinct colors in the image before answering, then answer the question."
F = dict(
    _doc="1109c-final (owner 11-09 ~22:00 CST: all bugs fixed, complete version). Frozen before the final run; every rule here is in code at the shas of prereg_final.json.",
    a7="member track primary (4 vs 4 samples, picture-cluster sign flip, Holm over <= stage2_cap 3 candidates); pool line logged only; tripwire null D > delta voids the round's admits; C2 binomial freeze",
    a8=json.load(open("c1109deploy/amendment_a8.json")),
    a10=dict(allocator="ts", design="/Users/admin/selfedit/fable_1109_slot_allocator.md section 2 (exact)", families=["program", "cross", "patch", "gate", "tokens", "verbosity"],
             params="gamma .9, kappa0 1, a0 2, tau0 4 pp, TS cap 6 per family, floor 1 slot per available family",
             prior="debiased best-cell grid zx of the round-1 grid (clip(zx_best - E[max of n N(0,1)], 0, 3)); fit-based prior of the design gives tokens >= 3 slots, zx-based gives 2.7 (replay r1-r3): reported, not tuned",
             reward="stage 2: 100 x member-track D on its rows (w = rows/1600); stage 1 only: 4 x zx (w 210/1600); cost: -10 pp per unit dC above .1; A8 revises a carried proposal's record in place",
             replaces="MetaLevel P A/B (no P window; window r7-8 repeats notes or skills, whichever p is nearest .10)"),
    gate_v3=dict(gate_key="prompts", gate_key_min=20, gate_prompts=GP, recite_min=.95, misfire_max=.01,
                 control_rule="per gate write, a matched control arm written in the same mm_gate job with the same keys: the note is a REAL proposed gate note of a DIFFERENT category from this run's pool of proposed notes (state c1109_gate_notes, all rounds so far incl. this round), the one closest in length to the write's own note, ties broken by a fixed seed (crc32 of the write tag), with its own category's gold answers stripped ('answer is X' clauses; whole-word gold values of 2+ chars). Fallback only if no other category's note exists: '" + FALLBACK + "'. (coordinator 11-09: a real off-category note matches style, length and plausibility, so a win shows the write's own content produces the gain.)",
                 admit="member track on the write's OWN scope rows vs parent (in the Holm family) AND vs its control on the same rows D > 0 with p <= .05 AND mean right rate above chance for every gold value on those rows (chance 1/len(options), else 1/distinct gold values); a gate write is never carried by A8"),
    audit_fixes=["stop rule: no '3 rounds with no accepted member' stop while any A8 carry is alive", "carry log lists all live carries",
                 "stage 1: zx > 0 required; fired_min = min(fired_min, ceil(scope rows / 2)) for narrow members", "A1 from round 1; recite None = n/a",
                 "gate writes read on their own scope rows", "op_patch keeps stock max image tokens 16384", "stock answers copied, not symlinked",
                 "delta: bootstrap sd of pooled reads when fewer than 6 base reads (redraw delta logged)", "one-member pool = that member's plurality answer",
                 "line_report skips cancelled asks"],
    a11=dict(when="between r1 and r2 of p1109c_final_c (coordinator 11-10 ~02:05 CST; driver stopped with gpu stop after r1 closed, resumed from state.json)",
             screen="stage-1 passers: fired >= 3 (fired_min 8 -> 3, scaled min(3, ceil(scope/2))), zx > 0 AND net > 0 on fired rows vs the members' pool; ranked by zx, cap 3. Screen only: stage 2 (A7 member track, Holm over what is read, nulls, tripwire, C2, A8) and every admission threshold unchanged. Reason: r1 had 0 passers while s9 (fired 7, net +3, zx +2.00), s6 (4, +2, +1.34), s3 (6, +3, +0.96) were narrow positive edits blocked by fired_min 8.",
             r1_rejudge="r1s9_ddacd4, r1s6_790728, r1s3_93731d (r1's saved stage-1 table under the new screen, top 3 by zx) go into state c1109_rejudge and are read in r2's stage 2 with r2's Holm family (amendment-2 path: rebuilt from their stored proposals, cached rows not re-asked)",
             early_a7="a candidate passing its own screen during the stage-1 wait gets its A7 paired draw asked at once as a LOW (_se) line through the same evaluate_many request as request_nulls (same member, rows, chunking, out file, line seed), so the later request waits on that same line: answers asked once, identical by construction (tests/t_a11.py); stage 1 does not wait on it",
             stay_max="card code code-xf/rsi/hboost.py STAY_MAX 3600 -> 14400 s (scheduling only; stay_max_4h.sh, commit c7dab7e, applied 2026-11-10 01:58:37 CST, sha d6ed5661 -> 62e037e3)",
             tokens_fix="supervisor + coordinator 11-10: the tokens grid cells all capped max <= 8192 (1024/0, 2048/0, 4096/3600, 8192/2048), downscaling every larger image vs stock max 16384; r1s6 and r1s9 ran at max 4096 min 3600. Fix: cells (16384,1024), (16384,2048), (16384,3600), (16384,4096) plus one labelled downscale cell (8192,2048); a tokens edit without max keeps 16384; no 'or 4096' fallbacks; the proposer menu states stock max 16384 / min 64. r2's grid reruns only the new tokens cells (others from cache); the TS prior is reset and read from r2's stock grid. r1s9 and r1s6 are re-judged as r1s9_ddacd4x / r1s6_790728x: same program, tokens max 16384, min 3600 (disclosed re-read); r1s3 as is",
             samples="unchanged (2 per read, zx kept); owner 11-10: headline reporting is pass@1 (per-sample mean or sample 0), majority vote logged separately, never the headline"),
    grid_reuse="round-1 stock grid from p1109c_grid0 (same tree, fixed ops, stock only, same rows) copied into r001 of the final run; disclosed",
)
A11 = F.pop("a11")      # amendment 11 lives beside the anchor (harness/frozen is the run's anchor_sha: never edited after launch)
json.dump(dict(_doc="amendment 11 of p1109c_final_c (applied between r1 and r2); the anchor harness/frozen stays as launched (d1e4ab7)", **A11), open("c1109deploy/amendment_a11.json", "w"), indent=1)
if not os.environ.get("KEEP_FROZEN"):
    json.dump(F, open("harness/frozen/prereg_1109c_final.json", "w"), indent=1)
p = json.load(open("c1109deploy/prereg_full.json"))
c = p["c1109"]
c.update(fired_min=3, s1_net=True, early_a7=True, carry_seed=[], allocator="ts", gate_key="prompts", gate_key_min=20, gate_prompts=GP, gate_control_note=FALLBACK, gate_recite_min=.95, gate_misfire_max=.01,
         note=c["note"] + "; 1109c-final: harness/frozen/prereg_1109c_final.json (A7 + A8 + A10 TS allocator + gate_v3 with off-category control + audit fixes)")
p["written"] = "2026-11-09 ~23:00 CST, prereg of the 1109c-FINAL run (fresh dir p1109c_final, stock only, no seeds, 96 rows per bench, 20 rounds, batch_rows 50)"
p["rounds"], p["batch_rows"] = 20, 50
p["final"] = "harness/frozen/prereg_1109c_final.json"
for k in ("a8_note",):
    p[k] = p[k].replace("carry_seed r1s11_576f71 plus any r2 candidate", "carry_seed none (fresh run)") if k in p else None
files = sorted(set(p["code_shas_1109c"]) | {"harness/adapters/slot_bandit.py", "harness/frozen/prereg_1109c_final.json", "rsi/vharness.py", "rsi/mm_gate.py"})
p["code_shas_1109c"] = {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in files}
FIELDS = __import__("harness.adapters.hw_round", fromlist=["FIELDS"]).FIELDS
bad = sorted(set(c) - set(FIELDS) - {"note"})
assert not bad, bad
json.dump(p, open("c1109deploy/prereg_final.json", "w"), indent=1)
print("prereg_final ok;", len(files), "files; hw_round", p["code_shas_1109c"]["harness/adapters/hw_round.py"][:12], "hboost", p["code_shas_1109c"]["rsi/hboost.py"][:12])
